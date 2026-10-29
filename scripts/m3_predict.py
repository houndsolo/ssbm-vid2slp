"""
m3_predict.py - run a perception model over WHOLE rendered games and save its guesses of the
game state for every game frame (Stage C step 1: video -> game state table).

Training and m3_analyze.py only look at short clips. For a replay we need every frame, so:
  1. Decode each video once (worker processes, one game each), in pieces of 512 frames.
  2. Feed the model overlapping 32-frame windows (a new window every 16 frames) and keep the
     middle 16 frames of each window: frames in the middle have context on both sides, which
     is how the model was trained to be most accurate.
  3. Turn video frames into GAME frames with the render's frame map (made by reading the
     in-game timer, scripts/m3_frame_maps.py; the same reader will be used on real footage).
     Game frames that never appear in the video (Dolphin skips ~0.2%) are filled from their
     neighbours: positions/% interpolated, everything else copied from the previous frame.

Output: data/m3/pred/<run>/<sha256>.parquet, one row per (game frame >= 0, port), with the
same column names as data/parsed/<sha256>.parquet (character, action, x, y, facing, percent,
stocks, shield, airborne) plus "seen" (False = filled in, not on any video frame).

Run:
    .venv/bin/python scripts/m3_predict.py --run percep_v3 --split validation
    .venv/bin/python scripts/m3_predict.py --run percep_v3 --split train      # ~2,240 games
Games already done are skipped, so it can be restarted.
"""

import argparse
import json
import os
import time

import av
import numpy as np
import pandas as pd
import torch

from vid2slp.perception_data import CAT_FIELDS, REG_FIELDS, RENDERS
from vid2slp.perception_model import Perception

WINDOW = 32      # frames the model sees at once (as in training)
STEP = 16        # a new window every 16 frames; keep the middle 16 of each
SEGMENT = 512    # frames handed from a decoding worker to the GPU at a time
CONTEXT = 16     # extra frames on each side of a piece (= half a window)


class GameSegments(torch.utils.data.IterableDataset):
    """Decodes whole games and hands them over in pieces, so a game never sits in memory whole.

    Yields (game index, keep_from, keep_to, first, frames uint8 [n, 3, H, W], is_last):
    frames are video frames first .. first+n-1; the predictions to KEEP are for video frames
    keep_from .. keep_to-1. Each piece has CONTEXT extra frames on both sides (where they exist)
    so frames near a piece boundary still get windows with context on both sides.
    """

    def __init__(self, shas, size):
        self.shas, self.size = shas, size

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        for g in range(wid, len(self.shas), nw):          # each worker takes every nw-th game
            container = av.open(os.path.join(RENDERS, self.shas[g], "video.mp4"))
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            buf, base, k = [], 0, 0      # buf[i] is video frame base + i; k = piece number
            for frame in container.decode(stream):         # decode order = video frame order
                buf.append(frame.to_ndarray(format="rgb24", width=self.size[0], height=self.size[1]))
                if base + len(buf) >= (k + 1) * SEGMENT + CONTEXT:
                    lo = max(k * SEGMENT - CONTEXT, 0)
                    hi = (k + 1) * SEGMENT + CONTEXT
                    yield g, k * SEGMENT, (k + 1) * SEGMENT, lo, _stack(buf[lo - base:hi - base]), False
                    k += 1
                    cut = max(k * SEGMENT - CONTEXT, 0) - base    # forget frames no longer needed
                    buf, base = buf[cut:], base + cut
            container.close()
            total = base + len(buf)
            lo = max(k * SEGMENT - CONTEXT, 0)
            rest = _stack(buf[lo - base:]) if total > k * SEGMENT else None
            yield g, k * SEGMENT, total, lo, rest, True


def _stack(frames):
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)   # [n, 3, H, W]


def predict_video(model, frames, device):
    """frames uint8 [N, 3, H, W] (consecutive video frames) -> per-frame predictions (numpy dict)."""
    n = len(frames)
    if n < WINDOW:                                    # very short piece: pad with the last frame
        frames = torch.cat([frames, frames[-1:].expand(WINDOW - n, -1, -1, -1)])
        out = predict_video(model, frames, device)
        return {k: v[:n] for k, v in out.items()}
    starts = list(range(0, max(n - WINDOW, 0) + 1, STEP))
    if starts[-1] + WINDOW < n:
        starts.append(n - WINDOW)                     # last window ends exactly at the last frame
    out = {k: np.zeros((n, 2), np.int16) for k in CAT_FIELDS}
    out.update({k: np.zeros((n, 2), np.float32) for k in ["facing", "airborne"] + list(REG_FIELDS)})
    out["action_prob"] = np.zeros((n, 2), np.float32)  # model's confidence in its action guess
    filled_to = 0
    with torch.no_grad():
        for b in range(0, len(starts), 16):           # 16 windows per GPU batch
            chunk = starts[b:b + 16]
            video = torch.stack([frames[s:s + WINDOW] for s in chunk]).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(video)
            res = {k: pred[k].float().argmax(-1).cpu().numpy() for k in CAT_FIELDS}
            res["action_prob"] = pred["action"].float().softmax(-1).amax(-1).cpu().numpy()
            sig = torch.sigmoid(pred["bin"].float()).cpu().numpy()
            res["facing"], res["airborne"] = sig[..., 0], sig[..., 1]
            reg = pred["reg"].float().cpu().numpy()
            for j, (name, scale) in enumerate(REG_FIELDS.items()):
                res[name] = reg[..., j] * scale
            for i, s in enumerate(chunk):
                # keep from where the previous window stopped up to the middle of this one
                # (the very last window keeps everything to the end)
                keep_to = n if s == starts[-1] else s + WINDOW // 2 + STEP // 2
                a = max(filled_to, s)
                if keep_to <= a:
                    continue
                for k in out:
                    out[k][a:keep_to] = res[k][i, a - s:keep_to - s]
                filled_to = keep_to
    return out


def to_game_frames(sha, video_pred):
    """Video-frame predictions -> table indexed by game frame 0..last (both ports)."""
    fmap = np.load(os.path.join(RENDERS, sha, "frame_map.npy"))
    truth = pd.read_parquet(f"data/parsed/{sha}.parquet", columns=["frame", "port"])
    ports = sorted(truth.port.unique())
    last = int(truth.frame.max())
    n = last + 1
    # which video frame shows game frame f (first one if a frame were shown twice)
    video_of = np.full(n, -1)
    for v in range(len(fmap) - 1, -1, -1):
        f = fmap[v]
        if 0 <= f < n:
            video_of[f] = v
    seen = video_of >= 0
    if not seen.any():
        raise RuntimeError("no game frame >= 0 is labelled in this video")
    idx = np.where(seen)[0]
    # previous seen frame for every game frame (frames before the first seen one use the first)
    prev = idx[np.clip(np.searchsorted(idx, np.arange(n), side="right") - 1, 0, len(idx) - 1)]
    rows = []
    for slot, port in enumerate(ports):
        col = {"frame": np.arange(n), "port": np.full(n, port), "seen": seen}
        for k, v in video_pred.items():
            per_frame = v[video_of[prev], slot]                      # copy from previous seen frame
            if k in REG_FIELDS:                                      # numbers: interpolate instead
                per_frame = np.interp(np.arange(n), idx, v[video_of[idx], slot]).astype(np.float32)
            col[k] = per_frame
        rows.append(pd.DataFrame(col))
    table = pd.concat(rows).sort_values(["frame", "port"]).reset_index(drop=True)
    table["facing"] = np.where(table.facing > 0.5, 1.0, -1.0).astype(np.float32)
    table["airborne"] = (table.airborne > 0.5).astype(np.uint8)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--games", default=None, help="file with one sha256 per line (instead of --split)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N games (0 = all)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cfg = json.load(open(f"data/m3/runs/{args.run}/config.json"))
    size = (cfg["width"], cfg["height"])
    out_dir = f"data/m3/pred/{args.run}"
    os.makedirs(out_dir, exist_ok=True)

    # Games: rendered, labelled, not desynced, in the wanted split.
    if args.games:
        shas = open(args.games).read().split()
    else:
        splits = pd.read_csv("data/index/splits.csv")
        wanted = set(splits[splits.split == args.split].sha256)
        shas = []
        for sha in sorted(os.listdir(RENDERS)):
            info_path = os.path.join(RENDERS, sha, "info.json")
            if sha not in wanted or not os.path.exists(info_path):
                continue
            if not os.path.exists(os.path.join(RENDERS, sha, "frame_map.npy")):
                continue
            if json.load(open(info_path)).get("playback_ended_early"):
                continue
            shas.append(sha)
    shas = [s for s in shas if not os.path.exists(f"{out_dir}/{s}.parquet")]
    if args.limit:
        shas = shas[:args.limit]
    print(f"{len(shas)} games to predict with {args.run}", flush=True)

    device = torch.device("cuda:0")
    model = Perception(size=size, max_len=cfg["clip"], pretrained=False).to(device)
    model.load_state_dict(torch.load(f"data/m3/runs/{args.run}/best.pt", map_location=device)["model"])
    model.eval()

    loader = torch.utils.data.DataLoader(GameSegments(shas, size), batch_size=None,
                                         num_workers=args.workers, prefetch_factor=4)
    parts = {}             # game index -> list of (keep_from, predictions) received so far
    t0, done, frames_done = time.time(), 0, 0
    for g, keep_from, keep_to, first, frames, is_last in loader:
        if frames is not None and keep_to > keep_from:
            pred = predict_video(model, frames, device)
            parts.setdefault(g, []).append(
                (keep_from, {k: v[keep_from - first:keep_to - first] for k, v in pred.items()}))
        if not is_last:
            continue
        pieces = sorted(parts.pop(g, []), key=lambda p: p[0])
        video_pred = {k: np.concatenate([p[1][k] for p in pieces]) for k in pieces[0][1]}
        sha = shas[g]
        try:
            n_video = len(np.load(os.path.join(RENDERS, sha, "frame_map.npy")))
            assert len(video_pred["action"]) == n_video, f"{len(video_pred['action'])} predictions, {n_video} video frames"
            table = to_game_frames(sha, video_pred)
            table.to_parquet(f"{out_dir}/{sha}.parquet.tmp", index=False)
            os.replace(f"{out_dir}/{sha}.parquet.tmp", f"{out_dir}/{sha}.parquet")
        except Exception as e:                                   # one bad game must not stop the run
            print(f"{sha[:12]} ERROR {type(e).__name__}: {e}", flush=True)
            continue
        done += 1
        frames_done += n_video
        if done % 10 == 0 or done == len(shas):
            rate = frames_done / (time.time() - t0)
            print(f"  {done}/{len(shas)} games  {rate:.0f} video frames/s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
