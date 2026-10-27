"""
m3_render.py - render replays to video for milestone 3 (perception), with exact frame labels.

For each chosen game:
  1. Playback Dolphin plays the whole replay and dumps every displayed frame losslessly
     (FFV1 .avi, temporary; ~5 MB per second of footage).
  2. The in-game timer is read on every frame of that lossless video by a small digit
     network (src/vid2slp/hud_timer.py), which gives each video frame's exact game frame.
     (scripts/m3_frame_maps.py can redo this later for all games, e.g. after retraining it.)
     Why not just count frames: Dolphin skips displaying some game frames (deterministically,
     measured 2026-10-27), so counting drifts by a few frames over a game.
  3. The video is compressed to H.264 (CRF 23, ~16 MB per minute) -> video.mp4.
  4. Saved next to it:
       - timer_crops.npz: the timer area of every frame, lossless, so the frame map can be
         recomputed with a better timer reader later without re-rendering;
       - frame_map.npy: one int per video frame, its game frame, or -1 when unknown
         (e.g. the countdown, which all shows 08:00 00);
       - info.json: counts and timings.
The lossless .avi and Dolphin's work folder are deleted afterwards.

Labels (character state per game frame) are not copied: they are already in
data/parsed/<sha256>.parquet from milestone 1, keyed by the same game frame numbers.

Which games: a seeded random sample of each split (train / validation / test, the same
frozen split as everything else), so test games stay unseen.

Run (inside tmux; resumable, finished games are skipped):
    .venv/bin/python scripts/m3_render.py --split validation --count 50 --parallel 3
    .venv/bin/python scripts/m3_render.py --split train --count 400 --parallel 3
    add --gpu 0 to draw on the GT 1030 (much faster; same pictures, checked 2026-10-27)
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from multiprocessing import Pool

import av
import numpy as np
import pandas as pd
import torch

from vid2slp import hud_timer
from vid2slp.playback import regenerate

OUT = "data/m3/renders"
WORK = "data/m3/work"
CRF = 23


def timer_crops_from_video(avi):
    """The timer area of every frame of the lossless video, as one uint8 array [frames, h, w, 3]."""
    # .copy(): a crop is otherwise a view that keeps the whole frame in memory (~6 GB per game)
    crops = [hud_timer.timer_crop(f.to_ndarray(format="rgb24")).copy() for f in av.open(avi).decode(video=0)]
    return np.stack(crops)


def render_one(job):
    sha, slp, last_frame, start_seconds, gpu = job
    out = os.path.join(OUT, sha)
    if os.path.exists(os.path.join(out, "info.json")):
        return sha, "skipped"
    work = os.path.join(WORK, sha)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    try:
        regenerate(slp, work, end=last_frame, timeout=3600, stall_seconds=30, gpu=gpu,
                   dump_video={"codec": "ffv1", "fmt": "avi", "kbps": 25000})
        avi = os.path.join(work, "user", "Dump", "Frames", "framedump0.avi")
        t_render = time.time() - t0
        crops = timer_crops_from_video(avi)
        np.savez_compressed(os.path.join(out, "timer_crops.npz"), crops=crops)
        torch.set_num_threads(2)  # the network runs on the CPU here; don't fight the renders
        fmap, rejected, shown_start = hud_timer.frame_map_net(crops, last_frame, start_seconds,
                                                             hud_timer.load_digit_net())
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", avi, "-c:v", "libx264", "-preset", "medium",
                        "-crf", str(CRF), "-pix_fmt", "yuv420p", os.path.join(out, "video.mp4")], check=True)
    except Exception as e:  # keep going with the other games; the error is recorded
        json.dump({"sha256": sha, "error": str(e)}, open(os.path.join(out, "error.json"), "w"))
        shutil.rmtree(work, ignore_errors=True)
        return sha, f"ERROR {e}"
    np.save(os.path.join(out, "frame_map.npy"), fmap)
    labeled = fmap[fmap >= 0]
    increasing = bool((np.diff(labeled) > 0).all())  # must hold, else something was misread
    info = {
        "sha256": sha, "replay": slp, "last_frame": int(last_frame), "timer_start_seconds": int(shown_start),
        "timer_start_seconds_replay": int(start_seconds),  # can differ: see hud_timer.frame_map_net
        "video_frames": int(len(fmap)), "labeled_frames": int(len(labeled)),
        "timer_rejected_reads": int(rejected),
        "reader": "digitnet",
        # game frames between the first and last labelled one that never appear in the video
        "game_frames_skipped_inside": int(labeled.max() - labeled.min() + 1 - len(labeled)) if len(labeled) else None,
        "strictly_increasing": increasing,
        "render_seconds": round(t_render, 1), "total_seconds": round(time.time() - t0, 1),
        "mp4_bytes": os.path.getsize(os.path.join(out, "video.mp4")), "crf": CRF,
    }
    json.dump(info, open(os.path.join(out, "info.json"), "w"), indent=1)
    shutil.rmtree(work, ignore_errors=True)
    return sha, (f"{info['video_frames']} video frames, {info['labeled_frames']} labeled, "
                 f"increasing={increasing}, {info['total_seconds']:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "validation", "test"])
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--gpu", type=int, default=None,
                    help="draw on this NVIDIA GPU (nvidia-smi number; 0 = GT 1030) instead of the CPU")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)

    manifest = pd.read_csv("data/parsed/manifest.csv")
    index = pd.read_csv("data/index/slp_index.csv")[["sha256", "path", "last_frame"]]
    games = manifest[manifest.split == args.split].merge(index, on="sha256", suffixes=("_m", ""))
    if "error" in games:
        games = games[games["error"].isna()]
    # shuffle with the seed, then take the first `count`: a bigger --count later keeps the same first games
    games = games.sort_values("sha256").sample(frac=1, random_state=args.seed).head(args.count)
    import peppi_py
    jobs = []
    for row in games.itertuples():
        start_seconds = peppi_py.read_slippi(row.path, skip_frames=True).start.timer
        jobs.append((row.sha256, row.path, int(row.last_frame), int(start_seconds), args.gpu))
    # record which games were chosen (with the seed) so the selection is reproducible
    pd.DataFrame([j[:4] for j in jobs], columns=["sha256", "path", "last_frame", "timer_start_seconds"]).to_csv(
        f"data/m3/selection_{args.split}_seed{args.seed}_n{args.count}.csv", index=False)
    print(f"{args.split}: rendering {len(jobs)} games with {args.parallel} in parallel", flush=True)
    t0 = time.time()
    with Pool(args.parallel) as pool:
        for i, (sha, msg) in enumerate(pool.imap_unordered(render_one, jobs), 1):
            print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(jobs)} {sha[:12]} {msg}", flush=True)
    print(f"done in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
