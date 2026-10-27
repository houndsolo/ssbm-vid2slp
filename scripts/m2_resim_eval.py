"""
m2_resim_eval.py - the REAL test of inverse dynamics: replay predicted inputs in the game.

For each chosen game:
  1. Take the original .slp and replace BOTH players' inputs (frames >= 0) with:
       oracle  - the true inputs, but passed through our target encoding + id_to_slp
                 (must replay identically; otherwise our encoding loses information)
       model   - the trained model's predicted inputs
       neutral - no inputs at all (lower bound; should diverge almost immediately)
  2. Play each file in Slippi Playback Dolphin, which simulates the game from those inputs
     and writes a regenerated replay (src/vid2slp/playback.py).
  3. Compare regenerated vs original game states frame by frame.

Metrics per game and variant:
  first_divergence  - first frame where x, y, action state or percent of either player differs.
                      Melee is chaotic: after one difference, everything later drifts.
                      Very strict: an analog stick value off by 1/80 gives a tiny position
                      difference and counts as divergence even if the action is the same.
  first_action_divergence - first frame where either player's ACTION STATE differs
                      (closer to "behaves differently")
  action_match      - share of frames (both players) whose action state still matches
  mean_abs_dx       - mean |x difference| in game units over the compared frames

Splice mode (--splice 600,1200,...): a whole-game replay only tells us when the FIRST mistake
happens. To sample many starting points, "model@k" uses the TRUE inputs up to frame k (so the
game is exactly right at k) and the model's inputs from k on, for --horizon frames. The score
is sync_frames = how many frames after k the replay stays identical (capped at the horizon).

Uses VALIDATION games while developing (test games only for the final report).

Run (in tmux; each game takes ~(frames/60 + 10) s per variant, 4 run in parallel):
    .venv/bin/python scripts/m2_resim_eval.py --run base_v1 --games 12 --end 3600
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from vid2slp.id_data import load_split
from vid2slp.id_model import build_from_config, predict_split
from vid2slp.id_to_slp import write_predicted_slp
from vid2slp.playback import regenerate
from vid2slp.slp_parse import parse_replay

DATA = "data/m2"


def compare(orig, regen, end):
    o = orig[(orig.frame >= 0) & (orig.frame <= end)].set_index(["frame", "port"])
    r = regen[(regen.frame >= 0) & (regen.frame <= end)].set_index(["frame", "port"])
    common = o.index.intersection(r.index)
    o, r = o.loc[common], r.loc[common]
    diff = np.zeros(len(common), bool)
    for c in ["x", "y", "action", "percent"]:
        diff |= o[c].to_numpy() != r[c].to_numpy()
    frames = np.array([f for f, _ in common])
    act_diff = o.action.to_numpy() != r.action.to_numpy()
    return {
        "frames_compared": int(len(set(frames))),
        "first_divergence": int(frames[diff].min()) if diff.any() else None,
        "first_action_divergence": int(frames[act_diff].min()) if act_diff.any() else None,
        "action_match": float(np.mean(o.action.to_numpy() == r.action.to_numpy())),
        "mean_abs_dx": float(np.mean(np.abs(o.x.to_numpy() - r.x.to_numpy()))),
    }


def sub_split(split, g):
    """Arrays of game g only, as a one-game split."""
    a, b = int(split["offsets"][g]), int(split["offsets"][g + 1])
    return {"num": split["num"][a:b], "cat": split["cat"][a:b], "tgt": split["tgt"][a:b],
            "btn": split["btn"][a:b], "offsets": np.array([0, b - a])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--end", type=int, default=3600, help="last frame to simulate (3600 = 1 minute)")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splice", default="", help="comma list of frames k for model@k variants")
    ap.add_argument("--horizon", type=int, default=600, help="frames simulated after each splice point")
    ap.add_argument("--resync", action="store_true",
                    help="turn on Slippi Playback's resync, which corrects drift using the states stored "
                         "in the .slp (here: the original game's states)")
    ap.add_argument("--variants", default="oracle,model,neutral",
                    help="comma list; 'oracle' alone needs no trained model (checks the encoding)")
    args = ap.parse_args()
    wanted = args.variants.split(",")
    out_root = f"{DATA}/runs/{args.run}/resim_{args.split}{'_resync' if args.resync else ''}"
    shutil.rmtree(out_root, ignore_errors=True)
    os.makedirs(out_root)

    data_dir = DATA
    if "model" in wanted:
        data_dir = torch.load(f"{DATA}/runs/{args.run}/best.pt", map_location="cpu")["config"].get("data", DATA)
    split = load_split(data_dir, args.split, mmap=False)
    games_df = pd.read_csv("data/index/splits.csv").set_index("sha256")
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(len(split["sha256"]), args.games, replace=False))

    ck, model = {"step": None}, None
    if "model" in wanted:
        ck = torch.load(f"{DATA}/runs/{args.run}/best.pt", map_location="cuda:0")
        cfg = ck["config"]
        model = build_from_config(cfg, split["num"].shape[-1]).to("cuda:0")
        model.load_state_dict(ck["model"])

    jobs = []
    for g in chosen:
        sha = split["sha256"][g]
        path = games_df.loc[sha, "path"]
        one = sub_split(split, g)
        orig, _ = parse_replay(path)
        ports = sorted(orig.port.unique())
        neutral_tgt = np.zeros_like(one["tgt"])
        neutral_tgt[..., :4] = 80
        variants = {"oracle": (one["tgt"], one["btn"]),
                    "neutral": (neutral_tgt, np.zeros_like(one["btn"]))}
        ends = {name: args.end for name in variants}
        if model is not None:
            pred_tgt, pred_btn = predict_split(model, one, "cuda:0", cfg["window"])
            variants["model"] = (pred_tgt, pred_btn)
            ends["model"] = args.end
            for k in [int(x) for x in args.splice.split(",") if x]:
                if k + args.horizon >= len(one["tgt"]):
                    continue  # game too short for this splice point
                tgt, btn = one["tgt"].copy(), one["btn"].copy()
                tgt[k:], btn[k:] = pred_tgt[k:], pred_btn[k:]
                variants[f"model@{k}"] = (tgt, btn)
                ends[f"model@{k}"] = k + args.horizon
        for name, (tgt, btn) in variants.items():
            if name.split("@")[0] not in wanted:
                continue
            slp = f"{out_root}/{sha[:12]}_{name.replace('@', '_at_')}.slp"
            write_predicted_slp(path, slp, tgt, btn, ports)
            jobs.append((sha, path, name, slp, ends[name]))

    def run(job):
        sha, path, name, slp, end = job
        try:
            regen_path, seconds = regenerate(slp, slp[:-4] + "_run", -123, end, resync=args.resync)
            orig, _ = parse_replay(path)
            regen, _ = parse_replay(regen_path)
            row = {"sha256": sha, "variant": name, "seconds": round(seconds, 1), **compare(orig, regen, end)}
            if "@" in name:
                k = int(name.split("@")[1])
                after = compare(orig[orig.frame >= k], regen[regen.frame >= k], end)
                row["action_match_after"] = after["action_match"]
                row["mean_abs_dx_after"] = after["mean_abs_dx"]
                fd, fa = row["first_divergence"], row["first_action_divergence"]
                row["splice"] = k
                row["sync_frames"] = (end - k + 1) if fd is None else fd - k
                row["action_sync_frames"] = (end - k + 1) if fa is None else fa - k
            return row
        except Exception as e:
            return {"sha256": sha, "variant": name, "error": str(e)[:300]}

    with ThreadPoolExecutor(args.parallel) as pool:
        rows = list(pool.map(run, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(f"{out_root}/results.csv", index=False)

    print(f"\n{args.games} {args.split} games, frames 0..{args.end}, run {args.run} (step {ck['step']})"
          f"{'  [RESYNC ON]' if args.resync else ''}")
    ok = df[df.get("error", pd.Series(index=df.index, dtype=object)).isna()]
    for name in [v for v in ("oracle", "model", "neutral") if v in wanted]:
        part = ok[ok.variant == name]
        fd = part.first_divergence.fillna(args.end + 1)
        print(f"  {name:8s} games {len(part):3d}  never diverged {int(part.first_divergence.isna().sum()):3d}  "
              f"first divergence: median {fd.median():6.0f}  min {fd.min():6.0f}  max {fd.max():6.0f} frames   "
              f"action match {part.action_match.mean():.3f}   mean |dx| {part.mean_abs_dx.mean():6.1f}")
    if "splice" in ok:
        sp = ok[ok.variant.str.startswith("model@")]
        if len(sp):
            s_ = sp.sync_frames
            print(f"  model@k  splices {len(sp):3d}  frames in sync after switching to predicted inputs: "
                  f"median {s_.median():.0f}  p25 {s_.quantile(.25):.0f}  p75 {s_.quantile(.75):.0f}  "
                  f"survived all {args.horizon}: {(s_ > args.horizon).mean():.0%}  "
                  f"(<= 10 frames: {(s_ <= 10).mean():.0%})")
            print(f"  model@k  after the switch: action states match {sp.action_match_after.mean():.3f} of frames, "
                  f"mean |dx| {sp.mean_abs_dx_after.mean():.2f}")
            a_ = sp.action_sync_frames
            print(f"  model@k  frames until first ACTION-state difference: median {a_.median():.0f}  "
                  f"p25 {a_.quantile(.25):.0f}  p75 {a_.quantile(.75):.0f}  "
                  f"survived all {args.horizon}: {(a_ > args.horizon).mean():.0%}")
    if len(ok) < len(df):
        print("  errors:", df[~df.index.isin(ok.index)][["sha256", "variant", "error"]].to_dict("records"))
    json.dump({"run": args.run, "step": ck["step"], "split": args.split, "end": args.end,
               "rows": rows}, open(f"{out_root}/results.json", "w"), indent=1)
    print("wrote", f"{out_root}/results.csv")


if __name__ == "__main__":
    main()
