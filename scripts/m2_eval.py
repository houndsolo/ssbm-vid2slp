"""
m2_eval.py - score a trained inverse-dynamics checkpoint on a split, in detail.

Use --split validation while developing. Use --split test ONCE, at the end, on the single
checkpoint chosen on validation; that is the number we report (CLAUDE.md rules).

Prints and saves (to the run folder, eval_<split>.json):
  - all metrics from src/vid2slp/id_metrics.py, next to the baselines on the same split
  - per-button press F1 (exact frame and within 2 frames)
  - frame_all_correct broken down by the character being predicted

Run:
    .venv/bin/python scripts/m2_eval.py --run base_v1 --split validation
    .venv/bin/python scripts/m2_eval.py --run base_v1 --split test
Another dataset (e.g. perceived states, scripts/m2_build_perceived.py):
    .venv/bin/python scripts/m2_eval.py --run visible_v1 --data data/m2_perc --split validation
    .venv/bin/python scripts/m2_eval.py --run visible_v1 --data data/m2_perc --split validation_truth
"""

import argparse
import json
import os

import numpy as np
import torch

from vid2slp.id_data import BUTTONS, CATEGORICAL, load_split
from vid2slp.id_metrics import HEADLINE, axis_zone, cstick_dir, score, trigger_cat
from vid2slp.id_model import build_from_config, predict_split

DATA = "data/m2"
# internal character ids (as stored in post-frame state) for the common characters
INTERNAL_NAMES = {1: "Fox", 22: "Falco", 18: "Marth", 2: "Captain Falcon", 25: "Ganondorf",
                  7: "Sheik", 9: "Peach", 15: "Jigglypuff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="validation", help="validation, test (or validation_truth in data/m2_perc)")
    ap.add_argument("--data", default="", help="dataset folder (default: the one the run was trained on)")
    ap.add_argument("--checkpoint", default="best.pt")
    args = ap.parse_args()
    run_dir = f"{DATA}/runs/{args.run}"

    ck = torch.load(f"{run_dir}/{args.checkpoint}", map_location="cuda:0")
    cfg = ck["config"]
    data_dir = args.data or cfg.get("data", DATA)  # default: the dataset this checkpoint was trained on
    split = load_split(data_dir, args.split, mmap=False)
    model = build_from_config(cfg, split["num"].shape[-1]).to("cuda:0")
    model.load_state_dict(ck["model"])
    pred_tgt, pred_btn = predict_split(model, split, "cuda:0", cfg["window"])
    m = score(split["tgt"], pred_tgt, split["btn"], pred_btn, split["offsets"])
    # baselines exist only for the standard splits of the standard datasets; otherwise show "-"
    base = json.load(open(f"{DATA}/baselines.json")).get(args.split) if not args.data else None
    if base is None:
        nan = {k: float("nan") for k in m}
        base = {"neutral": nan, "lookup": nan}

    print(f"run {args.run} ({args.checkpoint}, step {ck['step']}) on {args.split}: {m['frames']:,} player-frames\n")
    print(f"{'metric':24s} {'neutral':>9s} {'lookup':>9s} {'model':>9s}")
    for k in HEADLINE + ["stick_y_exact", "stick_x_within2", "buttons_all_exact"]:
        print(f"{k:24s} {base['neutral'][k]:9.4f} {base['lookup'][k]:9.4f} {m[k]:9.4f}")
    print(f"\n{'button':8s} {'held F1':>8s} {'press F1 exact':>15s} {'press F1 +-2':>13s}   (lookup +-2)")
    for b in BUTTONS:
        print(f"{b:8s} {m[f'{b}_held_f1']:8.3f} {m[f'{b}_press_f1']:15.3f} {m[f'{b}_press_f1_tol2']:13.3f}"
              f"   ({base['lookup'][f'{b}_press_f1_tol2']:.3f})")

    # per-character frame_all_correct (frame-level, so no press-timing metrics here)
    char_col = [c for c, _ in CATEGORICAL].index("character")
    chars = split["cat"][:, :, char_col]
    t, p = split["tgt"].astype(np.int32) - 80, pred_tgt.astype(np.int32) - 80
    ok = ((axis_zone(t[..., 0]) == axis_zone(p[..., 0])) & (axis_zone(t[..., 1]) == axis_zone(p[..., 1]))
          & (cstick_dir(t[..., 2], t[..., 3]) == cstick_dir(p[..., 2], p[..., 3]))
          & (trigger_cat(t[..., 4] + 80) == trigger_cat(p[..., 4] + 80)) & (split["btn"] == pred_btn))
    per_char = {}
    print(f"\n{'character':16s} {'player-frames':>14s} {'frame_all_correct':>18s}")
    for cid, name in INTERNAL_NAMES.items():
        mask = chars == cid
        if mask.sum() >= 10000:
            per_char[name] = {"frames": int(mask.sum()), "frame_all_correct": float(ok[mask].mean())}
            print(f"{name:16s} {int(mask.sum()):14,d} {ok[mask].mean():18.4f}")

    out = {"run": args.run, "checkpoint": args.checkpoint, "step": ck["step"], "split": args.split,
           "metrics": m, "per_character": per_char}
    out["data"] = data_dir
    tag = args.split if not args.data else f"{os.path.basename(args.data.rstrip('/'))}_{args.split}"
    json.dump(out, open(f"{run_dir}/eval_{tag}.json", "w"), indent=1)
    print(f"\nwrote {run_dir}/eval_{tag}.json")


if __name__ == "__main__":
    main()
