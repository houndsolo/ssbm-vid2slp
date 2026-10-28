"""
m2_di_analysis.py - how well does the inverse-dynamics model (M2) get the defender's stick
during HITLAG, where DI and SDI happen? (owner's question 2026-10-27)

Background (Melee mechanics, in short):
  - Hitlag: when a hit connects, both players freeze for a few frames.
  - SDI: during the freeze, each fresh flick of the stick nudges the defender a few units.
    The defender is otherwise frozen, so a position change on a freeze frame IS an SDI nudge
    (or the final automatic nudge, ASDI, at the end of the freeze).
  - DI: the stick held on the LAST freeze frame tilts the launch angle. Only the part of
    the stick perpendicular (sideways) to the launch direction counts.
  Exactly which frame the game reads for DI (last hitlag frame or the frame after, in the
  .slp's numbering) is NOT verified here, so both are reported.

What is measured, on the validation games of the M2 dataset, for the player being hit
(action state is one of the DAMAGE_* states 75-91, with hitlag > 0):
  1. all frames of all players            (for context)
  2. defender freeze frames               (stick exact, stick zone, 8-way direction)
  3. "SDI frames": freeze frames (not the first of a freeze) where the defender moved
                                          (did the model point the stick the same way?)
  4. the DI frame (last freeze frame before a launch) and the frame after:
       - stick exact / zone / 8-way direction
       - perpendicular part of the stick relative to the launch direction: mean error, and
         how often the model DI'd to the same SIDE (when the real DI was clearly sideways)
     Launch direction = knockback velocity on the first frame after the freeze.

Run:
    .venv/bin/python scripts/m2_di_analysis.py --run feat2_v4
Output: printed, and saved to data/m2/runs/<run>/di_analysis_<split>.json
"""

import argparse
import json

import numpy as np
import torch

from vid2slp.id_data import CATEGORICAL, FEATURE_SETS, load_split
from vid2slp.id_metrics import axis_zone
from vid2slp.id_model import build_from_config, predict_split

DAMAGE_ACTIONS = range(75, 92)   # DAMAGE_HIGH_1 ... DAMAGE_FLY_ROLL
MOVE_THRESHOLD = 0.5             # world units: smaller position changes are ignored as noise
SIDEWAYS_THRESHOLD = 0.3         # |perpendicular stick| above this counts as "clearly DI'd sideways"


def direction8(x, y):
    """Stick (-80..80 ints) -> 0 = inside the deadzone, else 1..8 = which of 8 directions (45° each)."""
    angle = np.degrees(np.arctan2(y, x)) % 360
    sector = (np.round(angle / 45).astype(int) % 8) + 1
    return np.where(np.hypot(x, y) < 0.2875 * 80, 0, sector)   # 0.2875 = Melee's stick deadzone


def stick_scores(true, pred):
    """true/pred: [K, 2] stick x, y in -80..80. Returns a dict of simple accuracies."""
    if len(true) == 0:
        return {"frames": 0}
    exact = (true == pred).all(1)
    zone = (axis_zone(true[:, 0]) == axis_zone(pred[:, 0])) & (axis_zone(true[:, 1]) == axis_zone(pred[:, 1]))
    d8 = direction8(true[:, 0], true[:, 1]) == direction8(pred[:, 0], pred[:, 1])
    return {"frames": int(len(true)), "stick_exact": float(exact.mean()),
            "stick_zone": float(zone.mean()), "direction8": float(d8.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--checkpoint", default="best.pt")
    args = ap.parse_args()
    run_dir = f"data/m2/runs/{args.run}"

    ck = torch.load(f"{run_dir}/{args.checkpoint}", map_location="cuda:0")
    cfg = ck["config"]
    split = load_split(cfg["data"], args.split, mmap=False)
    model = build_from_config(cfg, split["num"].shape[-1]).to("cuda:0")
    model.load_state_dict(ck["model"])
    pred_tgt, _ = predict_split(model, split, "cuda:0", cfg["window"])

    # Unscale the numeric features we need (the arrays store value / scale).
    names = [c for c, _ in FEATURE_SETS["v2" if "v2" in cfg["data"] else "v1"]]
    scales = dict(FEATURE_SETS["v2" if "v2" in cfg["data"] else "v1"])

    def feat(name):
        return split["num"][:, :, names.index(name)] * scales[name]

    hitlag = feat("hitlag")
    x, y = feat("x"), feat("y")
    kb_x, kb_y = feat("vel_knockback_x"), feat("vel_knockback_y")
    action = split["cat"][:, :, [c for c, _ in CATEGORICAL].index("action")]
    true_stick = split["tgt"][:, :, :2].astype(np.int32) - 80
    pred_stick = pred_tgt[:, :, :2].astype(np.int32) - 80

    # Frames that start a new game must not be treated as "the frame after" the previous game.
    n = len(hitlag)
    game_start = np.zeros(n, bool)
    game_start[split["offsets"][:-1]] = True
    last_of_game = np.zeros(n, bool)
    last_of_game[split["offsets"][1:] - 1] = True

    in_freeze = (hitlag > 0) & np.isin(action, list(DAMAGE_ACTIONS))      # [N, 2] defender in hitlag
    prev_freeze = np.vstack([np.zeros((1, 2), bool), in_freeze[:-1]]) & ~game_start[:, None]
    moved = np.hypot(np.diff(x, axis=0, prepend=x[:1]), np.diff(y, axis=0, prepend=y[:1])) > MOVE_THRESHOLD
    sdi = in_freeze & prev_freeze & moved                                 # not the first freeze frame
    next_freeze = np.vstack([in_freeze[1:], np.zeros((1, 2), bool)]) & ~last_of_game[:, None]
    last_freeze = in_freeze & ~next_freeze                                # last frame of each freeze

    results = {"run": args.run, "split": args.split, "step": ck["step"]}
    results["all_frames"] = stick_scores(true_stick.reshape(-1, 2), pred_stick.reshape(-1, 2))
    results["freeze_frames"] = stick_scores(true_stick[in_freeze], pred_stick[in_freeze])
    results["sdi_frames"] = stick_scores(true_stick[sdi], pred_stick[sdi])

    # DI frame candidates: last freeze frame (offset 0) and the frame after (offset 1).
    # Launch direction: knockback velocity on the frame after the freeze; keep real launches only.
    rows, ports = np.nonzero(last_freeze)
    after = rows + 1
    keep = after < n
    rows, ports, after = rows[keep], ports[keep], after[keep]
    launch = np.stack([kb_x[after, ports], kb_y[after, ports]], 1)
    speed = np.hypot(launch[:, 0], launch[:, 1])
    keep = speed > 0.5
    rows, ports, launch, speed = rows[keep], ports[keep], launch[keep], speed[keep]
    unit = launch / speed[:, None]
    perp = np.stack([-unit[:, 1], unit[:, 0]], 1)                          # sideways direction
    results["launches"] = int(len(rows))
    for offset in (0, 1):
        r = np.minimum(rows + offset, n - 1)
        t, p = true_stick[r, ports], pred_stick[r, ports]
        s = stick_scores(t, p)
        t_perp = (t / 80.0 * perp).sum(1)
        p_perp = (p / 80.0 * perp).sum(1)
        sideways = np.abs(t_perp) > SIDEWAYS_THRESHOLD
        s["sideways_part_mean_error"] = float(np.abs(t_perp - p_perp).mean())
        s["clearly_sideways_share"] = float(sideways.mean())
        s["same_side_when_sideways"] = float((np.sign(t_perp[sideways]) == np.sign(p_perp[sideways])).mean())
        results[f"di_frame_offset{offset}"] = s

    print(f"run {args.run} ({args.checkpoint}, step {ck['step']}) on {args.split}")
    for k in ["all_frames", "freeze_frames", "sdi_frames", "di_frame_offset0", "di_frame_offset1"]:
        s = results[k]
        line = f"{k:18s} frames {s['frames']:>10,d}  stick exact {s['stick_exact']:.3f}  zone {s['stick_zone']:.3f}  " \
               f"8-way dir {s['direction8']:.3f}"
        if "sideways_part_mean_error" in s:
            line += (f"  sideways err {s['sideways_part_mean_error']:.3f}  same side {s['same_side_when_sideways']:.3f}"
                     f" (of the {s['clearly_sideways_share']:.0%} clearly sideways)")
        print(line)
    print(f"launches analysed: {results['launches']:,}")
    json.dump(results, open(f"{run_dir}/di_analysis_{args.split}.json", "w"), indent=1)
    print(f"wrote {run_dir}/di_analysis_{args.split}.json")


if __name__ == "__main__":
    main()
