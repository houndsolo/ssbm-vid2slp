"""
m3_encoder_check.py - does a perception model read a video compressed by a DIFFERENT encoder
just as well? (2026-10-28: switching renders from x264 on the CPU to NVENC on the V100.)

Takes one rendered game and a second mp4 of the SAME frames (e.g. the same lossless dump
compressed with NVENC), predicts the same evenly spaced clips from both, and prints action
accuracy, x/y error and damage % error for each. If the numbers match, the encoder change
does not hurt the model.

Run:
    .venv/bin/python scripts/m3_encoder_check.py --run percep_v2 --sha <sha256> --other-mp4 path/to/other.mp4
"""

import argparse
import json

import numpy as np
import torch

from vid2slp.perception_data import REG_FIELDS, Game, pts_table, to_tensors
from vid2slp.perception_model import Perception


def score(model, game, size, length, starts, device):
    """Action accuracy, x/y error and % error over the given clip starts."""
    acc, dx, dy, dp, n = 0.0, 0.0, 0.0, 0.0, 0
    reg = list(REG_FIELDS)
    with torch.no_grad():
        for s in starts:
            b = to_tensors(*game.clip(int(s), length, size))
            b = {k: v[None].to(device) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(b["video"])
            v = b["valid"]
            acc += (out["action"][v].argmax(-1) == b["action"][v]).float().sum().item()
            for name, store in (("x", "dx"), ("y", "dy"), ("percent", "dp")):
                j = reg.index(name)
                err = ((out["reg"][v][..., j].float() - b[name][v]).abs() * REG_FIELDS[name]).sum().item()
                if store == "dx":
                    dx += err
                elif store == "dy":
                    dy += err
                else:
                    dp += err
            n += int(v.sum().item()) * 2
    return {"action_acc": acc / n, "x_err": dx / n, "y_err": dy / n, "percent_err": dp / n, "player_frames": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--other-mp4", required=True)
    ap.add_argument("--clips", type=int, default=150)
    args = ap.parse_args()
    cfg = json.load(open(f"data/m3/runs/{args.run}/config.json"))
    device = torch.device("cuda:0")
    size = (cfg["width"], cfg["height"])
    model = Perception(size=size, max_len=cfg["clip"], pretrained=False).to(device)
    model.load_state_dict(torch.load(f"data/m3/runs/{args.run}/best.pt", map_location=device)["model"])
    model.eval()

    game = Game(args.sha)
    starts = np.linspace(game.first_valid, len(game) - cfg["clip"], args.clips).astype(int)
    original = score(model, game, size, cfg["clip"], starts, device)
    game.mp4 = args.other_mp4                     # same labels and clips, other video file
    game.pts = pts_table(args.other_mp4)
    assert len(game.pts) == len(game.labels["valid"]), "the other mp4 has a different number of frames"
    other = score(model, game, size, cfg["clip"], starts, device)
    for name, m in (("original (render's video.mp4)", original), (args.other_mp4, other)):
        print(f"{name}: action {m['action_acc']:.4f}  x err {m['x_err']:.2f}  y err {m['y_err']:.2f}  "
              f"% err {m['percent_err']:.2f}  ({m['player_frames']:,} player-frames)")


if __name__ == "__main__":
    main()
