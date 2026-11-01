"""
m3_eval_clips.py - score saved perception checkpoints on the validation clips m3_train.py uses
(normal renders and, with --kinds, extra render kinds such as port-moved "pt"), so an older model
can be compared with a newer one on exactly the same clips.

Run:
    .venv/bin/python scripts/m3_eval_clips.py --runs percep_v4cd percep_v5pt --kinds pt
    .venv/bin/python scripts/m3_eval_clips.py --runs percep_v6soft --kinds nn --paired
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3_train import FixedClips, evaluate, rendered_games    # noqa: E402  (scripts/m3_train.py)
from vid2slp.perception_data import LevelledClips, SoftenedClips           # noqa: E402
from vid2slp.perception_model import Perception             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--kinds", default="pt", help="extra render kinds, comma separated ('' = none)")
    ap.add_argument("--soften", default="", help="also score softened copies of the normal clips, e.g. 0.6,0.5 (perception_data.soften)")
    ap.add_argument("--paired", action="store_true",
                    help="for each extra kind, also score the NORMAL renders of the same games (normal@<kind>), "
                         "so the two can be compared game for game")
    ap.add_argument("--levels", action="store_true",
                    help="also score the normal clips at Genesis 7's TV levels, black 10 white 232 (perception_data.LevelledClips)")
    ap.add_argument("--batch", type=int, default=2)        # small: the GPU is shared
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    size, clip, per_game = (384, 320), 32, 20               # the m3_train.py defaults
    sets = {"normal": rendered_games("validation")}
    for k in filter(None, args.kinds.split(",")):
        sets[k] = rendered_games("validation", k)
        if args.paired:
            normal = set(sets["normal"])
            sets[f"normal@{k}"] = [s for s in (key.split(":", 1)[1] for key in sets[k]) if s in normal]
            sets[k] = [f"{k}:{s}" for s in sets[f"normal@{k}"]]   # only games that have both
    datasets = {name: FixedClips(shas, clip, size, per_game) for name, shas in sets.items()}
    for f in filter(None, args.soften.split(",")):
        datasets[f"soft{f}"] = SoftenedClips(datasets["normal"], float(f))
    if args.levels:
        datasets["tv"] = LevelledClips(datasets["normal"])
    loaders = {name: torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers)
               for name, ds in datasets.items()}
    device = "cuda"
    for run in args.runs:
        model = Perception(size=size, max_len=clip).to(device)
        model.load_state_dict(torch.load(f"data/m3/runs/{run}/best.pt", map_location=device)["model"])
        for name, loader in loaders.items():
            m = evaluate(model, loader, device)
            print(f"{run:14s} {name:9s} " + "  ".join(f"{k} {v:.4f}" for k, v in m.items()), flush=True)


if __name__ == "__main__":
    main()
