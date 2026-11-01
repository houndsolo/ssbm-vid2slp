"""
m3_widescreen_test.py - does the widescreen picture alone break perception?

Why: on the first real video perception called both players Fox and got stocks wrong. A stream
overlay alone does not do that (scripts/m3_overlay_test.py: characters and stocks stayed 100 %).
Many streams use Slippi's "Widescreen 16:9" code, which our renders never had. Here we render
validation games WITH widescreen (src/vid2slp/playback.py, widescreen=True: 16:9, drawn at 2x
and dumped at 1878 x 1056, like a high-resolution stream capture), then treat each one exactly
like a real video (scripts/real_prepare.py: fit the 4:3 game area, shrink to 584 x 480, read
the timer, cut), run perception, and score it against the true states, which we know.

Result (2026-11-01, docs/experiments.md): widescreen itself is harmless. With the exact crop
(the 584:480 middle at full height) perception scores the same as on normal renders; the
first failure came from a crop fitted 8 % too short (fixed in real_prepare.fit_crop).

Steps:
  --render: N games (default: the first 6 of data/m4/test_games_ucf084.txt, the same as the
            overlay test), frames up to --end, -> data/real/_ws_<sha12>/ (source.mp4 + game1/)
  then:     m3_predict.py --run percep_v4cd --real _ws_<sha12>   (for each)
  --score:  accuracy vs data/parsed, next to the clean renders' accuracy.

Run:
    .venv/bin/python scripts/m3_widescreen_test.py --render
    for d in data/real/_ws_*; do .venv/bin/python scripts/m3_predict.py --run percep_v4cd --real $(basename $d); done
    .venv/bin/python scripts/m3_widescreen_test.py --score
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_prepare                                     # noqa: E402  (scripts/real_prepare.py)
from m3_overlay_test import score_table                 # noqa: E402
from vid2slp import hud_timer                           # noqa: E402
from vid2slp.playback import regenerate                 # noqa: E402

WORK = "data/m3/work"


def render(args):
    splits = pd.read_csv("data/index/splits.csv").set_index("sha256")
    shas = open(args.games).read().split()[:args.n]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = hud_timer.load_digit_net(device)
    for sha in shas:
        d = f"data/real/_ws_{sha[:12]}"
        if os.path.exists(f"{d}/games.json"):
            continue
        os.makedirs(d, exist_ok=True)
        json.dump({"sha256": sha}, open(f"{d}/sha.json", "w"))
        work = f"{WORK}/ws_{sha[:12]}"
        shutil.rmtree(work, ignore_errors=True)
        regenerate(splits.loc[sha, "path"], work, end=args.end, timeout=3600, stall_seconds=60, gpu=0,
                   widescreen=True, dump_video={"codec": "ffv1", "fmt": "avi", "kbps": 25000})
        avi = f"{work}/user/Dump/Frames/framedump0.avi"
        # the "downloaded video": high quality H.264 like a good stream upload
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", avi, "-c:v", "libx264", "-preset", "fast",
                        "-crf", "18", "-pix_fmt", "yuv420p", f"{d}/source.mp4"], check=True)
        shutil.rmtree(work, ignore_errors=True)
        n_games, share = real_prepare.prepare(os.path.basename(d), net=net, device=device)
        print(f"{sha[:12]}: {n_games} game(s), timer read on {share:.1%} of frames", flush=True)


def score(args):
    rows = []
    for d in sorted(glob.glob(f"data/real/{args.prefix}_*")):
        sha = json.load(open(f"{d}/sha.json"))["sha256"]
        crop = json.load(open(f"{d}/crop.json"))
        pred = f"{d}/pred_{args.run}_countdown" if os.path.isdir(f"{d}/pred_{args.run}_countdown") else f"{d}/pred_{args.run}"
        ws = pd.read_parquet(f"{pred}/game1.parquet")
        clean = pd.read_parquet(f"data/m3/pred/{args.run}_countdown/{sha}.parquet")
        rows.append({"game": sha[:12], "widescreen": True, "fit": crop["fit_correlation"], **score_table(ws, sha)})
        rows.append({"game": sha[:12], "widescreen": False, "fit": None, **score_table(clean, sha)})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print(res.groupby("widescreen").mean(numeric_only=True).round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--games", default="data/m4/test_games_ucf084.txt")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--end", type=int, default=3700, help="last game frame rendered")
    ap.add_argument("--run", default="percep_v4cd")
    ap.add_argument("--prefix", default="_ws",
                    help="which folders to score: _ws (crop fitted by real_prepare) or _wsfix (re-cut with "
                         "the exact crop 297,0,1285,1056: the 584:480 middle at full height, 2026-11-01)")
    args = ap.parse_args()
    if args.render:
        render(args)
    if args.score:
        score(args)


if __name__ == "__main__":
    main()
