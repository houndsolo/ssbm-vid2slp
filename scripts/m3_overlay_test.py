"""
m3_overlay_test.py - how much does a stream overlay alone hurt perception?

Why: on the first real video (YouTube EAqS1j1oL2o, a Coinbox stream) perception was badly wrong
(both players called Fox, stock counts jumping). Real footage differs from our renders in several
ways at once (overlay, compression, widescreen camera). There is no replay for that video, so we
can't score it. Instead: paste THAT stream's overlay onto our own renders, whose true states we
know, and score perception there. If the score falls a lot, the overlay is (part of) the cause.

The overlay is taken from the shrunk stream (data/real/<id>/full_584.mp4, made by
real_prepare.py): the per-pixel median over ~400 gameplay frames, in the regions where the stream
draws on top of the game (found by looking at which pixels hardly change, look/std.png):
  - the scoreboard band at the top (only its static pixels, so the game's timer stays visible)
  - the webcam (bottom right), the chat box (bottom, right of the damage display)
  - the round viewer badges on the right edge

Steps:
  1. For each test game: decode the render, paste the overlay, encode to
     data/real/_overlay_<id>/game<k>/video.mp4 (+ the render's frame_map.npy).
  2. m3_predict.py --real _overlay_<id> (run separately).
  3. --score: compare with the true states (data/parsed), and with the same games without overlay.

Run:
    .venv/bin/python scripts/m3_overlay_test.py --make --id EAqS1j1oL2o --games data/m4/test_games_ucf084.txt --n 6
    .venv/bin/python scripts/m3_predict.py --run percep_v4cd --real _overlay_EAqS1j1oL2o
    .venv/bin/python scripts/m3_overlay_test.py --score --id EAqS1j1oL2o --run percep_v4cd
"""

import argparse
import json
import os
import shutil
import subprocess

import av
import numpy as np
import pandas as pd

RENDERS = "data/m3/renders"


def overlay_mask(std):
    """True where the stream draws its overlay (584 x 480 picture)."""
    h, w = std.shape
    y, x = np.mgrid[0:h, 0:w]
    top = (y < 65) & (std < 25)                       # scoreboard: static pixels only
    badges = (x >= 555) & (y >= 195) & (y <= 300)
    webcam = (x >= 528) & (y >= 342)
    chat = (x >= 360) & (x < 530) & (y >= 418)
    return top | badges | webcam | chat


def make(args):
    stats = np.load(f"data/real/{args.id}/look/overlay_stats.npz")
    mask = overlay_mask(stats["std"])
    overlay = stats["median"]
    out_root = f"data/real/_overlay_{args.id}"
    shas = open(args.games).read().split()[:args.n]
    for k, sha in enumerate(shas, start=1):
        out = f"{out_root}/game{k}"
        os.makedirs(out, exist_ok=True)
        shutil.copy(f"{RENDERS}/{sha}/frame_map.npy", f"{out}/frame_map.npy")
        json.dump({"sha256": sha}, open(f"{out}/info.json", "w"))
        # decode the render, paste the overlay, pipe raw frames into ffmpeg (x264 CRF 23 like renders)
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", "584x480", "-r", "60", "-i", "-", "-c:v", "libx264", "-preset", "medium",
                                "-crf", "23", "-pix_fmt", "yuv420p", f"{out}/video.mp4"], stdin=subprocess.PIPE)
        container = av.open(f"{RENDERS}/{sha}/video.mp4")
        for f in container.decode(video=0):
            img = f.to_ndarray(format="rgb24")
            img[mask] = overlay[mask]
            enc.stdin.write(img.tobytes())
        container.close()
        enc.stdin.close()
        enc.wait()
        print(f"game{k} = {sha[:12]}", flush=True)


def score_table(pred, sha):
    """Per-frame accuracy of a perception table against the true states, frames 0..3600."""
    t = pd.read_parquet(f"data/parsed/{sha}.parquet")
    m = pred.merge(t, on=["frame", "port"], suffixes=("_p", ""))
    m = m[(m.frame >= 0) & (m.frame <= 3600)]
    return {"action": (m.action_p == m.action).mean(),
            "character": (m.character_p == m.character).mean(),
            "stocks": (m.stocks_p == m.stocks).mean(),
            "x_err_median": (m.x_p - m.x).abs().median(),
            "percent_err_median": (m.percent_p - m.percent).abs().median()}


def score(args):
    out_root = f"data/real/_overlay_{args.id}"
    rows = []
    for g in sorted(os.listdir(out_root)):
        if not g.startswith("game"):
            continue
        sha = json.load(open(f"{out_root}/{g}/info.json"))["sha256"]
        with_overlay = pd.read_parquet(f"{out_root}/pred_{args.run}/{g}.parquet")
        # the same games without overlay (predicted with --countdown; frames 0+ are scored)
        clean = pd.read_parquet(f"data/m3/pred/{args.run}_countdown/{sha}.parquet")
        rows.append({"game": sha[:12], "overlay": True, **score_table(with_overlay, sha)})
        rows.append({"game": sha[:12], "overlay": False, **score_table(clean, sha)})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print(res.groupby("overlay").mean(numeric_only=True).round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--games", default="data/m4/test_games_ucf084.txt")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--run", default="percep_v4cd")
    args = ap.parse_args()
    if args.make:
        make(args)
    if args.score:
        score(args)


if __name__ == "__main__":
    main()
