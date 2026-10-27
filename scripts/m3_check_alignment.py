"""
m3_check_alignment.py - independent check that rendered videos and their frame labels line up.

The frame labels (frame_map.npy) come from reading the TIMER. This check uses something
else on screen: the damage % under each character's portrait. Whenever the replay says a
player's damage went up at game frame f, the HUD digits should start changing in the video
frame that frame_map says is game frame f (offset 0), the same for every hit in every game.
(The same test was used on the old project's clips in the audit: 94.4% at offset 0.)

Only games where the players use controller ports 1 and 2 are checked (the HUD position
depends on the port; these two boxes are measured on a 584x480 render).

Run:
    .venv/bin/python scripts/m3_check_alignment.py --parallel 4   # all rendered games
Prints one line per game (a game far from offset 0 is marked SUSPICIOUS), then the totals.
"""

import argparse
import glob
import multiprocessing
import os
from collections import Counter

import av
import numpy as np
import pandas as pd

from vid2slp.perception_data import game_labels

PERCENT_BOX = {0: (50, 400, 150, 450), 1: (186, 400, 291, 450)}   # x0, y0, x1, y1 (ports 1, 2)
CHANGE_PIXELS = 15
SEARCH = range(-4, 7)


def hud_masks(mp4):
    """White-pixel masks of both percent boxes for every video frame."""
    out = {0: [], 1: []}
    for f in av.open(mp4).decode(video=0):
        img = f.to_ndarray(format="rgb24").astype(np.int16)
        for slot, (x0, y0, x1, y1) in PERCENT_BOX.items():
            c = img[y0:y1, x0:x1]
            out[slot].append((c.max(2) > 150) & (c.max(2) - c.min(2) < 40))
    return {k: np.array(v) for k, v in out.items()}


def changed(m, i):
    return int((m[i] != m[i - 1]).sum()) if 0 < i < len(m) else 0


def check_game(sha):
    """Counter of HUD onset offsets for every hit in one game (None if not ports 1+2)."""
    manifest = pd.read_csv("data/parsed/manifest.csv").set_index("sha256")
    df = pd.read_parquet(manifest.loc[sha, "parquet"], columns=["port"])
    if sorted(df.port.unique()) != [0, 1]:
        return sha, None
    lab = game_labels(sha)
    masks = hud_masks(f"data/m3/renders/{sha}/video.mp4")
    counts = Counter()
    gf, valid = lab["game_frame"], lab["valid"]
    for slot in (0, 1):
        m, pct, stk = masks[slot], lab["percent"][:, slot], lab["stocks"][:, slot]
        for i in range(5, len(gf) - 7):
            # a hit at video frame i, with no skipped game frames around it
            if not (valid[i - 5:i + 7].all() and gf[i + 6] - gf[i - 5] == 11):
                continue
            if pct[i] > pct[i - 1] + 0.5 and stk[i] == stk[i - 1]:
                if any(changed(m, i + k) >= CHANGE_PIXELS for k in (-4, -3, -2)):
                    continue  # HUD still moving from an earlier hit
                onset = next((k for k in SEARCH if changed(m, i + k) >= CHANGE_PIXELS), None)
                counts[onset] += 1
    return sha, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=1, help="games checked at the same time")
    args = ap.parse_args()
    shas = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob("data/m3/renders/*/info.json"))
    counts, games = Counter(), 0
    with multiprocessing.Pool(args.parallel) as pool:
        for sha, c in pool.imap_unordered(check_game, shas):
            if c is None:
                continue
            games += 1
            counts.update(c)
            n = sum(c.values())
            good = c[0] + c[-1]
            # per game: a game whose video drifts away from its replay (desync) shows up here
            flag = "  <-- SUSPICIOUS" if n >= 5 and good / n < 0.7 else ""
            print(f"{sha[:12]}: {n} hits, {good} at offset 0/-1{flag}", flush=True)
    total = sum(counts.values())
    print(f"{games} games (ports 1+2), {total} hits")
    for k, n in sorted(counts.items(), key=lambda kv: (kv[0] is None, kv[0] if kv[0] is not None else 0)):
        print(f"  HUD starts changing at offset {k}: {n} ({n / total:.3f})")


if __name__ == "__main__":
    main()
