"""
m4_blast_zones.py - learn every stage's BLAST ZONES (the invisible lines a character dies at)
from real replays, for the Stage C writer.

Why: Slippi Playback's resync forces position, facing, action and damage % every frame, but NOT
the stock count. A stock is only lost when the game itself sees the character cross a blast zone.
In the first end-to-end test (2026-10-29) the replays sometimes missed a death or added one,
because perceived positions near the edge of the screen are a few units off. With the blast zones
known, the writer can put a dying character just past the line (so the game counts the death on
the right frame) and keep everybody else safely inside it.

Method: in training replays, find every frame where a character's action changes into one of the
death actions (0-10: DEAD_DOWN, DEAD_LEFT, DEAD_RIGHT, DEAD_UP, star KO, ...). The position on that
frame is just past the blast zone. Per stage and side, the blast zone is the death position
closest to the stage (e.g. bottom = the HIGHEST y among bottom deaths). Slow deaths land exactly
ON the line, so the exact line value shows up many times; to ignore a rare odd death (e.g. one
Final Destination death at x 220 instead of 246) we only count positions seen at least 3 times.
We also print the furthest position anybody reached WITHOUT dying (respawning counts as alive, so
top/bottom are not a clean check; left/right are).

Output: data/m4/blast_zones.json  {stage_id: {"left", "right", "top", "bottom", "deaths"}}

Run:
    .venv/bin/python scripts/m4_blast_zones.py
"""

import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

DEAD_DOWN, DEAD_LEFT, DEAD_RIGHT = 0, 1, 2
DEAD_UP_ACTIONS = (3, 4, 5, 6, 7, 8, 9, 10)     # normal top KO, star KOs, screen KOs
DEATH_ACTIONS = (DEAD_DOWN, DEAD_LEFT, DEAD_RIGHT) + DEAD_UP_ACTIONS


def closest_repeated(values, side):
    """Closest-to-the-stage death position among positions (rounded to 0.01) seen >= 3 times.
    side = +1 when the line is the LARGEST death value (left, bottom), -1 when it is the smallest."""
    counts = values.round(2).value_counts()
    repeated = counts[counts >= 3].index
    return float(repeated.max() if side > 0 else repeated.min())


def one_game(job):
    sha, stage = job
    t = pd.read_parquet(f"data/parsed/{sha}.parquet", columns=["frame", "port", "action", "x", "y"])
    t = t[t.frame >= 0].sort_values(["port", "frame"])
    deaths, alive = [], []
    for port, d in t.groupby("port"):
        action = d.action.to_numpy().astype(int)
        dead = np.isin(action, DEATH_ACTIONS)
        start = np.where(dead[1:] & ~dead[:-1])[0] + 1          # first frame of each death
        for i in start:
            deaths.append((stage, action[i], float(d.x.iloc[i]), float(d.y.iloc[i])))
        live = d[~dead]
        if len(live):
            alive.append((stage, live.x.min(), live.x.max(), live.y.min(), live.y.max()))
    return deaths, alive


def main():
    splits = pd.read_csv("data/index/splits.csv")
    stage_ids = pd.read_csv("data/index/slp_index.csv").drop_duplicates("sha256").set_index("sha256").stage_id
    train = splits[splits.split == "train"]
    jobs = [(s, int(stage_ids[s])) for s in train.sha256]
    deaths, alive = [], []
    with Pool(12) as pool:
        for d, a in pool.imap_unordered(one_game, jobs, chunksize=8):
            deaths += d
            alive += a
    deaths = pd.DataFrame(deaths, columns=["stage", "action", "x", "y"])
    alive = pd.DataFrame(alive, columns=["stage", "xmin", "xmax", "ymin", "ymax"])

    zones = {}
    for stage, d in deaths.groupby("stage"):
        a = alive[alive.stage == stage]
        z = {
            "left": closest_repeated(d[d.action == DEAD_LEFT].x, +1),
            "right": closest_repeated(d[d.action == DEAD_RIGHT].x, -1),
            "bottom": closest_repeated(d[d.action == DEAD_DOWN].y, +1),
            "top": closest_repeated(d[d.action.isin(DEAD_UP_ACTIONS)].y, -1),
            "deaths": int(len(d)),
        }
        closest = (d[d.action == DEAD_LEFT].x.max(), d[d.action == DEAD_RIGHT].x.min(),
                   d[d.action == DEAD_DOWN].y.max(), d[d.action.isin(DEAD_UP_ACTIONS)].y.min())
        zones[str(stage)] = z
        # sanity check: nobody alive should ever be beyond these lines
        print(f"stage {stage:3d}: {len(d):5d} deaths  "
              f"left {z['left']:8.2f} (alive min x {a.xmin.min():8.2f})  "
              f"right {z['right']:7.2f} (alive max x {a.xmax.max():7.2f})  "
              f"bottom {z['bottom']:8.2f} (alive min y {a.ymin.min():8.2f})  "
              f"top {z['top']:7.2f} (alive max y {a.ymax.max():7.2f})")
        print(f"           closest deaths: left {closest[0]:.2f} right {closest[1]:.2f} "
              f"bottom {closest[2]:.2f} top {closest[3]:.2f}")
    json.dump(zones, open("data/m4/blast_zones.json", "w"), indent=1)
    print("wrote data/m4/blast_zones.json")


if __name__ == "__main__":
    main()
