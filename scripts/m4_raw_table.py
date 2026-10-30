"""
m4_raw_table.py - for every PROCESSED stick position, the RAW stick bytes most often behind it.

Why: the input model predicts processed stick values (what the game uses), but UCF 0.84 also
reads the raw bytes (e.g. dashback and "1.0 cardinal" checks). Many raw positions give the same
processed value: the game scales the raw (x, y) onto the circle of radius 80 if it is outside,
truncates, and then zeroes values below 23 (formula checked on 426,224 frames, 2026-10-30):
    processed = deadzone23(trunc(raw * min(1, 80 / |raw|))) / 80
Writing raw = processed * 80 loses information: with the TRUE processed values, the 12 test games
replay with action match 0.631 instead of 0.783 with the true raw bytes (2026-10-30).
This table gives, for each processed (x, y), the most common raw (x, y) in real UCF 0.84 games.

Run:    .venv/bin/python scripts/m4_raw_table.py
Output: data/m4/raw_given_processed.json  {"px,py": [rx, ry]}  (px, py in -80..80)
"""

import json

import numpy as np
import pandas as pd

from vid2slp.matchups import wanted_games
from vid2slp.slp_codes import ucf_version

splits = pd.read_csv("data/index/splits.csv")
train = splits[(splits.split == "train") & splits.sha256.isin(wanted_games())].sample(400, random_state=0)
parts = []
for sha, path in zip(train.sha256, train.path):
    if ucf_version(path) != "0.84":
        continue
    t = pd.read_parquet(f"data/parsed/{sha}.parquet", columns=["frame", "stick_x", "stick_y", "raw_stick_x", "raw_stick_y"])
    parts.append(t[t.frame >= 0].dropna())
t = pd.concat(parts)
t["px"] = np.round(t.stick_x * 80).astype(int)
t["py"] = np.round(t.stick_y * 80).astype(int)
mode = t.groupby(["px", "py", "raw_stick_x", "raw_stick_y"]).size().reset_index(name="n")
mode = mode.sort_values("n").groupby(["px", "py"]).tail(1)
out = {f"{px},{py}": [int(rx), int(ry)] for px, py, rx, ry in zip(mode.px, mode.py, mode.raw_stick_x, mode.raw_stick_y)}
json.dump(out, open("data/m4/raw_given_processed.json", "w"))
hit = t.merge(mode, on=["px", "py"], suffixes=("", "_mode"))
exact = ((hit.raw_stick_x == hit.raw_stick_x_mode) & (hit.raw_stick_y == hit.raw_stick_y_mode)).mean()
print(f"{len(parts)} UCF 0.84 games, {len(t)} frames, {len(out)} processed positions; "
      f"the most common raw is the true raw on {exact:.3f} of frames")
