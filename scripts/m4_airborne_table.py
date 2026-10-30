"""
m4_airborne_table.py - for every (character, action) pair: is the character usually in the air?

Why: the closed loop (scripts/m4_closed_loop.py --replan) runs the input model on the ENGINE's
recent states, but the engine's state line has no "airborne" flag (the model uses one). The action
says it almost always: measured on 200 training games, "majority airborne value of this
(character, action)" is right on 98.7% of frames (2026-10-30).

Run:    .venv/bin/python scripts/m4_airborne_table.py
Output: data/m4/airborne_by_action.json  {"<character>,<action>": 0 or 1}
"""

import json

import pandas as pd

from vid2slp.matchups import wanted_games

splits = pd.read_csv("data/index/splits.csv")
train = splits[(splits.split == "train") & splits.sha256.isin(wanted_games())].sample(200, random_state=0)
parts = []
for sha in train.sha256:
    t = pd.read_parquet(f"data/parsed/{sha}.parquet", columns=["frame", "character", "action", "airborne"])
    parts.append(t[t.frame >= 0])
table = pd.concat(parts).groupby(["character", "action"]).airborne.mean()
out = {f"{c},{a}": int(v > 0.5) for (c, a), v in table.items()}
json.dump(out, open("data/m4/airborne_by_action.json", "w"))
print(f"{len(out)} (character, action) pairs written to data/m4/airborne_by_action.json")
