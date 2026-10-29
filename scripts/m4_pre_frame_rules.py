"""
m4_pre_frame_rules.py - learn, from real replays, two facts the Stage C .slp writer needs.

1. START-of-frame action. A replay stores each frame's state twice: at the start (pre-frame)
   and at the end (post-frame). Perception gives end-of-frame states. Usually
   start(f) = end(f-1), but some changes happen automatically at the very start of a frame
   (jump squat -> jump, landing lag -> standing, ...): then start(f) already shows the NEW action.
   With resync on, Slippi Playback forces the stored start-of-frame action every frame
   (slippi-ssbm-asm Playback/Core/RestoreGameFrame.asm), so writing the old action makes the
   replay lag one frame at every such change. Here we count, for every action change a -> b
   between end(f-1) and end(f), how often start(f) was already b.
   (Same for facing: does a turn show up at the start of the frame?)

2. GROUND heights. Resync also forces the stored x/y every frame. A standing character whose
   y is 0.9 too high is in the air and starts falling. So for each stage we collect the exact
   heights where characters stand (main stage, platforms), per 2-unit x bin, to snap
   perceived y onto when perception says "on the ground".

Reads training-split replays only. Output: data/m4/pre_frame_rules.json, data/m4/ground_heights.json

Run:
    .venv/bin/python scripts/m4_pre_frame_rules.py --games 600
"""

import argparse
import json
import os
import struct
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

from vid2slp.slp_write import POST_FRAME, POST_STATE_FIELDS, PRE_FRAME, PRE_STATE_FIELDS, frame_events

X_BIN = 2.0


def read_states(data, cmd, table):
    out = {}
    for pos, f, p in frame_events(data, cmd):
        out[(f, p)] = {k: struct.unpack(fmt, data[pos + o:pos + o + struct.calcsize(fmt)])[0]
                       for k, (o, fmt) in table.items()}
    return out


def one_game(job):
    path, stage = job
    try:
        data = open(path, "rb").read()
        pre = read_states(data, PRE_FRAME, PRE_STATE_FIELDS)
        post = read_states(data, POST_FRAME, POST_STATE_FIELDS)
    except Exception:
        return None
    changes, turns, ground = Counter(), Counter(), defaultdict(Counter)
    for (f, p), s in pre.items():
        a, b = post.get((f - 1, p)), post.get((f, p))
        if f < 1 or a is None or b is None:
            continue
        if a["action"] != b["action"]:
            changes[(a["action"], b["action"], s["action"] == b["action"])] += 1
        if a["facing"] != b["facing"]:
            turns[s["facing"] == b["facing"]] += 1
        if not b["airborne"]:
            ground[int(np.floor(b["x"] / X_BIN))][round(b["y"], 3)] += 1
    return stage, changes, turns, ground


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    splits = pd.read_csv("data/index/splits.csv")
    stage_ids = pd.read_csv("data/index/slp_index.csv").drop_duplicates("sha256").set_index("sha256").stage_id
    train = splits[splits.split == "train"].sample(args.games, random_state=args.seed)
    jobs = [(r.path, int(stage_ids[r.sha256])) for r in train.itertuples()]

    changes, turns = Counter(), Counter()
    ground = defaultdict(lambda: defaultdict(Counter))       # stage -> x bin -> y -> count
    with Pool(12) as pool:
        for res in pool.imap_unordered(one_game, jobs, chunksize=4):
            if res is None:
                continue
            stage, c, t, g = res
            changes.update(c)
            turns.update(t)
            for xb, ys in g.items():
                ground[stage][xb].update(ys)

    # rule: for change a -> b, write b at the start of the frame if that happened > 50% of the time
    pairs = defaultdict(lambda: [0, 0])
    for (a, b, early), n in changes.items():
        pairs[(a, b)][int(early)] += n
    rules = {f"{a},{b}": round(early / (late + early), 3) for (a, b), (late, early) in pairs.items()
             if late + early >= 5}
    early_total = sum(n for (a, b, e), n in changes.items() if e)
    print(f"action changes: {sum(changes.values()):,}; already at frame start: {early_total / sum(changes.values()):.1%}")
    print(f"facing changes: {sum(turns.values()):,}; already at frame start: {turns[True] / max(sum(turns.values()), 1):.1%}")

    # ground heights: per stage and x bin, heights seen at least 20 times and on >= 2% of frames
    heights = {}
    for stage, bins in ground.items():
        heights[stage] = {}
        for xb, ys in bins.items():
            total = sum(ys.values())
            keep = sorted(y for y, n in ys.items() if n >= 20 and n >= 0.02 * total)
            if keep:
                heights[stage][xb] = keep
        print(f"stage {stage}: {len(heights[stage])} x bins with ground heights")
    os.makedirs("data/m4", exist_ok=True)
    json.dump({"action_early": rules, "facing_early": turns[True] / max(sum(turns.values()), 1),
               "games": len(jobs), "seed": args.seed}, open("data/m4/pre_frame_rules.json", "w"), indent=0)
    json.dump({"x_bin": X_BIN, "heights": heights}, open("data/m4/ground_heights.json", "w"))
    print("wrote data/m4/pre_frame_rules.json and data/m4/ground_heights.json")


if __name__ == "__main__":
    main()
