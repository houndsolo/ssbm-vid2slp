"""
real_mismatch.py - WHERE does a rebuilt real game disagree with what perception read?

Why (2026-11-10): the Genesis 7 set 1 rebuild (5njRxIAgS-o, Hax vs Leffen) follows perception
on 78-85 % of player-frames (action). To push that up we first need to know what the other
15-22 % are:
  - timing: the right action, a few frames early or late (cheap to fix in the search?);
  - drift: long stretches where the rebuilt game went somewhere else (search lost the thread);
  - unsure perception: frames where perception itself was not confident (maybe perception's
    mistake, which a rebuild cannot and should not follow);
  - which action pairs are confused most.
There is no true replay of a real game, so every number here is "vs perception".

Run:
    .venv/bin/python scripts/real_mismatch.py --id 5njRxIAgS-o --tag v2 --games 1,2,3
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_rebuild import consistent                       # noqa: E402  (same rules as the rebuild)
from vid2slp.melee_ids import ACTION_NAMES                # noqa: E402
from vid2slp.slp_parse import parse_replay                # noqa: E402

REAL = "data/real"


def name(a):
    return ACTION_NAMES.get(int(a), str(int(a)))


def runs_of(flags):
    """Lengths of the runs of True in a boolean array, one length per True frame (so that
    'share of mismatched frames that sit in runs longer than N' is a simple mean)."""
    out = np.zeros(len(flags), dtype=int)
    start = None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out[start:i] = i - start
            start = None
    return out


def load(args, k):
    """One row per (frame, slot) with the rebuilt (r_*) and perceived (p_*) values side by side."""
    rebuilt, _ = parse_replay(f"{REAL}/{args.id}/rebuild/{args.tag}/game{k}.slp")
    slot = {p: i for i, p in enumerate(sorted(rebuilt.port.unique()))}
    rebuilt = rebuilt.assign(port=rebuilt.port.map(slot))
    seen = pd.read_parquet(f"{REAL}/{args.id}/pred_{args.perception}_countdown/game{k}.parquet")
    seen = consistent(seen[seen.frame >= -39])
    cols = ["frame", "port", "action", "x", "y", "percent", "stocks"]
    t = rebuilt[cols].merge(seen[cols + ["action_prob", "seen"]], on=["frame", "port"], suffixes=("_r", "_p"))
    return t[t.frame >= 0].sort_values(["port", "frame"]).reset_index(drop=True)


def report(t, k):
    print(f"\n===== game {k}: {len(t)} player-frames =====")
    t = t.copy()
    t["ok"] = t.action_r == t.action_p
    print(f"action match {t.ok.mean():.3f}   x error mean {np.abs(t.x_r - t.x_p).mean():.1f} "
          f"median {np.abs(t.x_r - t.x_p).median():.1f}")

    # 1. timing: does the perceived action appear in the rebuilt game within +-2 / +-5 frames?
    for tol in (2, 5):
        near = np.zeros(len(t), dtype=bool)
        for port, g in t.groupby("port"):
            r = g.action_r.to_numpy()
            p = g.action_p.to_numpy()
            hit = np.zeros(len(g), dtype=bool)
            for d in range(-tol, tol + 1):
                shifted = np.roll(r, d)
                hit |= shifted == p
            near[g.index] = hit
        print(f"  same action within +-{tol} frames: {near.mean():.3f}")

    # 2. drift: mismatches in long runs (the rebuilt game doing something else for a while)
    long = np.zeros(len(t), dtype=int)
    for port, g in t.groupby("port"):
        long[g.index] = runs_of((~g.ok).to_numpy())
    bad = ~t.ok
    for n in (10, 30, 60):
        print(f"  share of mismatched frames inside mismatch runs > {n} frames: {(long[bad] > n).mean():.3f}")

    # 3. perception confidence
    bins = [0, 0.5, 0.8, 0.95, 0.99, 1.01]
    t["conf"] = pd.cut(t.action_prob, bins, right=False)
    g = t.groupby("conf", observed=True).ok.agg(["mean", "size"])
    g["share"] = g["size"] / len(t)
    print("  by perception's confidence in its action (share of frames, match):")
    for c, row in g.iterrows():
        print(f"    {str(c):14s} {row['share']:.3f}  {row['mean']:.3f}")

    # 4. most common confusions (perceived -> rebuilt)
    pairs = t[bad].groupby(["action_p", "action_r"]).size().sort_values(ascending=False).head(15)
    print("  most common confusions (perceived -> rebuilt, share of all mismatches):")
    for (p, r), n in pairs.items():
        print(f"    {name(p):22s} -> {name(r):22s} {n / bad.sum():.3f}")

    # 5. drift in time: worst 300-frame stretches
    t["seg"] = t.frame // 300
    seg = t.groupby("seg").agg(match=("ok", "mean"), xerr=("x_r", lambda s: 0)).reset_index()
    seg["xerr"] = t.assign(e=np.abs(t.x_r - t.x_p)).groupby("seg").e.mean().to_numpy()
    worst = seg.sort_values("match").head(5)
    print("  worst 300-frame stretches (start frame, match, x error):")
    for _, row in worst.iterrows():
        print(f"    {int(row.seg) * 300:6d}  {row.match:.3f}  {row.xerr:.1f}")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="5njRxIAgS-o")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--games", default="1,2,3")
    ap.add_argument("--perception", default="percep_v6soft")
    args = ap.parse_args()
    for k in [int(x) for x in args.games.split(",")]:
        report(load(args, k), k)


if __name__ == "__main__":
    main()
