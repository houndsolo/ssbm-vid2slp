"""
m4_pool_check.py - compare rebuilds of the same game stretch made with different speed settings:
are the replays identical, and how long did each take?

Why: real_rebuild.py --engines N (engine pool, src/vid2slp/engine_pool.py) and the engine's
mark/back (one engine kept per game) must not change the result, only the time. So the same
stretch is rebuilt once with the original code (the reference) and once per setting, and the
replays are compared byte by byte (apart from the recording's start time, "startAt", which is
the clock time when it was written).

Run (after the rebuilds, e.g. --tag check_ref_end600 from the original code, check_e8_end600 ...):
    .venv/bin/python scripts/m4_pool_check.py --id 5njRxIAgS-o --game 3 --ref check_ref_end600 \
        --tags check_e1_end600,check_e8_end600
"""

import argparse
import re

import pandas as pd


def without_time(data):
    """The replay bytes with the recording's start time blanked: the only part that may differ."""
    return re.sub(rb"startAtSU\x14.{20}", b"startAt", data, flags=re.DOTALL)


def load(args, tag):
    out = f"data/real/{args.id}/rebuild/{tag}"
    seconds = float(pd.read_csv(f"{out}/results.csv").seconds.iloc[0])
    return without_time(open(f"{out}/game{args.game}.slp", "rb").read()), seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="5njRxIAgS-o")
    ap.add_argument("--game", type=int, default=3)
    ap.add_argument("--ref", required=True, help="tag of the reference rebuild (original code)")
    ap.add_argument("--tags", required=True, help="comma-separated tags to compare with it")
    args = ap.parse_args()
    ref, ref_s = load(args, args.ref)
    print(f"{args.ref}: {ref_s:.0f} s (reference)")
    for tag in args.tags.split(","):
        data, s = load(args, tag)
        print(f"{tag}: {s:.0f} s, {ref_s / s:.2f}x the reference's speed, replay identical: {data == ref}")


if __name__ == "__main__":
    main()
