"""
m2_add_raw.py - add RAW main-stick targets to an existing input-model dataset.

Why: UCF 0.84 reads the raw stick bytes, not only the processed values the model predicts (see
src/vid2slp/id_data.py, RAW_CLASSES). With the TRUE processed inputs but raw = processed * 80, the
12 Stage C test games replayed with action match 0.666 instead of 0.930 with the true raw bytes
(2026-10-30). So the model gets two more outputs: raw stick x and y.

For each split of the dataset it writes <split>_raw.npy: int16 [N, 2, 2] (raw x, y as 0..255,
-100 where the replay has no raw value), in the same row order as the other arrays (games in
<split>_games.txt order, frames >= 0, checked against <split>_offsets.npy).

Run:    .venv/bin/python scripts/m2_add_raw.py --data data/m2_perc
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np

from vid2slp.id_data import RAW_MISSING, raw_arrays


def work(sha):
    return raw_arrays(f"data/parsed/{sha}.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/m2_perc")
    args = ap.parse_args()
    for split in ("train", "validation", "validation_truth", "test"):
        if not os.path.exists(f"{args.data}/{split}_games.txt"):
            continue
        games = open(f"{args.data}/{split}_games.txt").read().split()
        offsets = np.load(f"{args.data}/{split}_offsets.npy")
        out = np.lib.format.open_memmap(f"{args.data}/{split}_raw.npy", "w+", np.int16, (int(offsets[-1]), 2, 2))
        with Pool(12) as pool:
            for g, raw in enumerate(pool.imap(work, games, chunksize=4)):
                a, b = int(offsets[g]), int(offsets[g + 1])
                assert len(raw) == b - a, f"{games[g]}: {len(raw)} raw frames vs {b - a} rows"
                out[a:b] = raw
        out.flush()
        have = (out[..., 1] != RAW_MISSING).mean()
        print(f"{split}: {len(games)} games, {offsets[-1]:,} frames, raw y stored on {have:.3f} of them", flush=True)


if __name__ == "__main__":
    main()
