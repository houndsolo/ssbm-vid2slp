"""
m2_build_dataset.py - build the milestone-2 (inverse dynamics) arrays for train/validation/test.

Reads data/parsed/<sha256>.parquet (from parse_slps.py) and the frozen split, and writes
    data/m2/<split>_{num,cat,tgt,btn,offsets}.npy  and  data/m2/<split>_games.txt
The array layout and the exact feature/target definitions are in src/vid2slp/id_data.py.

Arrays are written straight into files on disk (memory-mapped), so building the
~25M-frame training split does not need tens of GB of RAM.

Run (CPU, ~1-2 min):
    .venv/bin/python scripts/m2_build_dataset.py
"""

import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from vid2slp import id_data

OUT = "data/m2"


def work(job):
    path, stage_id = job
    return id_data.game_arrays(path, stage_id)


def build(split, games):
    lengths = (games.last_frame + 1).to_numpy()  # frames 0..last_frame
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    n = int(offsets[-1])
    arrays = {
        "num": np.lib.format.open_memmap(f"{OUT}/{split}_num.npy", "w+", np.float32, (n, 2, id_data.NUM_FEATURES)),
        "cat": np.lib.format.open_memmap(f"{OUT}/{split}_cat.npy", "w+", np.int16, (n, 2, id_data.CAT_FEATURES)),
        "tgt": np.lib.format.open_memmap(f"{OUT}/{split}_tgt.npy", "w+", np.int16, (n, 2, 5)),
        "btn": np.lib.format.open_memmap(f"{OUT}/{split}_btn.npy", "w+", np.uint8, (n, 2)),
    }
    jobs = list(zip(games.parquet, games.stage_id))
    with Pool(12) as pool:
        # imap keeps the original order, so game g lands at offsets[g].
        for g, (num, cat, tgt, btn) in enumerate(pool.imap(work, jobs, chunksize=4)):
            a, b = offsets[g], offsets[g + 1]
            assert len(num) == b - a, f"length mismatch in {games.sha256.iloc[g]}"
            arrays["num"][a:b], arrays["cat"][a:b], arrays["tgt"][a:b], arrays["btn"][a:b] = num, cat, tgt, btn
    for arr in arrays.values():
        arr.flush()
    np.save(f"{OUT}/{split}_offsets.npy", offsets)
    open(f"{OUT}/{split}_games.txt", "w").write("\n".join(games.sha256) + "\n")
    gib = sum(a.nbytes for a in arrays.values()) / 2**30
    print(f"{split:10s} games {len(games):5d}  frames {n:,}  ({n / 216000:.1f} h)  {gib:.2f} GiB", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = pd.read_csv("data/parsed/manifest.csv")
    index = pd.read_csv("data/index/slp_index.csv")[["sha256", "stage_id"]]
    games = manifest.merge(index, on="sha256").sort_values("sha256")
    games = games[games["error"].isna()] if "error" in games else games
    for split in ("train", "validation", "test"):
        build(split, games[games.split == split].reset_index(drop=True))


if __name__ == "__main__":
    main()
