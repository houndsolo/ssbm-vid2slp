"""
parse_slps.py - parse every game in the frozen split into a per-frame Parquet table.

Input:  data/index/splits.csv  (made by make_splits.py)
Output: data/parsed/<sha256>.parquet   one file per game, one row per (frame, port);
        columns are described in src/vid2slp/slp_parse.py
        data/parsed/manifest.csv       one row per game: split, rows, frame checks, errors

Games already parsed are skipped, so it is safe to re-run after an interruption.
Uses 12 worker processes (the server has 16 cores).

Run (CPU only, a few minutes; use tmux):
    .venv/bin/python scripts/parse_slps.py
"""

import os
import time
from multiprocessing import Pool

import pandas as pd

from vid2slp.slp_parse import parse_replay

SPLITS = "data/index/splits.csv"
OUT_DIR = "data/parsed"


def parse_one(job):
    """Parse one game; returns a manifest row. Never raises, so one bad file can't stop the run."""
    sha, split, path = job
    out = os.path.join(OUT_DIR, f"{sha}.parquet")
    row = {"sha256": sha, "split": split, "path": path, "parquet": out}
    try:
        table, info = parse_replay(path)
        # Write to a temp name first, then rename: an interrupted write never looks finished.
        table.to_parquet(out + ".tmp", index=False, compression="zstd")
        os.replace(out + ".tmp", out)
        row.update(info)
        row["rows"] = len(table)
        row["parquet_bytes"] = os.path.getsize(out)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return row


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    games = pd.read_csv(SPLITS)
    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    old = pd.read_csv(manifest_path) if os.path.exists(manifest_path) else pd.DataFrame()

    # Games parsed OK last time. (The manifest only has an "error" column if some game failed.)
    if len(old) and "error" in old.columns:
        done = set(old[old.error.isna()].sha256)
    else:
        done = set(old.sha256) if len(old) else set()
    jobs = [(r.sha256, r.split, r.path) for r in games.itertuples()
            if r.sha256 not in done or not os.path.exists(os.path.join(OUT_DIR, f"{r.sha256}.parquet"))]
    print(f"games in split file: {len(games)}   already parsed: {len(games) - len(jobs)}   to parse: {len(jobs)}",
          flush=True)

    start = time.time()
    rows = []
    with Pool(12) as pool:
        for i, row in enumerate(pool.imap_unordered(parse_one, jobs, chunksize=4), 1):
            rows.append(row)
            if i % 250 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {time.time() - start:.0f}s", flush=True)

    new = pd.DataFrame(rows)
    manifest = pd.concat([old[old.sha256.isin(done)] if len(old) else old, new], ignore_index=True)
    manifest.to_csv(manifest_path, index=False)

    ok = manifest[manifest.get("error", pd.Series(index=manifest.index, dtype=str)).isna()]
    print(f"\nparsed OK: {len(ok)}   failed: {len(manifest) - len(ok)}")
    print("per split:", ok.split.value_counts().to_dict())
    print(f"total rows (frame x player): {int(ok.rows.sum()):,}   "
          f"disk: {ok.parquet_bytes.sum() / 2**30:.2f} GiB")
    print(f"games with duplicate (rollback) frame ids: {(ok.duplicate_frame_ids > 0).sum()}   "
          f"games with missing frames after de-duplication: {(ok.missing_frames > 0).sum()}")
    print("wrote", manifest_path)


if __name__ == "__main__":
    main()
