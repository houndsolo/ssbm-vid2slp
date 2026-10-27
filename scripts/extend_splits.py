"""
extend_splits.py - add newly indexed games to the EXISTING train / validation / test split
without moving any game that already has a split.

Why not just rerun make_splits.py? It shuffles all opponents again, so games that our trained
models (M2 inverse dynamics, M3 perception) learned from could land in validation or test,
and every held-out number measured so far would stop being held-out. Here:
  1. Every game already in data/index/splits.csv keeps its split.
  2. A new game against an opponent who already has a split goes to that same split
     (an opponent is never in two splits, same rule as make_splits.py).
  3. New opponents are shuffled with a fixed seed and placed like make_splits.py does:
     fill TEST, then VALIDATION, up to ~10% each of the NEW total; opponents with more than
     2% of all games always go to TRAIN.
  4. Writes data/index/splits.csv (old file kept as splits_before_<date>.csv) and adds an
     "extensions" entry to data/index/splits_config.json.

Run (after build_slp_index.py has written the new index):
    .venv/bin/python scripts/extend_splits.py --index data/index/slp_index_2026-10-27.csv
"""

import argparse
import json
import os
import random
import shutil
from collections import Counter

import pandas as pd

from make_splits import COMMON_PLAYER, MAX_HELDOUT_GROUP, TARGETS, opponent_of

SPLITS = "data/index/splits.csv"
CONFIG = "data/index/splits_config.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="the new slp index (old + new files)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="2026-10-27", help="date tag for the backup file name")
    args = ap.parse_args()

    old = pd.read_csv(SPLITS)
    games = pd.read_csv(args.index)
    games = games[games.eligible].copy()
    games["group"] = games.codes.map(opponent_of)

    # Every old game must still be there, unchanged; otherwise stop and look by hand.
    missing = set(old.sha256) - set(games.sha256)
    if missing:
        raise SystemExit(f"{len(missing)} games of the old split are not in the new index - stopping")

    # 1 + 2: known opponents keep their split.
    group_split = dict(zip(old.group, old.split))
    old_split = dict(zip(old.sha256, old.split))
    total = len(games)
    counts = Counter(old.split)
    known = games.group.isin(group_split) & ~games.sha256.isin(old_split)
    counts.update(games[known].group.map(group_split))

    # 3: new opponents, shuffled with a fixed seed (sorted first so it's reproducible).
    sizes = games[~games.group.isin(group_split)].group.value_counts()
    groups = sorted(sizes.index)
    random.Random(args.seed).shuffle(groups)
    for g in groups:
        n = int(sizes[g])
        placed = "train"
        for split, frac in (TARGETS.items() if n <= MAX_HELDOUT_GROUP * total else []):
            target = frac * total
            if counts[split] < target and counts[split] + n <= target + 0.01 * total:
                placed = split
                break
        group_split[g] = placed
        counts[placed] += n

    games["split"] = games.group.map(group_split)

    # Safety checks: old games unchanged, no opponent or file in two splits.
    moved = sum(old_split[s] != sp for s, sp in zip(games.sha256, games.split) if s in old_split)
    leaks = games.groupby("group").split.nunique().gt(1).sum()
    dup_files = games.groupby("sha256").split.nunique().gt(1).sum()
    print(f"old games that changed split: {moved}   opponents in >1 split: {leaks}   files in >1 split: {dup_files}")
    if moved or leaks or dup_files:
        raise SystemExit("safety check failed - nothing written")

    summary = {}
    for split in ("train", "validation", "test"):
        part = games[games.split == split]
        new_part = part[~part.sha256.isin(old_split)]
        summary[split] = {"games": int(len(part)), "new_games": int(len(new_part)),
                          "opponents": int(part.group.nunique()),
                          "hours": round(float((part.last_frame + 123).sum() / 60 / 3600), 1)}
        print(f"{split:10s} games {len(part):5d} ({len(part) / total:5.1%})  of which new {len(new_part):5d}  "
              f"opponents {part.group.nunique():4d}  hours {summary[split]['hours']:6.1f}")

    backup = SPLITS.replace(".csv", f"_before_{args.tag}.csv")
    shutil.copy(SPLITS, backup)
    games[["sha256", "split", "group", "path", "stage", "characters", "last_frame", "version"]] \
        .sort_values(["split", "path"]).to_csv(SPLITS, index=False)
    config = json.load(open(CONFIG)) if os.path.exists(CONFIG) else {}
    config.setdefault("extensions", []).append(
        {"date": args.tag, "index": args.index, "seed": args.seed, "common_player": COMMON_PLAYER,
         "rule": "old games and known opponents keep their split; new opponents placed like make_splits.py",
         "summary": summary})
    json.dump(config, open(CONFIG, "w"), indent=1, ensure_ascii=False)
    print("wrote", SPLITS, "(old copy:", backup + ")", "and updated", CONFIG)


if __name__ == "__main__":
    main()
