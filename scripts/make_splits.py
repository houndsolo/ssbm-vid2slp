"""
make_splits.py - freeze the train / validation / test split of the replay library.

Rules (from CLAUDE.md): split by replay FILE, never by frame. We go further and split
by OPPONENT: every eligible game in the library has the same player (DARK#880) on one
side, so we group games by the OTHER player's connect code and put each opponent's games
entirely into one split. That way a test game never shares an opponent (or a session
against that opponent) with any training game.

Method:
  1. Read data/index/slp_index.csv, keep eligible games.
  2. Group games by opponent connect code.
  3. Shuffle the groups with a fixed seed, then fill TEST up to ~10% of games, VALIDATION
     up to ~10%, and the rest go to TRAIN. A group that would overshoot a target by
     more than 1% of all games is skipped for that split. Opponents with more than 2% of
     all games always go to TRAIN; otherwise one big opponent could make up most of
     validation (that happened on the first try: SEED#602 was 70% of validation).
  4. Write data/index/splits.csv and data/index/splits_config.json (seed + counts).

Run:
    .venv/bin/python scripts/make_splits.py            # seed 0
    .venv/bin/python scripts/make_splits.py --seed 1
Re-running with the same seed and index gives the identical split.
"""

import argparse
import json
import random
from collections import Counter

import pandas as pd

INDEX = "data/index/slp_index.csv"
OUT = "data/index/splits.csv"
CONFIG_OUT = "data/index/splits_config.json"
COMMON_PLAYER = "DARK＃880"   # appears in every eligible game (full-width '#' as stored by Slippi)
TARGETS = {"test": 0.10, "validation": 0.10}
MAX_HELDOUT_GROUP = 0.02     # opponents bigger than this share of all games always go to train


def opponent_of(codes):
    """The connect code that is not COMMON_PLAYER (or the pair, if neither/both are)."""
    parts = [c for c in str(codes).split(" ") if c]
    others = [c for c in parts if c != COMMON_PLAYER]
    return others[0] if len(others) == 1 else "|".join(sorted(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    games = pd.read_csv(INDEX)
    games = games[games.eligible].copy()
    games["group"] = games.codes.map(opponent_of)
    sizes = games.group.value_counts()

    groups = sorted(sizes.index)          # sort first so the shuffle is reproducible
    random.Random(args.seed).shuffle(groups)
    total = len(games)
    counts = Counter()
    assignment = {}
    for g in groups:
        n = int(sizes[g])
        placed = "train"
        for split, frac in (TARGETS.items() if n <= MAX_HELDOUT_GROUP * total else []):
            target = frac * total
            if counts[split] < target and counts[split] + n <= target + 0.01 * total:
                placed = split
                break
        assignment[g] = placed
        counts[placed] += n

    games["split"] = games.group.map(assignment)
    games[["sha256", "split", "group", "path", "stage", "characters", "last_frame", "version"]] \
        .sort_values(["split", "path"]).to_csv(OUT, index=False)

    # Report and save the config so this run can be reproduced and checked.
    summary = {}
    for split in ("train", "validation", "test"):
        part = games[games.split == split]
        biggest = part.group.value_counts()
        summary[split] = {
            "games": int(len(part)),
            "opponents": int(part.group.nunique()),
            "hours": round(float((part.last_frame + 123).sum() / 60 / 3600), 1),
            "stages": part.stage.value_counts().to_dict(),
            "largest_opponent_share": round(float(biggest.iloc[0] / len(part)), 3),
        }
        print(f"{split:10s} games {len(part):5d} ({len(part) / total:5.1%})  opponents {part.group.nunique():4d}  "
              f"hours {summary[split]['hours']:6.1f}  largest opponent {biggest.iloc[0] / len(part):.0%}")
    leaks = games.groupby("group").split.nunique().gt(1).sum()
    dup_files = games.groupby("sha256").split.nunique().gt(1).sum()
    print(f"opponents in more than one split: {leaks}   files in more than one split: {dup_files}")
    json.dump({"seed": args.seed, "index": INDEX, "group_by": "opponent connect code",
               "max_heldout_group": MAX_HELDOUT_GROUP,
               "common_player": COMMON_PLAYER, "targets": TARGETS, "summary": summary},
              open(CONFIG_OUT, "w"), indent=1, ensure_ascii=False)
    print("wrote", OUT, "and", CONFIG_OUT)


if __name__ == "__main__":
    main()
