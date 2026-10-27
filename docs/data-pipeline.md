# Data pipeline (milestone 1)

Turns the .slp replay library into per-frame tables for training. CPU only, ~1 minute total.

## Rebuild everything from scratch
```bash
uv pip install --python .venv/bin/python -e .        # once: lets scripts `import vid2slp`
.venv/bin/python scripts/build_slp_index.py data/recordings   # ~15 s
.venv/bin/python scripts/make_splits.py                                         # seed 0
.venv/bin/python scripts/parse_slps.py                                          # ~15 s
.venv/bin/python -m pytest -q                                                   # checks
```

## What each step makes
| File | What it is |
|---|---|
| `data/index/slp_index.csv` | One row per .slp file: hash, version, stage, characters, connect codes, length, and `eligible` + `why_not` |
| `data/index/splits.csv` | Eligible games with their `split` (train/validation/test) and `group` (opponent code) |
| `data/index/splits_config.json` | Seed and per-split counts, so the split can be reproduced |
| `data/parsed/<sha256>.parquet` | One game: one row per (frame, port), inputs + state. Columns are documented in `src/vid2slp/slp_parse.py` |
| `data/parsed/manifest.csv` | One row per parsed game: split, row count, duplicate/missing frame counts, errors |

## Rules baked in
- **Eligible game** = 2 human players, not teams, NTSC, legal stage, at least 30 s long, not a duplicate file.
- **Split by opponent**: every game includes DARK#880, so games are grouped by the
  other player's connect code and each opponent is in exactly one split. Opponents with
  more than 2% of all games always go to train, which keeps validation/test diverse.
- **Rollback duplicates**: frame numbers can repeat inside a .slp; the last copy wins.
  85% of games have this.
- Frames start at -123 (pre-"GO" countdown); training code should usually start at frame 0.
- Only the "leader" character is stored (Ice Climbers' Nana is ignored; 15 player-slots).
- `raw_stick_y` is missing (NaN) for replays older than Slippi 3.15.

## Library numbers (2026-10-26)
4,755 files → 49 unreadable (peppi-py 0.8.6 doesn't know game-end code 8; a newer parser
version may fix this) → **3,435 eligible games, 136 hours**. Stages: Battlefield 1,741,
Yoshi's 584, FD 380, Stadium 274, Dream Land 252, FoD 204. Characters are mostly Fox (4,311
player-slots) and Falco (1,406), then Marth 292, Ganon 225, Falcon 221.

| Split | Games | Opponents | Hours |
|---|---|---|---|
| train | 2,738 | 704 | 108.5 |
| validation | 353 | 160 | 14.1 |
| test | 344 | 146 | 13.8 |

## Library update (2026-10-27)
12,848 more .slp from the owner's other machine were moved (same disk, rename) into
`data/recordings/incoming-2026-10-27/`. The whole of `data/recordings` is read-only.
- Index rebuilt over old + new files (`build_slp_index.py data/recordings`); the previous
  index is kept as `data/index/slp_index_before_2026-10-27.csv`.
- The split was EXTENDED, not remade: `scripts/extend_splits.py` keeps every old game's split
  and puts new games of a known opponent into that opponent's split; only new opponents are
  shuffled (seed 0). Old split kept as `data/index/splits_before_2026-10-27.csv`.
  (Rerunning `make_splits.py` would reshuffle everything and leak trained-on games into test.)
- 17,603 files -> 2,457 duplicates, 49 unreadable -> **12,663 eligible games, 480 hours**.

| Split | Games | of which new | Opponents | Hours |
|---|---|---|---|---|
| train | 10,121 | 7,383 | 1,587 | 382.6 |
| validation | 1,267 | 914 | 306 | 48.5 |
| test | 1,275 | 931 | 258 | 48.5 |

Fox-Fox / Fox-Falco games (the project's scope): 9,277 train / 1,156 validation / 1,193 test.
