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
