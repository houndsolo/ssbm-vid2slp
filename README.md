# vid2slp

Turn Super Smash Bros. Melee video (60 fps YouTube/VODs) into Slippi `.slp` replays with
per-frame controller inputs. Exact inputs can't be recovered; the target is a replay that
plays out the same as the video.

## How it works

1. **Perception** (`m3_*`): video frames -> game state per frame (character, action, position,
   facing, damage, stocks). Trained on library replays rendered with Slippi Playback.
2. **Inverse dynamics** (`m2_*`): game states -> likely controller inputs (probabilities).
3. **Closed loop** (`m4_closed_loop.py`): a headless, rewindable Melee engine plays the game while
   inputs are chosen; candidates that keep the engine closest to what perception saw are kept.
   The output `.slp` is the engine's own recording, so it plays back identically in Slippi.
4. **Real footage** (`real_*`): crop/cut the video into games, read stage, ports, costumes and
   start positions from it, rebuild, compare side by side with the video.

## Status (2026-11)

- Rendered games (true replay known): action match **0.89** vs the true game on 15 held-out test
  games; up to **0.92** with more search + best of 4 runs (validation games). Perception alone: 0.96.
- First real tournament set rebuilt end to end (Genesis 7, Hax vs Leffen, 3 games): every stock
  loss reproduced; the replay follows what perception read on 78-85 % of frames. No true replay
  exists for real footage, so real-footage numbers are vs perception.
- Speed: ~45 min of CPU per minute of game; one game alone runs ~3.4x faster (`--engines`).

All results, including failures: [docs/experiments.md](docs/experiments.md).

## Run (real video)

```
.venv/bin/python scripts/real_prepare.py --id <youtube-id>                       # download, crop, cut games
.venv/bin/python scripts/m3_predict.py --run percep_v6soft --countdown --real <youtube-id>
.venv/bin/python scripts/real_rebuild.py --id <youtube-id> --games 1,2,3 --tag v1  # closed loop
.venv/bin/python scripts/real_compare_video.py --id <youtube-id> --tag v1 --games 1,2,3
```

Most accurate (several seeds, keep the best fit per game):
`bash scripts/real_rebuild_max.sh <youtube-id> 1,2,3 4 v3max` then `scripts/real_pick.py`.
Output: `data/real/<id>/rebuild/<tag>/game<k>.slp`.

## Requirements

Linux, NVIDIA GPU (developed on a V100, torch cu126), Python 3.12 via `uv`
(`uv pip install --python .venv/bin/python -e .`), the native engine built with
`scripts/native_build.sh` (patches in `native/`), a Melee NTSC 1.02 ISO, and Slippi Playback
Dolphin for rendering. Data, models, ISO and replays are not in the repo.

## Layout

- `src/vid2slp/` library: replay parsing/writing, models, engine host (`engine.py`, `engine_pool.py`)
- `scripts/` one job per script; `m2_` inputs, `m3_` perception, `m4_` closed loop, `real_` real footage
- `docs/` design notes and the experiment log
- `native/` patches for the headless engine
- `tests/` pytest checks
