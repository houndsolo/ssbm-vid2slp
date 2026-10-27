# Milestone 2: inverse dynamics (game state → controller inputs)

Goal: given what the characters did (the per-frame game state of both players), work out
what was pressed. Trained only on .slp replays; no video involved yet.

## Data
- Built from the milestone-1 tables: `scripts/m2_build_dataset.py` → `data/m2/`.
  Train 23.1M frames (107 h), validation 3.0M (13.9 h), test 2.9M (13.6 h); the split is
  by opponent (see `docs/data-pipeline.md`). Frames −123..−1 (countdown) are dropped.
- **Features** (per frame, both players): position, movement since last frame, velocities,
  percent, shield, stocks, hitlag, frames into the current action, jumps left, airborne,
  facing; and categories: action state, character, platform, hurtbox state, L-cancel
  status, last attack landed, stage. Defined in `src/vid2slp/id_data.py`.
- **Targets** (per frame, per player): stick x/y and C-stick x/y (each one of 161 values,
  the game's 1/80 steps), trigger (141 values, 1/140 steps), and 8 button flags:
  A, B, Z, JUMP (X or Y held), SHIELD (L or R held), JUMP_EDGE, SHIELD_EDGE (a button of
  that pair went down this frame), DPAD_UP (taunt).

## How a replay is actually played back (measured, not assumed)
Slippi Playback Dolphin 3.5.2 re-simulates the whole game from the inputs stored in the
.slp, and can write a regenerated .slp of what it simulated (`src/vid2slp/playback.py`).
An unmodified replay regenerates with 100% identical states, so this is an exact
simulator for checking predicted inputs. Facts found with `scripts/m2_resim_probe.py`:

| Question | Answer |
|---|---|
| Which input fields does playback use? | Only the *processed* ones (stick/C-stick/trigger floats + processed button word). Zeroing the physical fields changes nothing. |
| Stick-direction bits 16–23 of the button word | Recomputed by the game; can be written as 0. |
| Bit 31 ("trigger pressed") | Matters; in real data it equals `trigger > 0`. |
| X vs Y, L vs R | Interchangeable, **but only if every new press is reproduced**: pressing R while L is held (wavedash out of shield) is a new press. Hence the *_EDGE targets. |
| Z | Sets the A bit too and always gives a 0.35 processed trigger. |
| D-pad up | It's the taunt; must be written (used in ~15% of games). |

Running Dolphin here: inside the old `yt2slp-render:gcc14` Docker image (this server lacks
its OpenGL library), with `--shm-size=2g` (otherwise it aborts instantly), under Xvfb.
About 25–60 frames per second per instance; 4 in parallel loads the 16-core CPU fully.

**Oracle check** (true inputs → our targets → .slp → playback must match the original):
the first version failed (wavedash-out-of-shield, then taunts); with the final targets:
**40 of 40 validation games replayed with zero differences** over their first 3,600 frames
(or the whole game if shorter; minimum 2,545 frames): action states 100% equal, position
difference 0. So the target format loses nothing the game needs (`scripts/m2_resim_eval.py
--run oracle_check --variants oracle --games 40`).

## Baselines (no learning; `scripts/m2_baselines.py`)
| Metric (validation) | Always neutral | Lookup table |
|---|---|---|
| Stick zone correct, both axes | 0.321 | 0.415 |
| C-stick direction | 0.914 | 0.930 |
| Trigger category | 0.823 | 0.866 |
| Press F1 within ±2 frames (mean of A, B, Z, jump, shield presses) | 0 | 0.645 |
| Whole frame correct | 0.234 | 0.321 |

The lookup table predicts the most common input for (character, previous / current /
next action state) seen in training.

## Model (`src/vid2slp/id_model.py`, `scripts/m2_train.py`)
Transformer encoder, 8 layers, width 384, 8 heads, ~15M parameters. It reads a 256-frame
window (≈4 s) of both players' states and predicts every frame's inputs; attention has a
distance penalty (ALiBi) instead of position embeddings. fp16 + GradScaler, AdamW,
batch 64 windows, 60k steps (~70 min on the V100). Validation is scored every 5k steps on
all 353 validation games; the best checkpoint is kept. Test is scored once at the end.

MODEL_RESULTS
