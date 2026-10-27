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

Running Dolphin here: inside the `vid2slp-render:gcc14` Docker image (built by the old project; renamed from yt2slp-render) (this server lacks
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

## Results

### Per-frame accuracy (does the model guess the right input?)
Two training runs, both 60k steps, seed 0; the checkpoint with the best validation score
(step 55k) was kept. `feat2_v4` adds "physics" features (how much each velocity changed
since the last frame, and finer position deltas), because the replay tests showed most
mistakes were the exact analog stick value, which the game turns into a velocity change.

| Validation | Lookup | `base_v3` (v1 features) | `feat2_v4` (v2 features) |
|---|---|---|---|
| Whole frame correct | 0.321 | 0.649 | **0.661** |
| Stick x exact (1 of 161 values) | 0.497 | 0.742 | **0.763** |

`feat2_v4` was chosen on validation, then scored **once** on the test split (344 games,
opponents never seen in training):

| Test | Always neutral | Lookup | `feat2_v4` |
|---|---|---|---|
| Whole frame correct | 0.232 | 0.320 | **0.658** |
| Stick zone correct, both axes | | 0.415 | **0.795** |
| Stick x exact | | 0.497 | **0.781** |
| Press F1 within ±2 frames | 0 | 0.651 | **0.761** |

Validation and test agree closely, so the model is not overfitted to the validation games.

### Replay tests (do the predicted inputs make the game do the same thing?)
`scripts/m2_resim_eval.py`: 12 games, first 3,600 frames (1 minute). Two tests:
- *whole-minute*: every frame uses predicted inputs;
- *splice*: true inputs up to frame k (600, 1200, ..., 3000), predicted after; measure how
  long the re-simulated game stays identical (exact) and how long the action states agree.

**Resync OFF** (pure simulation from inputs; any small error snowballs):

| | `base_v3` val | `feat2_v4` val | `feat2_v4` test |
|---|---|---|---|
| Frames exactly identical after switch (median / p75) | 13 / 38 | 22 / 46 | 48 / 119 |
| Frames until first action-state difference (median / p75) | 110 / 185 | 124 / 182 | 120 / 234 |
| Splices still matching after 600 frames | 0% | 3% | 5% |
| Whole minute: action states matching / mean x error | 0.368 / 47.3 | 0.345 / 52.4 | 0.347 / 48.4 |

**Resync ON** (Slippi Launcher's default, `shouldResync` = true): playback corrects itself
using the game states stored in the .slp.

| | `base_v3` val | `feat2_v4` test |
|---|---|---|
| Whole minute: action states matching / mean x error | 0.981 / 0.3 | 0.978 / 0.2 |
| After a splice: action states matching / mean x error | 0.983 / 0.20 | 0.981 / 0.19 |

### What this means (and what it does not)
- Open-loop, the predicted inputs keep the game identical for about 1 second and the same
  actions for about 2 seconds, then the simulation drifts (a slightly wrong stick value
  changes a position, which later changes what hits what). This is expected: many inputs are
  ambiguous and one wrong 1/80 step is enough.
- With resync, playback stays aligned (98% of frames, sub-unit position error) because it
  snaps back to the stored states. **Caveat:** in these tests the stored states are the TRUE
  states from the original replay. In the real pipeline (Stage C) they will be the
  perception model's estimates, so the result will only be as good as perception.
  Unverified: how tolerant resync is of imperfect stored states.
- The per-frame number (66% of frames fully correct) is a strict score: "correct" means the
  stick zone, C-stick direction, trigger level and all 8 buttons match on that frame.
