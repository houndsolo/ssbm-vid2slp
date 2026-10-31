# Stage C: closed loop (the game engine in the loop)

Written 2026-10-30, updated 2026-10-31. Numbers are in docs/experiments.md (rows of 2026-10-30).

## The problem it solves

The first Stage C pipeline (scripts/m4_end_to_end.py) guessed every frame's controller inputs
from the video once, wrote them into a .slp, and relied on Slippi's "resync" to keep the replay
on track. Resync forces each character's position, facing, action number and damage every frame,
but it does NOT run the game's logic for that action. So the replay can show the right action
number while the character really behaves like something else (for example still hanging on a
ledge). The stricter scores show it: the right action AND the right animation timing only about
half of the time.

## What the closed loop does

It plays the game in the native engine (a fast, rewindable copy of Melee; src/vid2slp/engine.py)
while it chooses the inputs:

1. The input model (M2, run `perc_v1`) gives a probability for every stick and button value on
   every frame, computed from what perception saw in the video.
2. Every 4 frames the engine saves its state. For each player it tries about 15 candidate input
   sequences: the model's most likely inputs, the same shifted 1-2 frames earlier or later, random
   samples from the model's probabilities, and the model re-run on what actually happened in the
   rebuilt game so far ("replanning"). Each candidate is played 4 + 60 frames ahead, and the one
   whose result looks most like the video (same action, position, facing, damage, stocks) is kept.
3. "Restarts": the game is rebuilt in 5-second segments (300 frames). Each segment is searched 3
   times with different random candidates, and the attempt that matches the video best is kept.
   The game is chaotic, so one unlucky choice can make the rebuilt game drift away for good;
   this keeps such an attempt from ruining the rest of the game. The choice uses only the video
   (perceived states), never the original replay.
   Optional (2026-10-30): `--extra-restarts N` gives a segment up to N more attempts while its
   best attempt still fits the video poorly (cost per frame above `--restart-threshold`), and
   `--backtrack N` searches the previous segment and the current one again together (N attempts)
   when the current one is still poor but the previous one looked fine: the mistake is often
   made near the end of the previous segment (for example a throw in the wrong direction) and
   only shows as a lost stock in the next one.
4. The output .slp is the engine's own recording of the kept inputs. Its stored states are the
   engine's real states, so it plays back the same in Slippi Playback with or without resync
   (checked: every output tested so far replays identically).

## Things it borrows from the original game

With `--countdown --from-video` (2026-10-31) nothing but the port numbers (1 and 2, as in every
replay of our library) comes from the original replay:

- Stage: from the video (scripts/m3_stage_id.py, small thumbnails compared with training games;
  164/164 held-out games right). Run `m3_stage_id.py --games <list>` before the closed loop.
- Characters: from perception (the most common guess per port; right on all 27 Stage C games).
- Starting file: an unrelated TRAINING-split replay with the same stage and characters, picked
  automatically (`template` column in results.csv). Its inputs are all replaced, its random
  numbers too (`--start-seed`), and its player names, connect codes and Slippi IDs are blanked
  in the output. One-game check: same result as starting from the original file.
- Random numbers: our own (`--start-seed N`, a seed made from N on the first frame we play;
  the recording stores it). No measurable cost (0.915 action vs 0.898-0.921 with the original's).
- Countdown: players can already move in the last 39 frames of the countdown (Slippi frames -39
  to -1). Without `--countdown` the loop took control only at frame 1, so those frames silently
  replayed the ORIGINAL game's real inputs (found 2026-10-31; all earlier scores include it).
  `--countdown` hands control over at the start of the countdown and chooses those inputs too;
  perception must then cover the countdown frames (`m3_predict.py --countdown`). Perception
  `percep_v4cd` was fine-tuned for this: countdown frames 0.47 -> 0.90 action, normal frames
  unchanged (0.96).

Measured cost of removing the countdown shortcut (validation, one seed, percep_v3 whose countdown
perception was weak): 0.868 action instead of 0.915. A run with percep_v4cd is in progress
(docs/experiments.md has the latest numbers).

Things it does NOT borrow: inputs (model only), states (engine's own), the per-frame random seeds
(the engine's random numbers run on by themselves; `--seeds N` lets the loop choose seeds that
fit the video), and the game's code list (our fixed UCF 0.84 list, data/m4/ucf084_gecko.bin).

## Limits found so far

- The engine implements UCF 0.84 only. Replays recorded with older UCF versions (about half of
  our library) cannot be reproduced exactly. That doesn't matter for new replays we write (they
  always carry UCF 0.84), but test games must be UCF 0.84 for a fair check
  (data/m4/test_games_ucf084.txt).
- Pokemon Stadium transformations are not implemented in the engine.
- Speed: with the recommended settings about 48 minutes per minute of game (2,870 s), 12 games
  at once on this server (16 cores), on the engine as sped up on 2026-10-30 (1.8x faster, same
  results). A 300-frame segment attempt takes about a minute per game.
- Run-to-run luck is large: the same settings with another random seed can move one game's
  action match by +-0.07 or more, mostly through a few segments where the rebuilt game drifts
  away from the video. Compare settings over several seeds before trusting a difference.

## How to run

```
# once: build the engine host and the helper tables
bash scripts/native_build.sh
.venv/bin/python scripts/m4_airborne_table.py

# once per batch of games: stage from the video, and perception including the countdown
.venv/bin/python scripts/m3_stage_id.py --games data/m4/test_games_ucf084.txt
.venv/bin/python scripts/m3_predict.py --run percep_v4cd --countdown --games data/m4/test_games_ucf084.txt

# the 12 validation games, first minute, nothing borrowed but ports (in tmux; ~1-1.5 hours)
# Earlier numbers WITH the borrowed countdown inputs (no --countdown --from-video):
#   validation 3 seeds 0.911 action / 0.820 action+anim / 0.988 stocks;
#   HELD-OUT TEST (15 test-split games, data/m4/test_split_ucf084.txt): 0.885 / 0.782 / 0.995
#   (plain 3 restarts: 0.855 / 0.751 / 0.962)
.venv/bin/python scripts/m4_closed_loop.py --game-list data/m4/test_games_ucf084.txt \
    --end 3600 --replan --lookahead 60 --restarts 3 --segment 300 \
    --extra-restarts 5 --restart-threshold 3.0 --backtrack 4 --start-seed 0 \
    --countdown --from-video --perception percep_v4cd --verify 4 --tag my_run
```

Output: data/m4/closed_loop/<tag>/ with one .slp per game, results.csv (scores against the
original replay) and config.json.
