# Stage C: closed loop (the game engine in the loop)

Written 2026-10-30. Numbers are in docs/experiments.md (rows of 2026-10-30).

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

## Things it borrows from the original game (shortcuts still to remove)

- Stage, characters, costumes, ports and the game's starting random seed come from the original
  replay used as a template. The video would have to provide these.
- Frames 0 and 1 are played before the loop takes control (they use the model's inputs).

Things it does NOT borrow: inputs (model only), states (engine's own), the per-frame random seeds
(the engine's random numbers run on by themselves; `--seeds N` lets the loop choose seeds that
fit the video), and the game's code list (our fixed UCF 0.84 list, data/m4/ucf084_gecko.bin).

## Limits found so far

- The engine implements UCF 0.84 only. Replays recorded with older UCF versions (about half of
  our library) cannot be reproduced exactly. That doesn't matter for new replays we write (they
  always carry UCF 0.84), but test games must be UCF 0.84 for a fair check
  (data/m4/test_games_ucf084.txt).
- Pokemon Stadium transformations are not implemented in the engine.
- Speed: about 1 hour per minute of game for 12 games at once on this server (16 cores), with
  the engine before 2026-10-30's speedup (the engine is now 1.8x faster, same results).
- Run-to-run luck is large: the same settings with another random seed can move one game's
  action match by +-0.07 or more, mostly through a few segments where the rebuilt game drifts
  away from the video. Compare settings over several seeds before trusting a difference.

## How to run

```
# once: build the engine host and the helper tables
bash scripts/native_build.sh
.venv/bin/python scripts/m4_airborne_table.py

# the 12 test games, first minute, best settings so far (in tmux; ~1-2 hours)
.venv/bin/python scripts/m4_closed_loop.py --game-list data/m4/test_games_ucf084.txt \
    --end 3600 --replan --lookahead 60 --restarts 3 --segment 300 --verify 4 --tag my_run
```

Output: data/m4/closed_loop/<tag>/ with one .slp per game, results.csv (scores against the
original replay) and config.json.
