# Milestone 3: perception (video → game state)

Goal: from video frames, recover each character's state (action, position, facing,
percent, stocks...). First step: make training videos from the .slp library, where every
video frame is labelled with its exact game frame, so the labels can be taken from the
parsed replay (`data/parsed/<sha256>.parquet`).

## Rendering replays to video (`scripts/m3_render.py`)
Playback Dolphin 3.5.2 (same Docker setup as milestone 2) plays the whole replay at full
speed and dumps every displayed frame with its built-in FFmpeg (`src/vid2slp/playback.py`,
`dump_video=`). Measured 2026-10-27:

| | |
|---|---|
| Frame size | 584×480, 60 fps (clean emulator output, no overlay) |
| Speed | ~44 rendered frames/s per Dolphin instance (software OpenGL is the bottleneck; dual-core mode did not help). 3 instances in parallel. |
| Size | H.264 CRF 23: ~16–21 MB per minute (bright stages more). The whole library would be ~130 GB. |
| Direct H.264 from Dolphin | Works, but the last ~1 s is lost because Dolphin is stopped, not closed. So we dump lossless FFV1 and compress afterwards. |

### Dolphin skips some frames, so frame counting is not enough
Re-rendering 4 of the old project's verified clips and matching them picture-by-picture
showed that "video frame k = game frame k + constant" does **not** hold: the constant was
−119, −117 and −112 at the start of different games, and drifted by 1–3 frames within
10 seconds of video. The same frames are skipped at full and at normal speed, so it is
deterministic (from the game/emulator), not a performance problem.

### Fix: read the in-game timer on every frame (`src/vid2slp/hud_timer.py`)
During an 8-minute countdown the timer (mm:ss plus small centiseconds) shows a different
value on every game frame. The value shown on game frame f is Slippi's own formula
(slippi-js `frameToGameTimer`): whole seconds = trunc(480 − f/60), centiseconds =
ceil(((60 − f mod 60) mod 60) × 99 / 59). So reading the timer gives the exact game frame.

- Digit templates are built automatically (`scripts/m3_timer_templates.py`) from 1,737
  frames of re-rendered old clips whose frame numbers are known from the picture matching.
- Check on those frames: **1,723 read correctly, 14 rejected (unreadable), 0 misread.**
- Unreadable frames are filled only when certain: between two read frames a and b, if the
  game frame advanced by exactly b − a, nothing was skipped in between.
- The countdown (frames −123..0) all shows 08:00 00 and stays unlabelled.

Known problem (being fixed): the first templates were learned on Battlefield (dark sky).
On Yoshi's Story (bright sky behind the timer) 62% of frames were rejected in a test game.
The renderer therefore saves the timer area of every frame (`timer_crops.npz`) so the
reader can be improved and re-run without re-rendering.
