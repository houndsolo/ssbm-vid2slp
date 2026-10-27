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
| Speed, CPU (software OpenGL) | ~44 frames/s for one instance; ~66 frames/s total with 3 in parallel (CPU full). Dual-core mode did not help. |
| Speed, GPU (Vulkan, after installing `libnvidia-gl-580-server`, `scripts/install_gpu_graphics.sh`) | GT 1030: 101 frames/s alone, ~235 total with 4–6 in parallel. V100: 118 alone, 401 with 4, **498 with 6**, 528 with 8 (then the CPU is the limit). GPU and CPU pictures are the same (mean pixel difference 0.5 of 255, same frame count). |
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

The first templates were learned on Battlefield (dark sky); on Yoshi's Story (bright sky
behind the timer) 62% of frames were rejected. Replaced by a small digit network
(`hud_timer.DigitNet`, trained by `scripts/m3_timer_digits.py` on frames whose game frame is
certain): on that game it labels 8,760 of 8,784 game frames (vs 6,848), never disagrees
with the templates, and reads all 1,737 frames of the old verified clips correctly (0
confident misreads). A misread filter (game frames must rise at least as fast as video
frames, and at most ~2x) runs before gap filling. The renderer saves the timer area of every
frame (`timer_crops.npz`) so `scripts/m3_frame_maps.py` can redo all labels after retraining.

### Independent alignment check (`scripts/m3_check_alignment.py`)
Uses the damage % under the portraits instead of the timer: when the replay says damage went
up at game frame f, the HUD digits should start changing at the video frame labelled f.
First 5 games: 114 of 122 hits at offset 0 (93.4%), 8 at −1, none later. The old project's
frame-exact clips gave 94.4% at 0 on the same test, so the labels are frame-exact.
