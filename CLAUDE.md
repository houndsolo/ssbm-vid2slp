# vid2slp: SSBM video -> Slippi (.slp) replays

## Goal
Input: video of Melee matches (YouTube/VODs, 60fps).
Output: .slp replays with per-frame controller inputs.
Exact inputs are not recoverable, so "behaviorally equivalent" is the target.

## About the owner
Network engineer learning automation/Linux; NOT a software developer.
- Write simple, basic Python with thorough comments explaining what and why.
- Prefer small scripts over clever abstractions.
- After every task, summarize in plain English: what changed, how to run it,
  and the measured result.
- If something is uncertain or unverified, say so plainly. Do not smooth it over.

## Machine and hardware constraints
- Work happens on the V100 server as user `prompter`.
- This project: /home/prompter/ssbm-vid2slp
- Old Codex project: /home/prompter/yt2slp (READ-ONLY, see "Old attempt" below)
- GPU: V100 32GB (compute capability sm_70).
- torch must be cu126 wheels, torch<2.15 (newer wheels dropped Volta).
- fp16 autocast + GradScaler only (no bf16).
- No flash-attn 2; use F.scaled_dot_product_attention.
- No torch.compile unless tested.
- Always run long jobs inside tmux and log output to a file.

## Architecture
A) Inverse dynamics: per-frame game-state window -> controller inputs.
   Trained on parsed SLPs only (no video needed). Build and measure this first.
B) Perception: video frames -> game state (character, action state, position,
   facing, percent, stocks). Trained on SLPs rendered to video with
   per-frame labels taken from the SLP.
C) Assembly: states -> inputs -> write .slp, then verify by playing the result
   back in Slippi Playback Dolphin next to the source video. Later, re-simulate
   with the old project's engine to correct drift.

## Known hard problems
- Need 60fps source; 30fps loses information.
- Frame alignment (use the in-game timer and match start).
- Camera zoom/position changes make world-position estimation hard.
- Stream overlays, crops, compression (domain gap from rendered to real video).
- Many stick values give identical behavior, so inputs are ambiguous.

## Status: starting point from the old Codex attempt
- The old project built: a headless native Melee Unlocked engine host with
  checkpoint/restore and SLP recording, a generated "vision corpus" of clips,
  and two experimental models (root tracking, root controller).
- It has only produced comparison videos that draw predictions over source
  footage. NO learned reconstruction (video -> playable .slp) has been generated.
  Nothing has been shown to work end to end.
- Its README is out of date relative to its experiments. Treat doc claims as
  unverified. Trust only numbers you measure yourself on held-out data.

## Old attempt: what is left (audited and cleaned up 2026-10-26)
The audit is in docs/audit-old-assets.md. On 2026-10-26 the owner approved deleting
the old generated corpora, training runs and caches (~530 GB). What remains, and
should be treated as READ-ONLY inputs:
- /home/prompter/yt2slp/recordings/ : the owner's .slp library (4,755 files) and
  vanilla.iso (NTSC 1.02). This is the ONLY copy of the replays; never delete it.
- /home/prompter/yt2slp/.docker-state/local/playback/squashfs-root/AppRun : Slippi
  Playback Dolphin 3.5.2 (the renderer the old corpus used).
- /home/prompter/yt2slp/.docker-state/build/ : the native engine build (melee_headless),
  kept for Stage C re-simulation. Also kept: local/assets, local/native-* folders.
- Old code: /home/prompter/yt2slp (git, commit 17a9abe), cloned in reference/yt2slp.
Moved into this project:
- data/old_pilot/ : the 284 old clips (mp4 + labels.json), verified frame-aligned.
- reference/old-results/ : result JSONs of the old root-tracking/controller models.

## FIRST TASK (DONE 2026-10-26): audit the old assets
Report back in plain English, with numbers:
1. How many clips are in each corpus, total size, resolution, fps, length?
2. What produced the videos (native engine render vs Slippi Playback Dolphin)?
   Do they look like real YouTube/tournament footage or like engine output?
3. Are per-frame labels stored with each clip? Where, in what format, and are
   they aligned to video frames? Verify with an overlay check on a few clips.
4. What do the root-tracking and root-controller models actually predict, and
   what held-out metrics were measured (if any)?
5. Which characters/stages/matchups are covered?
6. Recommendation: reuse these clips for Stage B, or re-render from the SLP library.

## Rules
- Never claim an improvement without a number on HELD-OUT replays.
- Split train/val/test by replay FILE, never by frame (frames from one game leak).
- Log every experiment as a row in docs/experiments.md (date, change, metric, verdict).
- Keep data, renders, and checkpoints out of git.
- Set random seeds and record the config for every run.
- Ask before deleting anything or copying large data (check disk with du/df first).

## Layout
- src/vid2slp/  library code (parsing, models, assembly)
- scripts/      small runnable scripts, one job each, heavily commented
- data/         parsed SLP tables, renders (gitignored)
- docs/         experiments.md, notes
- tests/        pytest checks, including label/frame alignment tests
- reference/    read-only notes about the old project (code stays in /home/prompter/yt2slp)
