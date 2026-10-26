# Audit of the old yt2slp (Codex) assets

Date: 2026-10-26. Everything below was measured directly from the files on disk,
not taken from the old README/docs. Nothing under `/home/prompter/yt2slp` was
modified. Raw tables behind the numbers are in `data/audit/` (gitignored), and
every number can be regenerated with the scripts named in each section.

Base path below: `OLD = /home/prompter/yt2slp/.docker-state/local`

---

## 0. Environment check (V100 server)

| Check | Result |
|---|---|
| GPUs | **Two cards.** nvidia-smi: GPU 0 = GeForce GT 1030 (2 GB), GPU 1 = Tesla PG500-216 (the 32 GB SXM2 V100, sm_70). Driver 580.178.04 |
| torch | 2.14.1+cu126, has sm_70 kernels, `cuda:0` in PyTorch **is the V100** (PyTorch orders fastest-first, nvidia-smi orders by PCI bus, so the numbering differs) |
| fp16 matmul 8192³ | 13.7 ms ≈ 80 TFLOPS, max relative error 2.6e-4 vs fp32 |
| autocast fp16 + GradScaler, SDPA | work |
| venv | `.venv` (created with `uv`, Python 3.12.15). No `pip` inside, which is normal for uv; use `uv pip install --python .venv/bin/python ...`. Has peppi-py 0.8.6, opencv, PyAV, pandas, pyarrow, pytest |
| Disk | `/` 936 G total, **137 G free** (85% used) |
| `/home/prompter/yt2slp/.docker-state` | **537 G** (vision-corpus-v3 239 G, vision-corpus-ceph-v1 127 G, vision-training 67 G, rest smaller) |
| ffmpeg / ffprobe / tmux | installed |

Nothing was broken, so nothing was "fixed". Added: `scripts/check_env.py` (re-runs these
checks and fails loudly if `cuda:0` is ever not the V100), and a `.gitignore`
(`.venv/ data/ logs/ *.pt *.mp4 *.png`, except PNGs under `docs/`).

---

## 1. Corpus size, resolution, fps, length

Script: `scripts/audit_corpus_stats.py` → `data/audit/corpus_clips.csv`

| | vision-corpus-v3 | vision-corpus-ceph-v1 |
|---|---|---|
| Clip folders | 300 | 160 |
| **Finished clips (have source.mp4)** | **186** | **98** |
| Failed clips (all capture attempts crashed, no video) | 114 (38%) | 62 (39%) |
| Distinct source replays | 97 | 98 |
| Resolution / fps / codec | 642×528, 60.0 fps, h264 (x264 CRF 18) | same |
| Frames per clip | exactly 1200 (20.0 s) in every clip | same |
| Total video | 62.0 min | 32.7 min |
| mp4 size | 1.92 GiB | 1.02 GiB |
| Folder size on disk | 239 G | 127 G |
| Splits (clips) | 131 train / 55 validation / **0 test** | 79 train / 19 validation / **0 test** |
| Replay in >1 split? | no (0) | no (0) |
| Slippi versions of source replays | 3.19.0–3.19.1 | 3.13–3.19 |

**Total usable: 284 clips = 94.7 minutes of video** (~341k frames).
The mp4s are only ~3 GB; the other ~360 GB is per-frame PNG dumps (screenshots of
every capture step and decoded frames), which are redundant with the mp4s.

The old split plan (`configs/learning-corpus-split-v3.json`) assigned 70 train /
30 validation / 22 test matches, grouped by recording date. **The 22 test matches were
never rendered.** It also notes "player disjointness unverified": the same players
(Slippi display names) can appear in both train and validation.

---

## 2. What produced the videos? Do they look like real footage?

**All of them were rendered by Slippi Playback Dolphin 3.5.2** (the extracted
AppImage at `OLD/playback/squashfs-root/AppRun`), run with `--batch` on
`recordings/vanilla.iso` (game ID GALE01 rev 2 = **NTSC 1.02**). The capture script
paused Dolphin, advanced one frame at a time, took a screenshot per step, and read an
on-screen frame counter to know which replay frame each screenshot was.
Evidence: `clips/<clip>/attempt-*/capture/capture.json` (`"command": [".../AppRun", "--exec=.../vanilla.iso", "--slippi-input=...", "--batch"]`, `"playback_version": "3.5.2"`).

Even the "native" fixture (`native-image-fixture-v1/source-120`, 96 frames, Fox vs Falco)
was rendered by the same Playback Dolphin. Its **.slp** came from the native engine
(`native-pad-calibration-v100-v1/roundtrip-hop-2.slp`), but its pixels did not.
**I found no video rendered by the native engine itself.**

How it looks: a clean emulator render at GameCube internal resolution (642×528, 4:3),
Battlefield only, Slippi display names under the HUD, no stream overlay, no
webcam, no letterboxing/scaling to 1080p, no YouTube re-compression, no crowd/commentary
framing. **It looks like engine/emulator output, not like tournament or YouTube footage.**
A perception model trained only on this will face a large domain gap on real VODs.

---

## 3. Per-frame labels: where, what format, and are they aligned?

### Where and what

Per finished clip, in `clips/<clip>/attempt-<id>/`:

| File | Content |
|---|---|
| `labels.json` | Made by `@slippi/slippi-js 9.1.3`. Game settings + 2400 rows (1200 frames × 2 players). Each row: replay `frame`, `port`, **effective controls** (buttons bitmask, stick x/y, c-stick x/y, trigger, as processed by the game), **physical controls** (raw buttons, L/R analog, raw stick X only), button press/release edges, and **post-frame state**: action state, animation, internal character id, x, y, facing, percent, stocks, airborne, jumps, 3 velocities. Each value has a mask flag. |
| `video/capture-index.json` | replay frame → video frame, plus which screenshot was used and the timer text read from it |
| `video/frame-labels.json` | the old pipeline's own timer-OCR / HUD cross-check results |
| `video/source.mp4` | the 1200-frame video |
| `shard.json` | split, source .slp path, interval, ports, characters |

Missing from labels: raw stick Y and raw C-stick values ("not exposed by pinned parser 9.1.3").
The effective (game-processed) values are present, which are what matter for replay.

### Alignment: what I measured

Script: `scripts/check_label_alignment.py` (3 sample clips with pictures) and
`--all` (every finished clip) → `data/audit/alignment_all.json`, log in `logs/`.

1. **Mapping is trivial.** In all 287 capture indexes (284 corpus clips + 3 pilot clips),
   video frame *i* = replay frame `start + i`, with no gaps or repeats.

2. **Label values vs the raw .slp: exact.** I re-parsed every source .slp with a
   *different* parser (peppi-py) and compared x, y, percent, action state, stocks,
   facing, stick x/y and buttons for every labeled row.
   Result over all 284 clips: **681,600 label rows checked, 0 mismatched values.**
   Gotcha found on the way: frame IDs inside a .slp are **not always contiguous**.
   Rollback writes some frame IDs twice (e.g. 6,636 rows for 6,633 distinct frames; 243 of the 284 source
   replays have this), so
   frames must be looked up by ID with "last copy wins", never by row number.
   Our own parser must do this.

3. **HUD damage onset (the most independent check).** For every hit where the label
   says a player's percent went up at replay frame *f*, I measured at which video
   frame the HUD damage digits first change (bright-grey pixel flips in the HUD box).
   Result over all 284 clips: **1,090 hits measured. 1,029 (94.4%) at offset 0, 61 (5.6%) at
   offset −1, and none at any other offset.** I looked at 3 of the 61 "−1" cases by hand.
   Two were noise: a small flicker (23–37 pixels, just over the threshold) the frame before
   the real change (≈1,500 pixels) at offset 0. The third was two hits on consecutive
   frames, where the HUD began shaking on the labeled frame of the *first* hit (offset 0 again).
   I did not check the other 58 by hand.
   Offset 0 means the HUD starts reacting (shaking) on exactly the labeled frame.
   By eye, the new number becomes readable one frame later.

4. **Timer.** The seconds digit of the in-game timer changes once per 60 frames.
   Result: in every clip the seconds digit flips at replay frame ≡ 1 (mod 60).
   277 clips show exactly 20 such flips and 7 show 19 (a flip lands on the clip edge).
   The neighbouring phases 0 and 2 (what an off-by-one error would produce) get 352 and
   344 flips, the same as the 341-per-phase average of background noise (bright things passing
   behind the timer box), so **no clip is off by one frame.**
   Caveat: the old pipeline used this same timer to choose frames, so this
   mostly confirms its bookkeeping; check 3 is the stronger evidence.

5. **Visual overlay check (3 clips).** Contact sheets show 5 consecutive video
   frames around a hit and around a landing, with the stored labels printed under
   each frame (event frame boxed in red):
   - `docs/audit/alignment-vision-corpus-v3-023af390.png` (v3, Fox vs Fox)
   - `docs/audit/alignment-vision-corpus-v3-04f3877e.png` (v3, Falco vs Fox)
   - `docs/audit/alignment-vision-corpus-ceph-v1-0211ba3f.png` (ceph, Fox vs Fox, older Slippi version)

   In all three, the hit spark and HUD shake appear on the labeled hit frame, and the
   character touches the ground on the frame where the label switches `air=1 → air=0`, `y → 0`.

**Verdict: labels are correct and frame-exact (offset 0, ±1 at worst) in this corpus.**
What is *not* verified: character world position vs pixel position. That needs the
camera, which the labels do not store (the ceph corpus has `geometry-snapshots.json` from
a separate camera capture, which I did not audit).

---

## 4. What do the two models predict, and what was measured?

Script: `scripts/summarize_old_models.py`. Both use the old project's "validation"
split (whole replays not used for training). **Neither ever evaluated its test split**
(`final_test_matches_opened: 0`). The validation split was also used to judge pass/fail,
so these are development numbers, not clean final-test numbers.

### root-tracking-context-v1: video → each character's on-screen position

- Predicts, per frame, the pixel location of each player's "root" (the game position
  point at the character's feet, projected into the 642×528 image). It does **not**
  predict world coordinates, action state, or inputs.
- Model: frozen ResNet18-based heatmap locator ("RootLocalizer64") plus a small trainable
  temporal residual over 5 frames (t−2..t+2, so it uses two future frames).
  Trained 20,000 steps on 48 clips, validated on 16 clips.

| Variant | Train mean / median px | **Validation mean / median px** | Val within 8 / 16 / 32 px |
|---|---|---|---|
| single-frame source model | 6.9 / 4.8 | 19.1 / 8.1 | 49% / 75% / 89% |
| 5 copies of one frame (control) | 5.9 / 4.3 | 19.1 / 8.1 | 49% / 75% / 89% |
| 5 real frames (final) | 4.7 / 3.8 | **17.4 / 7.7** | 52% / 78% / 90% |
| "always guess image centre" baseline | | 134 | |

For scale, a character is roughly 30–40 px tall in this video. So about half the time
the guess lands on the character's feet, and about 1 in 10 predictions is more than
32 px off (wrong spot). The big train/validation gap (4.7 vs 17.4 px) suggests overfitting.
The temporal context helped by about 9%. The root-review videos draw crosshairs at
the predicted and true feet; in a frame I sampled, the errors were 1–9 px.

### root-controller-expanded-v1: video → controller inputs

- Predicts, per frame and player: which of 12 buttons are held, stick and C-stick
  axes as 17 bins each, trigger, plus action state and airborne as side outputs.
  Input is image features only (no game state). Model: a bidirectional GRU over
  16-frame sequences of frozen image features; three variants ("global", "center",
  "ordered") vs an earlier "accepted" reference. 68 train / 29 validation clips.

Validation (64,075 player-frames), "ordered" variant; the others are within ±0.01:

| Output | Model | Always-neutral baseline |
|---|---|---|
| Stick X bin accuracy | 0.478 | 0.456 |
| Stick Y bin accuracy | 0.691 | 0.672 |
| C-stick X / Y | 0.965 / 0.948 | 0.966 / 0.946 |
| Trigger | 0.758 | **0.843 (model is worse)** |
| Button held, F1 | 0.33 | 0 |
| Button **press** F1 | **0.10** (precision ≈ 0.07: ~13,000 false presses vs ~1,000 correct) | 0 |
| Button release F1 | 0.026 | 0 |
| Action state accuracy (side output) | 0.56 | |

The result file itself says `criteria_met: false`, `development_controller_gain: false`,
`reconstructed_gameplay_validated: false`.

**What the controller-review.mp4 shows** (`validation-1`, 1842×720, 20 s, 60 fps):
the source clip on the left with the TRUE controls printed in text underneath, and
four text columns on the right with each model's PREDICTED buttons/stick/trigger for
P1 and P2. There is no visual overlay on the characters and no gameplay generated from
the predictions (the video says "No generated gameplay"). Watching it: during idle or
neutral moments all four columns read "nothing pressed", which is right. During
action they mostly miss. Example at replay frame 1413: P1 truly holds B with the stick
full left. All models get the stick but none predicts B, and the "accepted" model
invents L held + full trigger. P2's stick is truly full left, and only one of four models gets it.
Per-clip scores for the 4 review clips: press precision 0.05–0.10, with 2–3× more presses
predicted than actually happened.

**Verdict:** root tracking is a modest, partly working pixel locator. The controller
model is close to "always predict neutral" and is not usable. Neither produced a .slp.

---

## 5. Coverage: characters, stages, matchups

| | v3 | ceph |
|---|---|---|
| Stage | Battlefield only | Battlefield only |
| Matchups (clips) | Fox–Fox 99, Falco–Fox 87 | Falco–Fox 53, Fox–Fox 45 |
| Ports | always ports 1 & 2 | always ports 1 & 2 |
| Mode | human 1v1, 4 stock, 8 min, NTSC | same |

**Two characters, one stage, one HUD layout, one camera style.**

Separately, the old project keeps **4,755 .slp files (32 GB)** in `/home/prompter/yt2slp/recordings/`
(`scripts/scan_slp_library.py`): 4,706 readable, ~11,100 min (~185 h) of games,
3,675 1v1 and 1,030 4-player (doubles/FFA). The stages are mostly Battlefield (1,943),
then Yoshi's (783), FD (595), Stadium (477), Dream Land (445) and FoD (229).
Characters are heavily Fox (5,818 player-slots) and Falco (1,905), then Marth, Falcon,
Peach, Puff, Ganon and Sheik. 166 games are under 30 s.
I don't know where these replays came from (your own netplay, or a public dump).

---

## 6. Recommendation

**Re-render from the SLP library. Keep the old 284 clips only as a free, already-verified
pilot/sanity set for the first perception model.**

Why not build on the old clips:
- Too small: 95 minutes, 195 replays, one stage, two characters.
- No test split was ever rendered.
- They don't look like real footage (clean 642×528 emulator output), so they cannot
  measure the domain gap that matters.
- 38% of capture attempts failed, and the frame-stepping screenshot pipeline is slow and complex.
  It also left ~360 GB of PNGs for ~3 GB of video.
- The models trained on them are weak (see section 4).

What is worth keeping from the old project:
- The labels are correct and the alignment is frame-exact, so its rendering approach
  (Playback Dolphin + NTSC 1.02 ISO) works. The new pipeline should copy *that*, while
  rendering whole games at full speed into mp4 instead of stepping frame by frame.
- The HUD-onset and timer checks in `scripts/check_label_alignment.py` can become an
  automatic alignment test for our own renders.
- The 4,706-replay library in `recordings/` (if you're OK using it), the NTSC 1.02 ISO, and
  the extracted Playback Dolphin, all read in place without copying.
- Later (Stage C), the native engine host with checkpoint/restore for re-simulation.

Not verified in this audit: the native engine itself, the geometry/camera captures,
the other ~40 old experiment folders, and whether the Playback Dolphin AppImage still runs
outside the old Docker image.

---

## 7. Proposed plan: milestones 1–3

### Milestone 1: SLP data pipeline (CPU only, no GPU)
1. `scripts/build_slp_index.py`: scan the SLP folder(s) and write one manifest row per
   replay (sha256, path, Slippi version, stage, characters, ports, length, end method).
   Remove duplicates by hash. Default filter: 1v1, both human, legal stages, ≥ 30 s.
2. `scripts/make_splits.py`: freeze train/val/test **by replay file** (about 80/10/10),
   grouped by session/date and, where available, by player connect code, so the same
   set doesn't leak across splits. Record the seed.
3. `src/vid2slp/parse.py` + `scripts/parse_slps.py`: peppi-py → one Parquet file per
   replay with per-frame, per-player inputs (pre-frame: buttons, sticks, triggers, raw
   analog) and state (post-frame: action, position, facing, percent, stocks, airborne,
   jumps, velocities, hitlag, shield, state age). Handle the rollback duplicate frame IDs
   ("last copy wins").
4. `tests/`: a duplicate-frame test, plus a cross-check against the old corpus's
   `labels.json` (681,600 rows already shown to match .slp exactly), which makes a free
   correctness test for our parser.
   **Done when:** N replays parsed, failure count, hours, and disk size reported.

### Milestone 2: inverse dynamics (game state → controller inputs)
1. Input: both players' states over frames t−K … t+K. Output: that player's inputs at t
   (buttons, stick and C-stick as the game's processed values or bins, trigger).
2. Baselines on held-out **test replays**: "always neutral", plus a lookup table keyed by
   (action state, next action state). The old video model only beat "always neutral"
   by ~2 points on stick-X, so the bar to clear is clear.
3. Model: small 1D-conv or transformer, fp16 autocast + GradScaler, seed and config logged,
   one row per run in `docs/experiments.md`.
4. Metrics: stick/C-stick bin accuracy vs baseline, button-press F1 (exact frame and ±1 frame).
   Caveat: many different inputs give identical behavior, so per-frame accuracy
   *understates* quality. The real test is replaying predicted inputs and checking the
   game state stays the same (Stage C). I'll flag this rather than over-claim.

### Milestone 3: first perception model (video → game state)
1. Rendering: script Playback Dolphin to dump whole games (or long clips) at full speed
   straight to mp4. No per-frame PNGs. Measure throughput and MB/minute first
   (old clips: ~30 MB per rendered minute at CRF 18, so 100 h ≈ 180 GB, more than the
   137 GB free; we'll render selectively and/or use a higher CRF).
2. Alignment gate: before any training, new renders must pass the same HUD-onset
   and timer-phase test used in this audit (turned into a pytest).
3. Model: pretrained CNN per frame (plus small temporal context) predicting action state,
   facing, airborne, percent, stocks and on-screen position. World position needs the
   camera, so that's a separate step.
4. Metrics on held-out replays: action-state accuracy, percent exact-match, position
   error in pixels. Head-to-head with the old root tracker on its own validation clips
   (17.4 px mean) as a sanity bar. The old 284 clips double as a ready-made pilot set.
5. Domain gap: a handful of real VODs you provide, used only for visual checks at first.

### What I need from you
1. **SLP library path(s)** and roughly how many replays. Also: are the 4,706 replays in
   `/home/prompter/yt2slp/recordings/` yours, and may I *read* them in place (no copy)?
2. **Melee ISO**: may I use `/home/prompter/yt2slp/recordings/vanilla.iso` (verified
   NTSC 1.02) read-only in place, or do you have another path?
3. **Playback Dolphin**: OK to run the extracted AppImage from the old folder (read-only),
   or would you rather I download a fresh Slippi Playback build?
4. **Priorities**: which characters and stages matter most (default: 1v1, legal stages,
   most common characters first)?
5. **2–5 example videos** (YouTube links or files) of the kind you ultimately want to convert.
6. **Disk budget**: 137 GB free. How much may renders use? Is there other storage? (The
   old corpora hold ~360 GB of redundant PNG frame dumps. That's your call; I won't touch them.)

---

## 8. Cleanup done after the audit (2026-10-26, approved by the owner)

- **Moved** (renamed on the same disk, no copy) the 284 finished clips' `source.mp4`,
  `labels.json`, `capture-index.json` and `shard.json` into `data/old_pilot/` (3.9 GB),
  using `scripts/rescue_old_pilot_clips.py`. The audit scripts now read from there.
- **Copied** the old models' result JSONs (~170 KB) into `reference/old-results/`.
- **Deleted** 30 generated folders in `.docker-state/local/` (both vision corpora with
  their PNG dumps, `vision-training` checkpoints and review videos, the geometry/motion/
  localization/state corpora, the old `learning-cu126` Python environment, and smaller
  experiment outputs) plus `.docker-state/cache/uv`. Exact list: `logs/delete_list.txt`, log: `logs/cleanup.log`.
- **Kept:** `recordings/` (the only copy of the .slp library + ISO), Playback Dolphin
  (`local/playback`), the native engine build (`.docker-state/build`), `local/assets`,
  `local/native-*`, and small calibration/catalog files (~2.5 GB in total), plus all old code.
- Disk: **137 GB free → 666 GB free.**
- Not touched: Docker images (27 GB) and build cache (19 GB, 14 GB reclaimable), which the
  native engine's build environment uses; and a second **954 GB NVMe drive (`nvme0n1`)
  that is installed but unpartitioned and unmounted.**
