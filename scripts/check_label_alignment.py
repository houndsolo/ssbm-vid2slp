"""
check_label_alignment.py - do the old corpus labels line up with the video frames?

READ-ONLY. Clips are read from data/old_pilot/ (moved there from the old project by
scripts/rescue_old_pilot_clips.py); the source .slp files are read from
/home/prompter/yt2slp/recordings/. Outputs go to data/audit/ and docs/audit/.

The old corpus stores, for every clip:
  - source.mp4         : 1200 rendered video frames
  - labels.json        : per-frame game state + controller inputs, keyed by REPLAY frame number
  - capture-index.json : replay frame -> video frame (we found it is always video = replay - start)

We test alignment three independent ways:

  1. LABELS vs RAW SLP. Re-parse the original .slp with peppi-py (a different parser from
     the one the old project used) and compare x, y, percent, action state, stocks, facing,
     stick and buttons for every labeled frame. This says whether the label VALUES are right.

  2. HUD PERCENT ONSET. Whenever the labels say a player's damage went up at replay frame f,
     look at the HUD damage digits in the video around video frame (f - start) and find the
     first frame where the white digit pixels change. If labels and video are aligned, this
     offset should be the same small constant (ideally 0) for every hit in every clip.

  3. TIMER ROLLOVER. The in-game timer's "seconds" digits change once every 60 frames.
     Find the video frames where they change, convert to replay frame numbers, and look at
     (replay frame mod 60). If every clip gives the same value, every clip has the same
     video<->label phase. (Caveat: the old pipeline used this same timer to pick frames,
     so this check mostly confirms their bookkeeping; check 2 is the more independent one.)

It also draws a contact sheet (PNG) for a few clips: video frames around a hit and a
landing, with the stored label printed under each frame, so a human can eyeball it.

Run:
    .venv/bin/python scripts/check_label_alignment.py              # 3 sample clips + sheets
    .venv/bin/python scripts/check_label_alignment.py --all        # metrics on every clip
"""

import argparse
import csv
import glob
import json
import os
from collections import Counter

import cv2
import numpy as np
import peppi_py

OLD = "/home/prompter/yt2slp"
BASE = "data/old_pilot"
DATA_OUT = "data/audit"
SHEET_OUT = "docs/audit"

# Three sample clips for the contact sheets: v3 Fox-Fox, v3 Falco-Fox, ceph (older Slippi version).
SAMPLE_CLIPS = [
    "vision-corpus-v3/clips/023af3908c0f56b38bea244432f823757689ce0dd101973ed9e6697b5cdd3835-1113-2312",
    None,  # filled in at runtime: first finished v3 clip with Falco
    None,  # filled in at runtime: first finished ceph clip
]

# Screen boxes in native video pixels (x0, y0, x1, y1), read off a gridded frame.
# Every clip uses controller ports 1 and 2, so the HUD is always in these two places.
PERCENT_BOX = {0: (55, 440, 165, 495), 1: (205, 440, 320, 495)}
TIMER_SECONDS_BOX = (310, 52, 370, 90)
# A pixel counts as "HUD white" if it is bright (max channel > 150) and nearly grey
# (channels within 40 of each other). The digits are only ~205 bright after video
# compression and slightly tinted, so a strict "all channels > 200" test misses them.
WHITE_LEVEL = 150
MAX_TINT = 40
CHANGE_PIXELS = 15         # this many flipped white pixels = "the digits changed"
SEARCH_OFFSETS = range(-4, 7)  # look this many frames before/after the labeled hit


def finished_attempt(clip_dir):
    hits = sorted(glob.glob(os.path.join(clip_dir, "attempt-*", "video", "source.mp4")))
    return os.path.dirname(os.path.dirname(hits[-1])) if hits else None


def strict_white_mask(frame, box):
    """Stricter test for the timer: every channel above 200. The timer digits are pure
    white, and the strict test ignores the background showing through the timer box."""
    x0, y0, x1, y1 = box
    return (frame[y0:y1, x0:x1] > 200).all(axis=2)


def white_mask(frame, box):
    """Boolean image: True where the pixel is near-white (HUD digits are white)."""
    x0, y0, x1, y1 = box
    crop = frame[y0:y1, x0:x1].astype(np.int16)
    brightest, darkest = crop.max(axis=2), crop.min(axis=2)
    return (brightest > WHITE_LEVEL) & (brightest - darkest < MAX_TINT)


def read_video(path, keep_full):
    """Decode every frame. Always keep small HUD masks; keep full frames only if asked."""
    cap = cv2.VideoCapture(path)
    masks = {"p0": [], "p1": [], "sec": []}
    full = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        masks["p0"].append(white_mask(fr, PERCENT_BOX[0]))
        masks["p1"].append(white_mask(fr, PERCENT_BOX[1]))
        masks["sec"].append(strict_white_mask(fr, TIMER_SECONDS_BOX))
        if keep_full:
            full.append(fr)
    return {k: np.array(v) for k, v in masks.items()}, full


def changed(masks, i):
    """How many white pixels flipped between video frame i-1 and i."""
    if i <= 0 or i >= len(masks):
        return 0
    return int((masks[i] != masks[i - 1]).sum())


def slp_lookup(slp_path):
    """Parse the raw .slp with peppi-py and return {frame_id: row_index}.

    Frame IDs in a .slp are NOT always contiguous: rollback can write the same frame
    id more than once. The LAST copy is the final, real one, so later rows win here.
    """
    game = peppi_py.read_slippi(slp_path)
    ids = game.frames.id.to_numpy()
    index = {int(fid): row for row, fid in enumerate(ids)}
    return game, index, len(ids) - len(index)


def compare_labels_to_slp(labels, game, index):
    """Count frames where labels.json disagrees with our own parse of the .slp."""
    mism = Counter()
    checked = 0
    for r in labels["rows"]:
        row = index[r["frame"]]
        lead = game.frames.ports[r["port"]].leader
        s = r["state"]["values"]
        e = r["effective"]["values"]
        ours = {
            "x": lead.post.position.x[row].as_py(),
            "y": lead.post.position.y[row].as_py(),
            "percent": lead.post.percent[row].as_py(),
            "action": lead.post.state[row].as_py(),
            "stocks": lead.post.stocks[row].as_py(),
            "facing": lead.post.direction[row].as_py(),
            "stick_x": lead.pre.joystick.x[row].as_py(),
            "stick_y": lead.pre.joystick.y[row].as_py(),
            "buttons": lead.pre.buttons[row].as_py(),
        }
        theirs = {"x": s["x"], "y": s["y"], "percent": s["percent"], "action": s["action"],
                  "stocks": s["stocks"], "facing": s["facing"], "stick_x": e["stick_x"],
                  "stick_y": e["stick_y"], "buttons": e["buttons"]}
        for k in ours:
            # float32 values: compare with a tiny tolerance for JSON float printing
            if abs(float(ours[k]) - float(theirs[k])) > 1e-4:
                mism[k] += 1
        checked += 1
    return checked, mism


def percent_onsets(labels, start, masks):
    """For each labeled damage increase, find the video offset where the HUD digits change."""
    by_key = {(r["frame"], r["port"]): r["state"]["values"] for r in labels["rows"]}
    onsets = []
    events = []
    for port in (0, 1):
        m = masks[f"p{port}"]
        for f in range(start + 5, start + len(m) - 7):
            now, before = by_key.get((f, port)), by_key.get((f - 1, port))
            if not now or not before:
                continue
            # A hit: damage went up and the player did not just die/respawn.
            if now["percent"] > before["percent"] + 0.5 and now["stocks"] == before["stocks"]:
                vid = f - start
                # The HUD must have been still just before the search window, otherwise
                # an earlier shake animation would confuse the onset.
                quiet = all(changed(m, vid + k) < CHANGE_PIXELS for k in range(-4, -1))
                if not quiet:
                    continue
                onset = next((k for k in SEARCH_OFFSETS if changed(m, vid + k) >= CHANGE_PIXELS), None)
                onsets.append(onset)
                events.append((f, port, onset))
    return onsets, events


def timer_phase(start, masks):
    """Replay-frame-number mod 60 at each seconds-digit change in the video."""
    m = masks["sec"]
    return [(start + i) % 60 for i in range(1, len(m)) if changed(m, i) >= CHANGE_PIXELS]


def contact_sheet(clip_id, labels, start, frames, events, out_path):
    """Draw rows of video frames around one hit and one landing, with labels under each."""
    by_key = {(r["frame"], r["port"]): r["state"]["values"] for r in labels["rows"]}
    rows = []
    # Row A: the first clean hit we found. Row B: first landing (airborne -> grounded).
    hit = next((e for e in events if e[2] is not None), None)
    landing = None
    for f in range(start + 3, start + len(frames) - 3):
        a, b = by_key.get((f - 1, 0)), by_key.get((f, 0))
        if a and b and a["airborne"] and not b["airborne"]:
            landing = (f, 0)
            break
    picks = []
    if hit:
        picks.append(("HIT on P%d" % (hit[1] + 1), hit[0], hit[1]))
    if landing:
        picks.append(("LANDING of P1", landing[0], landing[1]))

    scale = 0.75
    for title, f0, port in picks:
        cells = []
        for k in (-2, -1, 0, 1, 2):
            f = f0 + k
            img = frames[f - start].copy()
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            s = by_key[(f, port)]
            o = by_key[(f, 1 - port)]
            text = [
                f"replay frame {f}  (video {f - start})" + ("  <== event" if k == 0 else ""),
                f"LABEL P{port + 1}: {s['percent']:.0f}%  stocks {s['stocks']}  action {s['action']}",
                f"  x={s['x']:.1f} y={s['y']:.1f} air={int(s['airborne'])} face={s['facing']:+.0f}",
                f"LABEL P{2 - port}: {o['percent']:.0f}%  stocks {o['stocks']}",
            ]
            panel = np.full((92, img.shape[1], 3), 255, np.uint8)
            for j, t in enumerate(text):
                cv2.putText(panel, t, (5, 20 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 0, 180) if k == 0 else (0, 0, 0), 1, cv2.LINE_AA)
            cell = np.vstack([img, panel])
            if k == 0:
                cv2.rectangle(cell, (0, 0), (cell.shape[1] - 1, cell.shape[0] - 1), (0, 0, 255), 4)
            cells.append(cell)
            cells.append(np.full((cell.shape[0], 6, 3), 128, np.uint8))
        row = np.hstack(cells[:-1])
        header = np.full((34, row.shape[1], 3), 40, np.uint8)
        cv2.putText(header, f"{title}   clip {clip_id[:12]}...{clip_id[-10:]}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        rows += [header, row]
    cv2.imwrite(out_path, np.vstack(rows))


def check_clip(clip_dir, make_sheet):
    clip_id = os.path.basename(clip_dir)
    attempt = finished_attempt(clip_dir)
    labels = json.load(open(os.path.join(attempt, "labels.json")))
    shard = json.load(open(os.path.join(clip_dir, "shard.json")))
    start = labels["interval"][0]

    masks, frames = read_video(os.path.join(attempt, "video", "source.mp4"), keep_full=make_sheet)
    game, index, dup_ids = slp_lookup(os.path.join(OLD, shard["file"]))
    checked, mism = compare_labels_to_slp(labels, game, index)
    onsets, events = percent_onsets(labels, start, masks)
    phases = timer_phase(start, masks)

    result = {
        "clip_id": clip_id,
        "corpus": clip_dir.split("/")[-3],
        "video_frames": len(masks["sec"]),
        "label_rows_checked": checked,
        "label_mismatches": sum(mism.values()),
        "mismatch_fields": dict(mism),
        "slp_duplicate_frame_ids": dup_ids,
        "hits_measured": len(onsets),
        "hit_onset_offsets": dict(Counter(onsets)),
        "timer_changes": len(phases),
        "timer_phase_mod60": dict(Counter(phases)),
    }
    if make_sheet:
        os.makedirs(SHEET_OUT, exist_ok=True)
        out = os.path.join(SHEET_OUT, f"alignment-{result['corpus']}-{clip_id[:8]}.png")
        contact_sheet(clip_id, labels, start, frames, events, out)
        result["sheet"] = out
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="measure every finished clip (no sheets)")
    args = ap.parse_args()
    os.makedirs(DATA_OUT, exist_ok=True)

    finished = [d for d in sorted(glob.glob(BASE + "/vision-corpus-*/clips/*"))
                if finished_attempt(d) and os.path.exists(d + "/shard.json")]
    if args.all:
        targets = [(d, False) for d in finished]
    else:
        falco = next(d for d in finished if "vision-corpus-v3/" in d
                     and json.load(open(d + "/shard.json"))["slot_characters"] != [2, 2])
        ceph = next(d for d in finished if "ceph" in d)
        targets = [(BASE + "/" + SAMPLE_CLIPS[0], True), (falco, True), (ceph, True)]

    results = []
    for n, (d, sheet) in enumerate(targets, 1):
        r = check_clip(d, sheet)
        results.append(r)
        print(f"[{n}/{len(targets)}] {r['corpus']} {r['clip_id'][:16]} "
              f"label mismatches {r['label_mismatches']}/{r['label_rows_checked']} "
              f"hit offsets {r['hit_onset_offsets']} timer flips at phase 1: "
              f"{r['timer_phase_mod60'].get(1, 0)}"
              + (f" -> {r['sheet']}" if sheet else ""), flush=True)

    # Corpus-wide totals
    all_onsets, all_phase = Counter(), Counter()
    for r in results:
        all_onsets.update(r["hit_onset_offsets"])
        all_phase.update(r["timer_phase_mod60"])
    print("\nTOTAL clips:", len(results))
    print("label rows checked:", sum(r["label_rows_checked"] for r in results),
          " mismatched values:", sum(r["label_mismatches"] for r in results))
    print("clips whose .slp has duplicate (rollback) frame ids:",
          sum(1 for r in results if r["slp_duplicate_frame_ids"]))
    print("HUD-percent onset offset (video frame - label frame) over all hits:", dict(all_onsets))
    # The seconds digit should flip once per 60 frames, always at the same phase.
    # A 20 s clip has 20 flips. Phases 0 and 2 are the "off by one frame" neighbours;
    # flips at any other phase are noise from bright things moving behind the timer.
    top_phase, top_count = all_phase.most_common(1)[0]
    neighbours = all_phase.get((top_phase - 1) % 60, 0) + all_phase.get((top_phase + 1) % 60, 0)
    bad_clips = [r["clip_id"] for r in results if r["timer_phase_mod60"].get(top_phase, 0) < 19]
    print(f"timer seconds-digit flips at phase {top_phase} (replay frame mod 60): {top_count} "
          f"(~20 expected per clip x {len(results)} clips = {20 * len(results)}); "
          f"at neighbouring phases (off-by-one): {neighbours}; "
          f"clips with fewer than 19 flips at phase {top_phase}: {len(bad_clips)}")

    name = "alignment_all.json" if args.all else "alignment_samples.json"
    json.dump(results, open(os.path.join(DATA_OUT, name), "w"), indent=1)
    print("wrote", os.path.join(DATA_OUT, name))


if __name__ == "__main__":
    main()
