"""
rescue_old_pilot_clips.py - ONE-OFF (run 2026-10-26, before deleting the old corpora).

Moves the small, useful part of the old project's 284 finished clips into this project:
    source.mp4, capture-index.json, labels.json, shard.json   (about 3.7 GB total)
and leaves behind the ~360 GB of redundant PNG frame dumps, which were then deleted.

Why keep them: the audit (docs/audit-old-assets.md) verified these labels are exact and
frame-aligned, so they are a ready-made test set for the first perception model.

It MOVES (renames) files rather than copying: same disk, instant, no extra space used.
The folder layout is kept the same as the old one, so the audit scripts still work:
    data/old_pilot/<corpus>/clips/<clip>/shard.json
    data/old_pilot/<corpus>/clips/<clip>/attempt-<id>/labels.json
    data/old_pilot/<corpus>/clips/<clip>/attempt-<id>/video/source.mp4
    data/old_pilot/<corpus>/clips/<clip>/attempt-<id>/video/capture-index.json

Run (only once; a second run finds nothing left to move):
    .venv/bin/python scripts/rescue_old_pilot_clips.py
"""

import glob
import os

OLD = "/home/prompter/yt2slp/.docker-state/local"
NEW = "data/old_pilot"

moved_clips = 0
for corpus in ["vision-corpus-v3", "vision-corpus-ceph-v1"]:
    for mp4 in sorted(glob.glob(f"{OLD}/{corpus}/clips/*/attempt-*/video/source.mp4")):
        attempt = os.path.dirname(os.path.dirname(mp4))
        clip = os.path.dirname(attempt)
        rel_clip = os.path.relpath(clip, OLD)          # <corpus>/clips/<clip>
        rel_attempt = os.path.relpath(attempt, OLD)    # <corpus>/clips/<clip>/attempt-<id>
        wanted = [
            (f"{clip}/shard.json", f"{NEW}/{rel_clip}/shard.json"),
            (f"{attempt}/labels.json", f"{NEW}/{rel_attempt}/labels.json"),
            (mp4, f"{NEW}/{rel_attempt}/video/source.mp4"),
            (f"{attempt}/video/capture-index.json", f"{NEW}/{rel_attempt}/video/capture-index.json"),
        ]
        # Only move clips that are complete (all four files present).
        if not all(os.path.exists(src) for src, _ in wanted):
            print("skipping incomplete clip:", rel_clip)
            continue
        for src, dst in wanted:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(src, dst)
        moved_clips += 1

print("moved clips:", moved_clips)
