#!/usr/bin/env bash
# relocate_old_project.sh - one-time move (owner's request 2026-10-27): bring the last useful
# pieces of the old project into this project, then remove /home/prompter/yt2slp entirely.
#
#   /home/prompter/yt2slp/recordings                    -> data/recordings   (the replays + ISO; ONLY copy)
#   .../.docker-state/local/playback                    -> data/tools/playback   (Playback Dolphin)
#   .../.docker-state/build                             -> data/tools/native-engine/build
#   .../.docker-state/local/assets, local/native-*      -> data/tools/native-engine/
#
# Everything is on the same disk, so "mv" only renames: nothing is copied, nothing can be
# half-written. Afterwards the replays are made read-only so a stray "rm -rf" can't delete them.
# The path updates in code and data files are done by scripts/relocate_paths.py (run after this).
#
# Run only when no render is running:
#     bash scripts/relocate_old_project.sh
set -euo pipefail

OLD=/home/prompter/yt2slp
NEW=/home/prompter/ssbm-vid2slp/data

# 1. Safety checks: nothing may be using the old folder right now.
if docker ps --format '{{.Names}}' | grep -q '^vid2slp-playback'; then
    echo "A Playback Dolphin container is running (a render?). Stopping without changes."; exit 1
fi
if [ "$(stat -c %d $OLD)" != "$(stat -c %d $NEW)" ]; then
    echo "Old and new folders are on different disks; mv would copy. Stopping."; exit 1
fi
if [ -e "$NEW/recordings" ] || [ -e "$NEW/tools" ]; then
    echo "$NEW/recordings or $NEW/tools already exists. Stopping."; exit 1
fi
before=$(find "$OLD/recordings" -type f | wc -l)

# 2. Move (rename) the pieces.
mkdir -p "$NEW/tools/native-engine"
mv "$OLD/recordings" "$NEW/recordings"
mv "$OLD/.docker-state/local/playback" "$NEW/tools/playback"
mv "$OLD/.docker-state/build" "$NEW/tools/native-engine/build"
mv "$OLD/.docker-state/local/assets" "$NEW/tools/native-engine/assets"
for d in "$OLD"/.docker-state/local/native-*; do
    mv "$d" "$NEW/tools/native-engine/"
done

# 3. Check the replays all arrived.
after=$(find "$NEW/recordings" -type f | wc -l)
echo "files in recordings: before $before, after $after"
[ "$before" = "$after" ] || { echo "COUNT MISMATCH - stopping before deleting anything"; exit 1; }

# 4. The old folder must now hold only empty folders; then remove it.
left=$(find "$OLD" -mindepth 1 -not -type d | wc -l)
if [ "$left" != "0" ]; then
    echo "$left files still in $OLD - not deleting it:"; find "$OLD" -not -type d | head; exit 1
fi
find "$OLD" -depth -type d -empty -delete
[ ! -e "$OLD" ] && echo "removed $OLD"

# 5. Protect the replays: read-only for everyone (undo with chmod -R u+w if ever needed).
chmod -R a-w "$NEW/recordings"
echo "data/recordings is now read-only"
