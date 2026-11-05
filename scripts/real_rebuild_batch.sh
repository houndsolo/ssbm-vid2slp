#!/bin/bash
# real_rebuild_batch.sh - rebuild many real games into .slp replays, several at the same time.
#
# Why: real_rebuild.py rebuilds the games of ONE video (one worker per game). To see which
# mistakes are common we want one game from each of many videos, so this runs real_rebuild.py
# once per (video, game) line of a list, PARALLEL of them at once (xargs -P).
#
# Restartable: a game whose output folder already has results.csv is skipped.
# Each game logs to logs/real_rebuild_batch/<tag>/<id>_game<k>.log.
#
# Run in tmux:
#   bash scripts/real_rebuild_batch.sh data/real/rebuild_batch1.csv b1 12 2>&1 | tee logs/real_rebuild_batch_b1.log
# The list is a csv whose first two columns are video id and game number (header line skipped).
cd /home/prompter/ssbm-vid2slp
LIST=$1
TAG=$2
PARALLEL=${3:-12}
LOGS=logs/real_rebuild_batch/$TAG
mkdir -p "$LOGS"

one_game() {
    id=$1; k=$2; tag=$3; logs=$4
    if [ -f "data/real/$id/rebuild/$tag/results.csv" ]; then
        echo "$(date '+%F %T') skip $id game $k (done)"
        return
    fi
    echo "$(date '+%F %T') start $id game $k"
    # --id=<id> (with "=") so a video id starting with "-" is not read as an option
    .venv/bin/python -u scripts/real_rebuild.py --id="$id" --games "$k" --tag "$tag" > "$logs/${id}_game$k.log" 2>&1
    echo "$(date '+%F %T') end $id game $k (exit $?)"
}
export -f one_game

tail -n +2 "$LIST" | cut -d, -f1,2 | tr ',' ' ' |
    xargs -P "$PARALLEL" -L 1 bash -c 'one_game "$0" "$1" '"$TAG $LOGS"
echo "$(date '+%F %T') all done"
