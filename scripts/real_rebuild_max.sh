#!/bin/bash
# real_rebuild_max.sh - rebuild a whole set as accurately as we can (2026-11-10).
#
# Runs real_rebuild.py --preset max (about 2.4x the search attempts) several times with different
# random seeds, all at the same time, each game on one engine (that uses the machine best: one
# game spread over many engines is only ~3.4x faster on 16 threads, while 12 separate runs each
# get a whole thread). Afterwards scripts/real_pick.py keeps, per game, the run that fits the
# video best (uses only the video). On the validation renders this gave 0.919 action match vs
# 0.882 for one run with the recommended settings.
#
# Run in tmux:
#   bash scripts/real_rebuild_max.sh 5njRxIAgS-o 1,2,3 4 v3max 2>&1 | tee logs/real_rebuild_max.log
# Outputs: data/real/<id>/rebuild/<tag>_s<seed>/ for each seed; logs in logs/<tag>_s<seed>.log
cd /home/prompter/ssbm-vid2slp
ID=$1
GAMES=$2
SEEDS=${3:-4}
TAG=$4
for s in $(seq 0 $((SEEDS - 1))); do
    .venv/bin/python -u scripts/real_rebuild.py --id="$ID" --games "$GAMES" --preset max --seed "$s" \
        --engines 1 --tag "${TAG}_s$s" > "logs/${TAG}_s$s.log" 2>&1 &
    sleep 60       # start them one after another so the GPU steps (models) don't all overlap
done
wait
echo "$(date '+%F %T') all runs done"
