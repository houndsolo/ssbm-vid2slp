#!/bin/bash
# real_batch.sh - download and cut every video in a list (data/real/download.csv by default),
# patiently: YouTube blocks this server for a while when it asks too much ("429 Too Many
# Requests" / "confirm you're not a bot", seen 2026-11-01). So:
#   1. wait until a simple request works again (checked every 30 minutes),
#   2. run real_prepare.py on the list (it skips videos already done, and a failed video
#      does not stop the others),
#   3. repeat until every video in the list has its games.json.
# Run in tmux:  bash scripts/real_batch.sh data/real/download.csv 2>&1 | tee logs/real_batch.log
cd /home/prompter/ssbm-vid2slp
LIST=${1:-data/real/download.csv}
TEST_ID=$(sed -n 2p "$LIST" | cut -d, -f1)          # first video of the list
while true; do
    # all done? (one games.json per video)
    TOTAL=$(($(wc -l < "$LIST") - 1))
    DONE=$(tail -n +2 "$LIST" | cut -d, -f1 | while read -r id; do [ -f "data/real/$id/games.json" ] && echo x; done | wc -l)
    echo "$(date '+%F %T') $DONE of $TOTAL videos done"
    [ "$DONE" -ge "$TOTAL" ] && break
    # wait until YouTube answers again
    until .venv/bin/yt-dlp --simulate --quiet --no-warnings "https://youtu.be/$TEST_ID" > /dev/null 2>&1; do
        echo "$(date '+%F %T') YouTube still blocking; waiting 30 min"
        sleep 1800
    done
    .venv/bin/python scripts/real_prepare.py --list "$LIST"
    sleep 600
done
echo "all done"
