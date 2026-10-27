"""
m3_frame_maps.py - recompute frame_map.npy for every rendered game from its saved timer crops,
with the current digit network (src/vid2slp/timer_digitnet.pt). No re-rendering needed.

Also prints, over all games: how many video frames got a game frame, how many game frames
(after the countdown) never appear in the video (skipped by Dolphin), and whether every map
is strictly increasing (it must be: a game frame can't come before an earlier one).

Run:
    .venv/bin/python scripts/m3_frame_maps.py
"""

import glob
import json
import os

import numpy as np

from vid2slp import hud_timer


def main():
    net = hud_timer.load_digit_net("cuda:0")
    totals = {"ended_early": 0, "games": 0, "video": 0, "labeled": 0, "skipped": 0, "not_increasing": 0}
    for info_path in sorted(glob.glob("data/m3/renders/*/info.json")):
        d = os.path.dirname(info_path)
        info = json.load(open(info_path))
        crops = np.load(f"{d}/timer_crops.npz")["crops"]
        # start from what the replay file says; frame_map_net checks it against the picture
        replay_start = info.setdefault("timer_start_seconds_replay", info["timer_start_seconds"])
        fmap, unused, shown_start = hud_timer.frame_map_net(crops, info["last_frame"], replay_start, net, "cuda:0")
        if shown_start != replay_start:
            print(f"{os.path.basename(d)[:12]}: timer starts at {shown_start} s on screen, replay says {replay_start} s")
        np.save(f"{d}/frame_map.npy", fmap)
        labeled = fmap[fmap >= 0]
        increasing = bool((np.diff(labeled) > 0).all())
        # game frames between the first and last labelled one that are missing from the video
        skipped = int(labeled.max() - labeled.min() + 1 - len(labeled)) if len(labeled) else 0
        # Desync check: in a few replays (seen 2026-10-27, games recorded on another build) the
        # playback goes wrong, the game ends early on screen ("Game!", then "Waiting for game")
        # and the video no longer matches the replay data, so its labels can't be trusted.
        # Sign: the timer stops being read well before the replay's last frame.
        ended_early = bool(len(labeled) == 0 or labeled.max() < info["last_frame"] - 60)
        if ended_early:
            print(f"{os.path.basename(d)[:12]}: playback ended early (last read game frame "
                  f"{labeled.max() if len(labeled) else None} of {info['last_frame']}); not used for training")
        info.update({"playback_ended_early": ended_early, "reader": "digitnet",
                     "timer_start_seconds": int(shown_start),
                     "labeled_frames": int(len(labeled)), "timer_rejected_reads": unused,
                     "game_frames_skipped_inside": skipped, "strictly_increasing": increasing})
        info.pop("game_frames_missing_from_video", None)
        json.dump(info, open(info_path, "w"), indent=1)
        totals["games"] += 1
        totals["video"] += len(fmap)
        totals["labeled"] += len(labeled)
        totals["skipped"] += skipped
        totals["not_increasing"] += not increasing
        totals["ended_early"] += ended_early
    t = totals
    print(f"{t['games']} games, {t['video']:,} video frames, {t['labeled']:,} labelled "
          f"({t['labeled'] / max(t['video'], 1):.4f}); game frames skipped by Dolphin: {t['skipped']:,} "
          f"({t['skipped'] / max(t['labeled'], 1):.4f}); maps not strictly increasing: {t['not_increasing']}; "
          f"playback ended early (excluded from training): {t['ended_early']}")


if __name__ == "__main__":
    main()
