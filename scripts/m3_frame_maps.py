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
    totals = {"games": 0, "video": 0, "labeled": 0, "skipped": 0, "not_increasing": 0}
    for info_path in sorted(glob.glob("data/m3/renders/*/info.json")):
        d = os.path.dirname(info_path)
        info = json.load(open(info_path))
        crops = np.load(f"{d}/timer_crops.npz")["crops"]
        fmap, unused = hud_timer.frame_map_net(crops, info["last_frame"], info["timer_start_seconds"], net, "cuda:0")
        np.save(f"{d}/frame_map.npy", fmap)
        labeled = fmap[fmap >= 0]
        increasing = bool((np.diff(labeled) > 0).all())
        # game frames between the first and last labelled one that are missing from the video
        skipped = int(labeled.max() - labeled.min() + 1 - len(labeled)) if len(labeled) else 0
        info.update({"reader": "digitnet", "labeled_frames": int(len(labeled)), "timer_rejected_reads": unused,
                     "game_frames_skipped_inside": skipped, "strictly_increasing": increasing})
        info.pop("game_frames_missing_from_video", None)
        json.dump(info, open(info_path, "w"), indent=1)
        totals["games"] += 1
        totals["video"] += len(fmap)
        totals["labeled"] += len(labeled)
        totals["skipped"] += skipped
        totals["not_increasing"] += not increasing
    t = totals
    print(f"{t['games']} games, {t['video']:,} video frames, {t['labeled']:,} labelled "
          f"({t['labeled'] / max(t['video'], 1):.4f}); game frames skipped by Dolphin: {t['skipped']:,} "
          f"({t['skipped'] / max(t['labeled'], 1):.4f}); maps not strictly increasing: {t['not_increasing']}")


if __name__ == "__main__":
    main()
