"""
audit_corpus_stats.py - count and describe the old project's video corpora.

READ-ONLY: this script only reads files.

NOTE: the audit numbers (300 / 160 clip folders, 114 / 62 failed) were measured on
2026-10-26 against the original folders in /home/prompter/yt2slp. Those were deleted
afterwards; only the 284 finished clips survive, moved to data/old_pilot/ by
scripts/rescue_old_pilot_clips.py. Re-running now reports just those 284.

What it measures for each corpus (vision-corpus-v3 and vision-corpus-ceph-v1):
  - how many clip folders exist, and how many actually finished (have source.mp4)
  - video resolution, frame rate, frame count, and length (via ffprobe)
  - total size on disk of the mp4 files
  - which characters and stages appear, and in which train/val/test split
  - whether any replay FILE shows up in more than one split (that would leak)
  - whether labels.json exists and has one row per player per frame

Run it:
    .venv/bin/python scripts/audit_corpus_stats.py
It prints a report and also saves the raw per-clip table to
    data/audit/corpus_clips.csv
"""

import csv
import glob
import json
import os
import subprocess
from collections import Counter, defaultdict

BASE = "data/old_pilot"
CORPORA = ["vision-corpus-v3", "vision-corpus-ceph-v1"]
OUT_DIR = "data/audit"

# Slippi "external" character IDs -> names (as stored in .slp game start blocks).
CHARACTERS = {
    0: "Captain Falcon", 1: "Donkey Kong", 2: "Fox", 3: "Game & Watch", 4: "Kirby",
    5: "Bowser", 6: "Link", 7: "Luigi", 8: "Mario", 9: "Marth", 10: "Mewtwo",
    11: "Ness", 12: "Peach", 13: "Pikachu", 14: "Ice Climbers", 15: "Jigglypuff",
    16: "Samus", 17: "Yoshi", 18: "Zelda", 19: "Sheik", 20: "Falco",
    21: "Young Link", 22: "Dr. Mario", 23: "Roy", 24: "Pichu", 25: "Ganondorf",
}
# Melee stage IDs for the tournament-legal stages (anything else prints its number).
STAGES = {
    2: "Fountain of Dreams", 3: "Pokemon Stadium", 8: "Yoshi's Story",
    28: "Dream Land", 31: "Battlefield", 32: "Final Destination",
}


def ffprobe(path):
    """Ask ffprobe for the first video stream's size, frame rate, and frame count."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_packets,codec_name",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ]
    info = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    s = info["streams"][0]
    num, den = s["avg_frame_rate"].split("/")
    return {
        "width": s["width"],
        "height": s["height"],
        "codec": s["codec_name"],
        "fps": round(int(num) / int(den), 3) if int(den) else 0,
        "video_frames": int(s["nb_read_packets"]),  # one packet = one frame for h264 here
        "duration_s": round(float(info["format"]["duration"]), 3),
    }


def find_finished_attempt(clip_dir):
    """Return the attempt folder that has a video/source.mp4, or None if every attempt failed."""
    hits = sorted(glob.glob(os.path.join(clip_dir, "attempt-*", "video", "source.mp4")))
    return os.path.dirname(os.path.dirname(hits[-1])) if hits else None


def describe_clip(corpus, clip_dir):
    """Collect one row of facts about one clip folder."""
    clip_id = os.path.basename(clip_dir)
    row = {"corpus": corpus, "clip_id": clip_id,
           "attempts": len(glob.glob(os.path.join(clip_dir, "attempt-*")))}
    attempt = find_finished_attempt(clip_dir)
    row["finished"] = attempt is not None
    row["has_shard"] = os.path.exists(os.path.join(clip_dir, "shard.json"))
    if attempt is None:
        return row

    mp4 = os.path.join(attempt, "video", "source.mp4")
    row["mp4_bytes"] = os.path.getsize(mp4)
    row.update(ffprobe(mp4))

    # labels.json holds the per-frame labels taken from the .slp (settings + rows).
    labels_path = os.path.join(attempt, "labels.json")
    row["has_labels"] = os.path.exists(labels_path)
    if row["has_labels"]:
        labels = json.load(open(labels_path))
        settings = labels["settings"]
        row["stage"] = settings["stage"]
        row["characters"] = "+".join(sorted(CHARACTERS.get(p["character"], str(p["character"]))
                                           for p in settings["players"]))
        row["n_players"] = len(settings["players"])
        row["label_rows"] = len(labels["rows"])
        row["label_frames"] = len({r["frame"] for r in labels["rows"]})
        row["interval_start"], row["interval_end"] = labels["interval"]
        row["replay_sha256"] = labels["replay_sha256"]
        row["slp_version"] = settings["version"]

    # shard.json carries the train/validation/test split and the source .slp path.
    shard_path = os.path.join(clip_dir, "shard.json")
    if row["has_shard"]:
        shard = json.load(open(shard_path))
        row["split"] = shard.get("split")
        row["group"] = shard.get("group")
        row["slp_file"] = shard.get("file")
    return row


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_rows = []
    for corpus in CORPORA:
        clip_dirs = sorted(glob.glob(os.path.join(BASE, corpus, "clips", "*")))
        rows = [describe_clip(corpus, d) for d in clip_dirs]
        all_rows += rows
        done = [r for r in rows if r["finished"]]

        print(f"\n===== {corpus}")
        print(f"clip folders: {len(rows)}   finished (have source.mp4): {len(done)}   "
              f"failed: {len(rows) - len(done)}   with shard.json: {sum(r['has_shard'] for r in rows)}")
        if not done:
            continue
        print("resolutions:", Counter(f"{r['width']}x{r['height']}" for r in done))
        print("fps:", Counter(r["fps"] for r in done), " codec:", Counter(r["codec"] for r in done))
        print("video frames per clip:", Counter(r["video_frames"] for r in done))
        total_s = sum(r["duration_s"] for r in done)
        print(f"total video length: {total_s:.0f} s = {total_s / 60:.1f} min")
        print(f"total mp4 size: {sum(r['mp4_bytes'] for r in done) / 2**30:.2f} GiB")
        print("label rows == 2 x frames:",
              Counter(r.get("label_rows") == 2 * r.get("label_frames", -1) for r in done),
              " label frames per clip:", Counter(r.get("label_frames") for r in done))
        print("stages:", Counter(STAGES.get(r.get("stage"), r.get("stage")) for r in done))
        print("matchups:", Counter(r.get("characters") for r in done))
        print("slp versions:", Counter(r.get("slp_version") for r in done))
        print("splits (clips):", Counter(r.get("split") for r in done))

        # Count distinct replays and check that no replay file is in two splits.
        splits_per_replay = defaultdict(set)
        for r in done:
            splits_per_replay[r.get("replay_sha256")].add(r.get("split"))
        leaks = [k for k, v in splits_per_replay.items() if len(v) > 1]
        replays_per_split = Counter(next(iter(v)) for v in splits_per_replay.values() if len(v) == 1)
        print(f"distinct replays: {len(splits_per_replay)}   replays per split: {dict(replays_per_split)}"
              f"   replays in >1 split (leak): {len(leaks)}")
        print("capture-date groups:", len({r.get('group') for r in done}))

    # Save the raw table so numbers in the audit doc can be re-checked.
    fields = sorted({k for r in all_rows for k in r})
    with open(os.path.join(OUT_DIR, "corpus_clips.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {OUT_DIR}/corpus_clips.csv ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
