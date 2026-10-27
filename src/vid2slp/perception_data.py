"""
perception_data.py - training data for milestone 3 (video frames -> game state).

Each rendered game (scripts/m3_render.py) has, in data/m3/renders/<sha256>/:
    video.mp4      the video (H.264)
    frame_map.npy  for every video frame (in decode order): its game frame, or -1 if unknown
The labels come from the milestone-1 table data/parsed/<sha256>.parquet, looked up by game
frame. So label row i belongs to video frame i, and is masked out when frame_map[i] == -1.

Players: "slot 0" = the lower controller port, "slot 1" = the higher one. The HUD shows the
lower port on the left, so the model can tell them apart.

What we predict per player (first version; all visible on screen, directly or over time):
    character   internal character id (classes, 0..39)
    action      action state id (classes, 0..511)
    x, y        world position (regression; divided by POS_SCALE)
    facing      1 = right, 0 = left
    airborne    1 = in the air
    percent     damage % (regression; divided by 100)
    stocks      stocks left (classes 0..4)
    shield      shield size 0..60 (regression; divided by 60)

Video timestamps: Dolphin skips some frames, so the mp4 has gaps in its timestamps. We never
convert time to frame number by multiplying; instead we build each video's sorted list of
timestamps once (pts_table) and look frames up in it.
"""

import os
import random
import subprocess

import av
import numpy as np
import pandas as pd
import torch

RENDERS = "data/m3/renders"
POS_SCALE = 100.0

CAT_FIELDS = {"character": 40, "action": 512, "stocks": 5}           # name -> number of classes
BIN_FIELDS = ["facing", "airborne"]
REG_FIELDS = {"x": POS_SCALE, "y": POS_SCALE, "percent": 100.0, "shield": 60.0}  # name -> divide by


def game_labels(sha):
    """Labels aligned to the video frames of one rendered game.

    Returns dict: name -> array [n_video_frames, 2] (slot 0 = lower port), plus
    "valid" [n_video_frames] (bool) and "game_frame" [n_video_frames].
    """
    fmap = np.load(os.path.join(RENDERS, sha, "frame_map.npy"))
    df = pd.read_parquet(f"data/parsed/{sha}.parquet")
    ports = sorted(df.port.unique())
    assert len(ports) == 2, f"{sha}: expected 2 players, got ports {ports}"
    first = int(df.frame.min())
    n_game = int(df.frame.max()) - first + 1
    out = {"game_frame": fmap, "valid": fmap >= 0}
    for name in list(CAT_FIELDS) + BIN_FIELDS + list(REG_FIELDS):
        table = np.zeros((n_game, 2), np.float32)
        for slot, port in enumerate(ports):
            rows = df[df.port == port]
            table[rows.frame.to_numpy() - first, slot] = rows[name].to_numpy()
        idx = np.clip(fmap - first, 0, n_game - 1)
        out[name] = table[idx]
    out["facing"] = (out["facing"] > 0).astype(np.float32)
    out["stocks"] = np.clip(out["stocks"], 0, 4)
    return out


def pts_table(mp4):
    """Sorted presentation timestamps of every frame (index i = video frame i)."""
    res = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "packet=pts",
                          "-of", "csv=p=0", mp4], capture_output=True, text=True, check=True)
    return np.sort(np.array([int(x) for x in res.stdout.split() if x.strip()], dtype=np.int64))


def read_clip(mp4, pts, start, length, size):
    """Decode video frames start .. start+length-1 as uint8 [length, H, W, 3] (resized to size=(W,H))."""
    container = av.open(mp4)
    stream = container.streams.video[0]
    # jump to the keyframe at or before the first wanted frame, then decode forward
    container.seek(int(pts[start]), stream=stream, backward=True, any_frame=False)
    frames = []
    for frame in container.decode(stream):
        i = int(np.searchsorted(pts, frame.pts))
        if i < start:
            continue
        frames.append(frame.to_ndarray(format="rgb24", width=size[0], height=size[1]))
        if len(frames) == length:
            break
    container.close()
    if len(frames) < length:
        raise RuntimeError(f"{mp4}: got {len(frames)} of {length} frames from {start}")
    return np.stack(frames)


class Game:
    """One rendered game: video path, timestamp table and labels."""

    def __init__(self, sha):
        self.sha = sha
        self.mp4 = os.path.join(RENDERS, sha, "video.mp4")
        self.pts = pts_table(self.mp4)
        self.labels = game_labels(sha)
        n = len(self.labels["valid"])
        assert len(self.pts) == n, f"{sha}: {len(self.pts)} video frames but frame map has {n}"
        # clip starts where the clip has at least one labelled frame (skip the countdown)
        self.first_valid = int(np.argmax(self.labels["valid"]))

    def __len__(self):
        return len(self.pts)

    def clip(self, start, length, size):
        video = read_clip(self.mp4, self.pts, start, length, size)
        labels = {k: v[start:start + length] for k, v in self.labels.items()}
        return video, labels


def to_tensors(video, labels):
    """numpy clip -> torch tensors (video stays uint8; normalised on the GPU)."""
    out = {"video": torch.from_numpy(video).permute(0, 3, 1, 2).contiguous()}   # [T, 3, H, W]
    out["valid"] = torch.from_numpy(labels["valid"].astype(np.bool_))
    for name in CAT_FIELDS:
        out[name] = torch.from_numpy(labels[name].astype(np.int64))
    for name in BIN_FIELDS:
        out[name] = torch.from_numpy(labels[name].astype(np.float32))
    for name, scale in REG_FIELDS.items():
        out[name] = torch.from_numpy(labels[name].astype(np.float32) / scale)
    return out


class RandomClips(torch.utils.data.IterableDataset):
    """Endless stream of random clips from a list of games (for training)."""

    def __init__(self, shas, length, size, seed=0):
        self.shas, self.length, self.size, self.seed = shas, length, size, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = random.Random(self.seed * 1000 + wid)
        games = {}
        while True:
            sha = rng.choice(self.shas)
            if sha not in games:
                games[sha] = Game(sha)   # each worker loads a game's labels the first time it is picked
            g = games[sha]
            start = rng.randint(g.first_valid, len(g) - self.length)
            yield to_tensors(*g.clip(start, self.length, self.size))
