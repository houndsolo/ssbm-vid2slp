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

Widescreen renders (scripts/m3_render.py --widescreen, since 2026-11-01) live in
data/m3/renders_ws/<sha256>/ with the same files; such a game is named "ws:<sha256>" here.

Made-up stream overlays (random_overlay, since 2026-11-01): banners, boxes and bars pasted onto
training clips so the model learns to ignore them. Real streams cover parts of the picture with
scoreboards, webcams and chat; an overlay taken from one real stream cut action accuracy from
0.963 to 0.812 (scripts/m3_overlay_test.py).

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
RENDERS_WS = "data/m3/renders_ws"
# other kinds of render (scripts/m3_render.py): key prefix -> folder
VARIANTS = {"ws": RENDERS_WS, "pt": "data/m3/renders_ports", "wp": "data/m3/renders_ws_ports"}


def game_dir(key):
    """Folder of a rendered game: "<sha256>" = normal render; "ws:<sha256>" = widescreen,
    "pt:<sha256>" = players moved to other ports, "wp:<sha256>" = both."""
    if ":" in key:
        prefix, sha = key.split(":", 1)
        return os.path.join(VARIANTS[prefix], sha)
    return os.path.join(RENDERS, key)


def game_sha(key):
    """The replay's sha256 of a game key."""
    return key.split(":", 1)[1] if ":" in key else key


def labels_path(key):
    """Label table of a rendered game: the render's own labels.parquet when it has one (port-moved
    renders: the moved game can differ from the original after a respawn), else data/parsed/."""
    own = os.path.join(game_dir(key), "labels.parquet")
    return own if os.path.exists(own) else f"data/parsed/{game_sha(key)}.parquet"
POS_SCALE = 100.0

CAT_FIELDS = {"character": 40, "action": 512, "stocks": 5}           # name -> number of classes
BIN_FIELDS = ["facing", "airborne"]
REG_FIELDS = {"x": POS_SCALE, "y": POS_SCALE, "percent": 100.0, "shield": 60.0}  # name -> divide by


COUNTDOWN_FIRST = -39   # players can move from this Slippi frame on (the last 39 countdown frames)


def extend_countdown(fmap, first_frame=COUNTDOWN_FIRST):
    """Give the countdown frames first_frame..-1 their video frames. The in-game timer doesn't run
    during the countdown, so frame_map.npy can't place them (-1 there). They are the video frames
    just before the first frame the timer places, counted backwards (Dolphin rarely skips a frame:
    about 0.2 %; checked 2026-10-31: no better frame offset on 27 games).
    Returns (game frame per video frame, valid mask): -1 is a real frame here, so use the mask."""
    fmap = fmap.copy()
    valid = fmap >= 0
    labelled = np.where(valid)[0]
    v1, f1 = labelled[0], fmap[labelled[0]]
    for v in range(v1 - 1, -1, -1):
        if f1 - (v1 - v) < first_frame:
            break
        fmap[v], valid[v] = f1 - (v1 - v), True
    return fmap, valid


def game_labels(sha, countdown=False):
    """Labels aligned to the video frames of one rendered game.

    Returns dict: name -> array [n_video_frames, 2] (slot 0 = lower port), plus
    "valid" [n_video_frames] (bool) and "game_frame" [n_video_frames].
    countdown: also label the countdown frames from -39 on (extend_countdown).
    """
    fmap = np.load(os.path.join(game_dir(sha), "frame_map.npy"))
    valid = fmap >= 0
    if countdown:
        fmap, valid = extend_countdown(fmap)
    df = pd.read_parquet(labels_path(sha))
    ports = sorted(df.port.unique())
    assert len(ports) == 2, f"{sha}: expected 2 players, got ports {ports}"
    first = int(df.frame.min())
    n_game = int(df.frame.max()) - first + 1
    out = {"game_frame": fmap, "valid": valid}
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

    def __init__(self, sha, countdown=False):
        self.sha = sha
        self.mp4 = os.path.join(game_dir(sha), "video.mp4")
        self.pts = pts_table(self.mp4)
        self.labels = game_labels(sha, countdown)
        n = len(self.labels["valid"])
        assert len(self.pts) == n, f"{sha}: {len(self.pts)} video frames but frame map has {n}"
        # clip starts where the clip has at least one labelled frame (skip the countdown)
        self.first_valid = int(np.argmax(self.labels["valid"]))
        # the video frame of game frame 0 (countdown clips are taken just before it)
        self.frame0 = int(np.argmax(self.labels["valid"] & (self.labels["game_frame"] >= 0)))
        # the last labelled video frame. Port-moved renders (m3_render.py --ports) often end
        # their game early (the moved game falls apart after the first death) while the video
        # runs on, unlabelled, for thousands of frames; clips must not start out there
        # (percep_v5real did, 2026-11-01: empty batches gave nan losses and skipped steps).
        self.last_valid = n - 1 - int(np.argmax(self.labels["valid"][::-1]))

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


# the timer area (fractions of width / height): never covered, real streams keep it visible
TIMER_AREA = (0.37, 0.66, 0.10, 0.18)
# the damage / stock display of ports 1 and 2 (bottom left half)
HUD_AREA = (0.0, 0.56, 0.74, 0.98)


def random_overlay(rng, h, w):
    """A made-up stream overlay for an h x w picture: (alpha [h, w] float 0..1, image [h, w, 3]).
    1-4 pieces, each one of:
      banner - a band along the top (scoreboard), 4-10 % of the height, part or all of the width
      box    - a box at an edge or corner (webcam, logo, chat), 12-35 % of width and height
      bar    - a low band at the bottom-left or bottom-right (chat, sponsor), 5-12 % high
    Content: a flat colour, a gradient, or blurred noise (camera-like), with random thin
    light/dark strokes (text-like). Opacity 0.6-1. The timer area is always left free."""
    img = np.zeros((h, w, 3), np.float32)
    alpha = np.zeros((h, w), np.float32)
    for _ in range(rng.randint(1, 4)):
        kind = rng.choice(["banner", "box", "box", "bar"])
        if kind == "banner":
            bh, bw = int(h * rng.uniform(0.04, 0.10)), int(w * rng.uniform(0.4, 1.0))
            y0, x0 = 0, rng.randint(0, w - bw)
        elif kind == "box":
            bh, bw = int(h * rng.uniform(0.12, 0.35)), int(w * rng.uniform(0.12, 0.35))
            y0 = rng.choice([0, h - bh, rng.randint(0, h - bh)])
            x0 = rng.choice([0, w - bw])
        else:
            bh, bw = int(h * rng.uniform(0.05, 0.12)), int(w * rng.uniform(0.2, 0.5))
            y0, x0 = h - bh, rng.choice([0, w - bw])
        style = rng.choice(["flat", "gradient", "noise"])
        colour = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
        if style == "flat":
            patch = np.broadcast_to(colour, (bh, bw, 3)).copy()
        elif style == "gradient":
            other = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
            t = np.linspace(0, 1, bh, dtype=np.float32)[:, None, None]
            patch = np.broadcast_to(colour * (1 - t) + other * t, (bh, bw, 3)).copy()
        else:
            nrng = np.random.default_rng(rng.randint(0, 2**31))
            coarse = nrng.uniform(0, 255, (max(bh // 16, 2), max(bw // 16, 2), 3)).astype(np.float32)
            patch = np.repeat(np.repeat(coarse, 16, 0), 16, 1)[:bh, :bw]
            if patch.shape[:2] != (bh, bw):
                patch = np.pad(patch, ((0, bh - patch.shape[0]), (0, bw - patch.shape[1]), (0, 0)), mode="edge")
        for _ in range(rng.randint(0, 6)):                     # text-like strokes
            sh, sw = rng.randint(1, max(bh // 6, 2)), rng.randint(2, max(bw // 2, 3))
            sy, sx = rng.randint(0, max(bh - sh, 1)), rng.randint(0, max(bw - sw, 1))
            patch[sy:sy + sh, sx:sx + sw] = 255.0 if rng.random() < 0.5 else 0.0
        img[y0:y0 + bh, x0:x0 + bw] = patch
        alpha[y0:y0 + bh, x0:x0 + bw] = rng.uniform(0.6, 1.0)
    x0, x1, y0, y1 = TIMER_AREA
    alpha[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 0.0
    if rng.random() < 0.8:
        # usually keep the two players' damage/stock display free too (streams rarely cover it;
        # when it is covered the clip's damage label can't be seen, which only adds noise)
        x0, x1, y0, y1 = HUD_AREA
        alpha[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 0.0
    return alpha, img


def paste_overlay(video, rng):
    """Paste one made-up overlay onto every frame of a clip (uint8 [T, H, W, 3]); it stays put
    for the whole clip, like a real one."""
    alpha, img = random_overlay(rng, video.shape[1], video.shape[2])
    a = alpha[None, :, :, None]
    return (video.astype(np.float32) * (1 - a) + img[None] * a).astype(np.uint8)


class RandomClips(torch.utils.data.IterableDataset):
    """Endless stream of random clips from a list of games (for training)."""

    def __init__(self, shas, length, size, seed=0, countdown_share=0.0, ws_shas=(), ws_share=0.0,
                 overlay_share=0.0):
        """countdown_share > 0: label the countdown frames -39..-1 too, and take that share of the
        clips around them (they are only ~1 % of a game's frames).
        ws_shas / ws_share: widescreen games ("ws:<sha256>") and the share of clips taken from them.
        overlay_share: share of clips that get a made-up stream overlay (paste_overlay)."""
        self.shas, self.length, self.size, self.seed = shas, length, size, seed
        self.countdown_share = countdown_share
        self.ws_shas, self.ws_share, self.overlay_share = list(ws_shas), ws_share, overlay_share

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = random.Random(self.seed * 1000 + wid)
        games = {}
        while True:
            if self.ws_shas and rng.random() < self.ws_share:
                sha = rng.choice(self.ws_shas)
            else:
                sha = rng.choice(self.shas)
            if sha not in games:
                # each worker loads a game's labels the first time it is picked
                games[sha] = Game(sha, countdown=self.countdown_share > 0)
            g = games[sha]
            if rng.random() < self.countdown_share:
                # a clip that contains countdown frames (anywhere from -39 to a clip after frame 0)
                start = rng.randint(max(0, g.first_valid - self.length + 1), g.frame0)
            else:
                start = rng.randint(g.first_valid, max(g.first_valid, min(len(g) - self.length,
                                                                          g.last_valid - self.length // 2)))
            video, labels = g.clip(start, self.length, self.size)
            if rng.random() < self.overlay_share:
                video = paste_overlay(video, rng)
            yield to_tensors(video, labels)
