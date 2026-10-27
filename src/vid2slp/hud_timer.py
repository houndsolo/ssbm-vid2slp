"""
hud_timer.py - read the in-game timer from a rendered video frame, to know its exact game frame.

Why: when Playback Dolphin dumps video at full speed it occasionally skips a frame
(measured 2026-10-27: the video->game frame offset drifted by 1-2 frames within 300 frames).
So "video frame k = game frame k + constant" is NOT safe. But the timer in the top middle
of the screen (mm:ss and small centiseconds) shows a different value on every game frame
of an 8-minute countdown, so reading it tells us exactly which game frame each video frame is.

The formula for what the timer shows on game frame f (f >= 0; frame 0 = "GO") is the one
Slippi's own library uses (slippi-js frameToGameTimer):
    whole seconds = trunc(start_seconds - f / 60)
    centiseconds  = ceil(((60 - f % 60) % 60) * 99 / 59)
During the countdown (frames -123..0) it shows the start time, e.g. 08:00 00, so those
frames cannot be told apart and get no label.

Reading the digits: the timer is drawn at a fixed place on screen (it does not move with the
camera), in white. We cut out each digit box, keep the white pixels, and compare with one
template per digit (an average white-pixel image), built by scripts/m3_timer_templates.py
from renders whose alignment was verified against the old project's frame-exact clips.
Geometry is for Playback Dolphin's 584x480 frame dump.
"""

import math
import os

import numpy as np

FRAME_SIZE = (584, 480)  # width, height of Playback Dolphin's frame dump

# The whole timer area (x0, x1, y0, y1) on a 584x480 render. scripts/m3_render.py saves this
# region of every frame ("timer crops") so the timer can be re-read later without re-rendering.
TIMER_BOX = (220, 382, 52, 80)
# Digit boxes (x0, x1, y0, y1) INSIDE the timer crop. "big" = minutes/seconds digits,
# "small" = centiseconds. Order: m0 m1 s0 s1 c0 c1. (Measured: full-frame x 222.., y 53..)
BIG_BOXES = [(2, 26, 1, 26), (28, 52, 1, 26), (66, 90, 1, 26), (91, 115, 1, 26)]
SMALL_BOXES = [(122, 141, 6, 27), (141, 160, 6, 27)]

TEMPLATES = os.path.join(os.path.dirname(__file__), "timer_templates.npz")


def frame_to_timer(frame, start_seconds=480):
    """The 6 digits 'mmsscc' the timer shows on game frame `frame` (frame >= 0)."""
    seconds = math.trunc(start_seconds - frame / 60)
    centi = math.ceil(((60 - frame % 60) % 60) * 99 / 59)
    return f"{seconds // 60:02d}{seconds % 60:02d}{centi:02d}"


def timer_lookup(last_frame, start_seconds=480):
    """dict: timer digits -> game frame, for frames 1..last_frame (all distinct)."""
    table = {}
    for f in range(1, last_frame + 1):
        table[frame_to_timer(f, start_seconds)] = f
    return table


def white_mask(rgb):
    """True where a pixel is bright and grey/white (the timer digits), False elsewhere."""
    rgb = rgb.astype(np.int16)
    return (rgb.min(axis=-1) > 170) & ((rgb.max(axis=-1) - rgb.min(axis=-1)) < 60)


def timer_crop(rgb):
    """Cut the timer area out of a full 584x480 frame."""
    x0, x1, y0, y1 = TIMER_BOX
    return rgb[y0:y1, x0:x1]


def digit_crops(crop):
    """Cut the 6 digit boxes out of a timer crop; returns (big [4,h,w], small [2,h,w]) white masks."""
    m = white_mask(crop)
    big = np.stack([m[y0:y1, x0:x1] for x0, x1, y0, y1 in BIG_BOXES])
    small = np.stack([m[y0:y1, x0:x1] for x0, x1, y0, y1 in SMALL_BOXES])
    return big, small


def load_templates(path=TEMPLATES):
    t = np.load(path)
    return t["big"], t["small"]  # each [10 digits, h, w], values 0..1


def read_timer(crop, templates):
    """Read the timer from a timer crop. Returns (digits 'mmsscc' or None, worst_margin).

    For each digit box, the template with the smallest mismatch wins. The answer is rejected
    (None) if any box is a poor match or nearly as close to a second digit (unclear reading).
    """
    big_t, small_t = templates
    big, small = digit_crops(crop)
    out, worst = "", 1.0
    for boxes, temps in ((big, big_t), (small, small_t)):
        for box in boxes:
            err = np.abs(temps - box[None]).mean(axis=(1, 2))   # mismatch per digit 0..9
            order = np.argsort(err)
            best, second = err[order[0]], err[order[1]]
            margin = second - best
            if best > 0.12 or margin < 0.03:
                return None, 0.0
            out += str(order[0])
            worst = min(worst, margin)
    return out, worst


def frame_map(crops, last_frame, start_seconds, templates):
    """Game frame of every video frame, from its timer crop. Returns (int32 array, rejected reads).
    -1 = unknown (countdown, unreadable timer that cannot be filled with certainty)."""
    lookup = timer_lookup(last_frame, start_seconds)
    frames, rejected = [], 0
    for crop in crops:
        digits, _ = read_timer(crop, templates)
        if digits is None:
            rejected += 1
        frames.append(lookup.get(digits, -1) if digits else -1)
    fmap = np.array(frames, dtype=np.int32)
    fill_gaps(fmap)
    return fmap, rejected


def fill_gaps(fmap):
    """Fill unknown frames ONLY when it is certain: between two known video frames a and b,
    if the game frames go up by exactly (b - a), no frame was dropped in between, so the
    frames in between are consecutive game frames. Changes fmap in place."""
    known = np.where(fmap >= 0)[0]
    for a, b in zip(known[:-1], known[1:]):
        if b - a > 1 and fmap[b] - fmap[a] == b - a:
            fmap[a + 1:b] = fmap[a] + np.arange(1, b - a)
