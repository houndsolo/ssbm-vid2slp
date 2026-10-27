"""
m3_timer_templates.py - build the digit templates used by src/vid2slp/hud_timer.py.

How (no hand labelling):
  1. Take a few of the old project's clips (data/old_pilot). Their video frames are verified
     frame-exact (docs/audit-old-assets.md): old video frame i = replay frame start + i.
  2. Re-render the same replay with our own Playback Dolphin frame dump.
  3. For a sample of old frames, find the best-matching frame of our render (small grey
     images, smallest pixel difference). Where neighbouring samples agree on the offset, the
     frames between them are known: our frame k = replay frame k + offset.
  4. For those known frames, the timer digits are known from the formula (frame_to_timer),
     so we average the white-pixel image of each digit -> one template per digit.
  5. Check: read the timer on every known frame with the new templates; it must give the
     formula's value on every one of them. Prints the result.

Run:
    .venv/bin/python scripts/m3_timer_templates.py
Writes src/vid2slp/timer_templates.npz (small, kept in git).
"""

import argparse
import glob
import os

import av
import cv2
import numpy as np
import pandas as pd

from vid2slp import hud_timer
from vid2slp.playback import regenerate

# old verified clips: chosen so that together they show every digit 0-9 in the seconds boxes
CLIPS = [
    "data/old_pilot/vision-corpus-v3/clips/023af3908c0f56b38bea244432f823757689ce0dd101973ed9e6697b5cdd3835-1113-2312",
    "data/old_pilot/vision-corpus-ceph-v1/clips/0211ba3ff2089136386d993ccb32f051468de006c9fcc6f1916637174e3f8d0c-1949-3148",
    "data/old_pilot/vision-corpus-ceph-v1/clips/021ef38e405d1993adc074e7787b8f740f0ece2aa036a88217048d879a43640f-4956-6155",
]
SPAN = 600          # how many frames of each old clip to use
SAMPLE_EVERY = 10   # compare every 10th old frame


def small_grey(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, (146, 120), interpolation=cv2.INTER_AREA).astype(np.float32)


def decode(path, limit=None):
    """All frames of a video as RGB arrays (in decode order = our frame index)."""
    out = []
    for f in av.open(path).decode(video=0):
        out.append(f.to_ndarray(format="rgb24"))
        if limit and len(out) >= limit:
            break
    return out


def known_frames(clip, work):
    """Render the clip's replay, align it to the old verified video; return {our index: replay frame}."""
    base = os.path.basename(clip)
    sha, start = base.split("-")[0], int(base.split("-")[1])
    index = pd.read_csv("data/index/slp_index.csv")
    slp = index[index.sha256 == sha].path.iloc[0]
    out = os.path.join(work, sha[:12])
    avi = os.path.join(out, "user", "Dump", "Frames", "framedump0.avi")
    if not os.path.exists(avi):
        regenerate(slp, out, end=start + SPAN + 60, timeout=1200,
                   dump_video={"codec": "ffv1", "fmt": "avi", "kbps": 25000})
    ours = decode(avi)
    ours_small = np.stack([small_grey(x) for x in ours])
    old = decode(glob.glob(os.path.join(clip, "attempt-*", "video", "source.mp4"))[0], SPAN)
    # offset (= replay frame - our index) at each sampled old frame
    samples = []
    for i in range(0, SPAN, SAMPLE_EVERY):
        err = ((ours_small - small_grey(old[i])) ** 2).mean(axis=(1, 2))
        k = int(err.argmin())
        samples.append((start + i, start + i - k))
    # frames between two consecutive samples with the SAME offset are taken as known
    known = {}
    for (f0, off0), (f1, off1) in zip(samples[:-1], samples[1:]):
        if off0 == off1:
            for f in range(f0, f1 + 1):
                known[f - off0] = f
    offsets = sorted({o for _, o in samples})
    print(f"{sha[:12]}: offsets seen {offsets}, known frames {len(known)}")
    return ours, known, hud_timer_start(slp)


def hud_timer_start(slp):
    import peppi_py
    return peppi_py.read_slippi(slp, skip_frames=True).start.timer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="data/m3/timer_calibration")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    big_sum = np.zeros((10,) + (hud_timer.BIG_BOXES[0][3] - hud_timer.BIG_BOXES[0][2],
                                 hud_timer.BIG_BOXES[0][1] - hud_timer.BIG_BOXES[0][0]))
    small_sum = np.zeros((10,) + (hud_timer.SMALL_BOXES[0][3] - hud_timer.SMALL_BOXES[0][2],
                                   hud_timer.SMALL_BOXES[0][1] - hud_timer.SMALL_BOXES[0][0]))
    big_n, small_n = np.zeros(10), np.zeros(10)
    checks = []
    for clip in CLIPS:
        ours, known, start_seconds = known_frames(clip, args.work)
        for k, f in known.items():
            digits = hud_timer.frame_to_timer(f, start_seconds)
            big, small = hud_timer.digit_crops(hud_timer.timer_crop(ours[k]))
            for j in range(4):
                big_sum[int(digits[j])] += big[j]
                big_n[int(digits[j])] += 1
            for j in range(2):
                small_sum[int(digits[4 + j])] += small[j]
                small_n[int(digits[4 + j])] += 1
            checks.append((hud_timer.timer_crop(ours[k]), digits))
    print("examples per big digit:  ", big_n.astype(int).tolist())
    print("examples per small digit:", small_n.astype(int).tolist())
    assert (big_n > 0).all() and (small_n > 0).all(), "some digit never seen; add a clip"
    big_t = big_sum / big_n[:, None, None]
    small_t = small_sum / small_n[:, None, None]
    np.savez_compressed(hud_timer.TEMPLATES, big=big_t.astype(np.float32), small=small_t.astype(np.float32))
    # self-check on the known frames
    templates = hud_timer.load_templates()
    right = sum(hud_timer.read_timer(img, templates)[0] == d for img, d in checks)
    print(f"check: timer read correctly on {right}/{len(checks)} known frames")
    print(f"wrote {hud_timer.TEMPLATES}")


if __name__ == "__main__":
    main()
