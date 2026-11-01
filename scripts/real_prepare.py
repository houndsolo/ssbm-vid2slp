"""
real_prepare.py - turn a real Melee video (YouTube, stream VOD) into one folder per game, in the
same format as our own renders (data/m3/renders/<sha>/), so perception can run on it.

Why: everything so far was measured on our own renders (Slippi Playback Dolphin, 584 x 480, no
overlay). A real video is different: 1920 x 1080, stream overlays, several games in one file,
"Waiting for game" screens in between. This script does the cutting; it does not judge quality.

Steps:
  1. Download (yt-dlp, 60 fps, video only) to data/real/<id>/source.mp4, if not there yet.
  2. Find where the game's 4:3 picture is in the video ("crop"), then shrink that part to our
     render size, 584 x 480 -> data/real/<id>/full_584.mp4 (near-lossless).
     - With --widescreen (Slippi's 16:9 widescreen code, used by the Coinbox streams) only a
       4:3 part in the middle is kept: the game draws its HUD (timer, damage, stocks) at 4:3
       size in the middle and just shows more of the stage left and right.
     - A stream layout often shrinks and shifts the game a little (here: about 92 % size,
       moved right and down by a few pixels), and the timer reader needs the digits within
       1-2 pixels of where they are in our renders. So the crop is FITTED: the white timer
       digits are averaged over ~300 frames (a stable picture of where each digit sits) and
       scaled/shifted until they overlay the same average from our renders best.
       Measured 2026-10-31 on EAqS1j1oL2o: before the fit no frame was read; after it the
       timer reads on ~100 % of gameplay frames.
     - The height is then corrected at the bottom of the screen: the damage "%" signs must sit
       on the same row as in our renders (fit_bottom; on Genesis 7 console footage they were
       6 pixels low, which made perception misread damage).
     - Last, the picture is shifted by fractions of a pixel until the timer reads best.
     - Brightness: if the timer's white is clearly below 255 (TV levels from a console capture),
       the picture is brightened so it is 255 again (white_level).
  3. Read the in-game timer on every video frame (src/vid2slp/hud_timer.py, the digit network
     trained on our renders). The timer gives the game frame (8-minute timer assumed).
  4. Games: runs of timer readings that keep going up. A new game starts where the game frame
     falls back (a new game starts at 08:00 again).
  5. Each game -> data/real/<id>/game<k>/:
       video.mp4        the game's video frames (from the countdown, when the video has it)
       timer_crops.npz  the timer area of every frame
       frame_map.npy    game frame of every video frame, -1 if unknown (as for our renders)
       info.json        where it came from and what was read

Limits (real footage, nothing is known for sure yet):
  - The timer reader was trained on clean renders. Stream overlays near the timer (here a
    character icon touches the small digits) may cause misreads; the script counts them.
  - Streams can show a game frame twice or skip one (capture at 60.00 vs the game's ~59.94 fps);
    the frame map rules allow skips but drop readings around a repeated frame.

Run:
    .venv/bin/python scripts/real_prepare.py --id EAqS1j1oL2o                 # one video
    .venv/bin/python scripts/real_prepare.py --list data/real/download.csv    # many (restartable)
"""

import argparse
import glob
import json
import os
import subprocess

import av
import numpy as np
import pandas as pd
import torch
from PIL import Image

from vid2slp import hud_timer

REAL = "data/real"
START_SECONDS = 480          # tournament games use the 8-minute timer
MAX_FRAME = START_SECONDS * 60
COUNTDOWN = 123              # the countdown lasts from game frame -123 to 0


def download(url, out):
    """60 fps video only, 720p if there is one (the game area ends up 480 pixels tall anyway,
    and 720p is about half the size of 1080p), else the best 60 fps up to 1080p.
    Slowly: YouTube answered "too many requests" when we checked formats 8 at a time, so wait
    between requests and between videos."""
    subprocess.run([".venv/bin/yt-dlp", "-f", "bv*[fps>=50][height<=720][height>=700]/bv*[fps>=50][height<=1080]",
                    "-o", out, "--write-info-json", "--sleep-requests", "2",
                    "--sleep-interval", "20", "--max-sleep-interval", "60", url], check=True)


def sample_frames(path, n=300):
    """n frames spread evenly over the video (seek, take the next frame), RGB arrays."""
    container = av.open(path)
    stream = container.streams.video[0]
    if stream.duration is not None:
        duration = float(stream.duration * stream.time_base)
    else:
        # some downloads (webm/mkv) only store the whole file's duration, in microseconds
        duration = container.duration / 1e6
    frames = []
    for t in np.linspace(0.02, 0.98, n) * duration:
        container.seek(int(t / stream.time_base), stream=stream)
        frames.append(next(container.decode(stream)).to_ndarray(format="rgb24"))
    container.close()
    return frames


PAD = 64   # black border added around a frame before cropping it (see small_frame)


def small_frame(f, crop, resample=Image.BILINEAR):
    """Cut crop = (x, y, w, h) (video pixels, may be fractions) out of frame f and shrink it to
    584 x 480 (PIL, bilinear). The frame is padded with black first, like shrink() does: a
    fitted crop can reach a little past the picture's edge, which PIL refuses ("box offset
    can't be negative", seen 2026-11-01 on tdZLQSauxlg and -V6FJeu8TlU)."""
    x, y, w, h = crop
    padded = np.pad(f, ((PAD, PAD), (PAD, PAD), (0, 0)))
    return Image.fromarray(padded).resize((584, 480), resample, box=(x + PAD, y + PAD, x + w + PAD, y + h + PAD))


def top_mask(frames, crop, rows=120):
    """Average white-pixel mask of the top `rows` rows (the timer) after cropping each frame to
    crop = (x, y, w, h) in video pixels and shrinking it to 584 x 480."""
    acc = np.zeros((rows, 584))
    for f in frames:
        small = np.asarray(small_frame(f, crop))
        acc += hud_timer.white_mask(small[:rows])
    return acc / len(frames)


def render_top_mask(n_games=6, rows=120):
    """The same average for our own renders (where the timer reader was trained)."""
    acc = []
    for path in sorted(glob.glob("data/m3/renders/*/video.mp4"))[:n_games]:
        masks = []
        container = av.open(path)
        for i, f in enumerate(container.decode(video=0)):
            if i >= 3000:
                break
            if i >= 400 and i % 25 == 0:               # game frames, every 25th
                masks.append(hud_timer.white_mask(f.to_ndarray(format="rgb24")[:rows]))
        container.close()
        acc.append(np.mean(masks, 0))
    return np.mean(acc, 0)


def fit_crop(frames, guess):
    """Scale and shift the guessed crop until the stream's average timer digits overlay our
    renders' (timer area only: the rest of the top of the screen is often covered by overlays).
      1. Coarse, for every width scale a from 0.45 to 1.30 and every height/width ratio from
         0.86 to 1.14: template matching (OpenCV) finds the best shift anywhere in the picture
         (stream layouts may shrink the game a lot or move it off-centre to make room for player
         cameras). The ratio matters: a Slippi widescreen picture's 4:3 game area is NOT 4:3 in
         video pixels (it is 584:480 like our renders, full height), so a guessed 4:3 crop is
         ~9 % too wide for its height. Before 2026-11-01 the ratio was fixed here, which left
         widescreen crops 8 % short vertically and broke perception on them.
      2. Fine: scales a (width) and c (height) separately and sub-pixel shifts around the best.
    Returns (crop (x, y, w, h) in video pixels, correlation 0..1: ~0.8 on a good fit)."""
    import cv2
    frame_height = frames[0].shape[0]
    stream = top_mask(frames, guess, rows=480).astype(np.float32)
    render = render_top_mask().astype(np.float32)
    x0, x1, y0, y1 = 200, 400, 40, 90                 # the timer area in 584 x 480
    template = render[y0:y1, x0:x1]
    r = template - template.mean()

    best = (-1.0, None)
    for a, ratio in [(a, r) for a in np.arange(0.45, 1.31, 0.01) for r in np.arange(0.86, 1.141, 0.02)]:
        c = a * ratio
        # scaled stream mask: pixel (u, v) of it is guessed-crop pixel (a*u, c*v)
        scaled = cv2.resize(stream, (int(round(584 / a)), int(round(480 / c))), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(scaled, template, cv2.TM_CCOEFF_NORMED)
        # near-blank patches give unstable scores (dividing by almost no contrast): only accept
        # places where the stream mask has at least half the template's contrast
        th, tw = template.shape
        mean = cv2.boxFilter(scaled, -1, (tw, th), anchor=(0, 0), normalize=True, borderType=cv2.BORDER_CONSTANT)
        sq = cv2.boxFilter(scaled * scaled, -1, (tw, th), anchor=(0, 0), normalize=True, borderType=cv2.BORDER_CONSTANT)
        std = np.sqrt(np.maximum(sq - mean * mean, 0))[:res.shape[0], :res.shape[1]]
        res = np.where(std >= 0.5 * template.std(), res, -1)
        _, v, _, (u0, v0) = cv2.minMaxLoc(res)
        if v > best[0]:
            best = (v, (a, c, a * (u0 - x0), c * (v0 - y0)))

    def score(a, c, bx, by):
        # output pixel (x, y) takes the guessed-crop pixel (a*x + bx, c*y + by); rows 0..y1 only
        w = np.asarray(Image.fromarray(stream).transform(
            (584, y1), Image.AFFINE, (a, 0, bx, 0, c, by), Image.BILINEAR))[y0:y1, x0:x1]
        w = w - w.mean()
        return float((w * r).sum() / np.sqrt((w * w).sum() * (r * r).sum() + 1e-9))

    if best[1] is None:
        # no place in the picture looked like a timer at any scale (seen 2026-11-01 on videos
        # whose game area is tiny or that are not gameplay at all)
        raise ValueError("could not find the in-game timer anywhere in the video")
    a0, c0, bx0, by0 = best[1]
    best = (score(a0, c0, bx0, by0), best[1])
    for a in np.arange(a0 - 0.015, a0 + 0.016, 0.0025):
        for c in np.arange(c0 - 0.015, c0 + 0.016, 0.0025):
            for bx in np.arange(bx0 - 1.5, bx0 + 1.6, 0.25):
                for by in np.arange(by0 - 1.5, by0 + 1.6, 0.25):
                    v = score(a, c, bx, by)
                    if v > best[0]:
                        best = (v, (a, c, bx, by))
    a, c, bx, by = best[1]

    gx, gy, gw, gh = guess
    # back to video pixels: guessed-crop pixel p is video pixel g0 + p * (guess size / 584 or 480)
    crop = (gx + bx * gw / 584, gy + by * gh / 480, a * gw, c * gh)
    # 3. The timer is a short strip at the top, so it pins the height only loosely (on our
    #    widescreen renders steps 1-2 gave 1043 of the true 1056 rows, 1.2 % short). Most streams
    #    show the game at the full picture height: when the fit already reaches within 3 % of
    #    both the top and bottom edge, take exactly the full height (keeping x and width, which
    #    the timer pins well). The height is then checked at the bottom of the screen by
    #    fit_bottom() (the damage "%" signs).
    H = frame_height
    x, y, w, h = crop
    if abs(y) <= 0.03 * H and abs(y + h - H) <= 0.03 * H:
        crop = (x, 0, w, H)
    return tuple(int(round(v)) for v in crop), round(best[0], 3)


PERCENT_ROW = 428.0   # centre row of the damage "%" signs in our 584 x 480 renders (all ports)
TIMER_ROW = 66.0      # centre row of the timer box (hud_timer.TIMER_BOX)


def percent_template():
    """The damage "%" sign (grey, 34 x 32) cut from one of our renders, cached in data/m3/."""
    cache = "data/m3/percent_template.npy"
    if not os.path.exists(cache):
        container = av.open(sorted(glob.glob("data/m3/renders/*/video.mp4"))[0])
        for i, f in enumerate(container.decode(video=0)):
            if i == 200:                                # in-game, port 1 at 0 %
                grey = np.asarray(Image.fromarray(f.to_ndarray(format="rgb24")).convert("L"))
                break
        container.close()
        np.save(cache, grey[412:444, 118:152].astype(np.float32))
    return np.load(cache)


def percent_row(frames, crop):
    """Median centre row of the damage "%" signs in the sampled frames, after cropping to
    `crop` and shrinking to 584 x 480 (template matching, sub-pixel). Returns (row, matches)."""
    import cv2
    tpl = percent_template()
    rows = []
    for f in frames:
        small = np.asarray(small_frame(f, crop).convert("L"))
        band = small[380:475].astype(np.float32)
        m = cv2.matchTemplate(band, tpl, cv2.TM_CCOEFF_NORMED)
        _, best, _, (u, v) = cv2.minMaxLoc(m)
        if best < 0.6:                                  # no clear "%" (menus, overlays, deaths)
            continue
        dv = 0.0
        if 0 < v < m.shape[0] - 1:                      # parabola through the peak: sub-pixel
            a_, b_, c_ = m[v - 1, u], m[v, u], m[v + 1, u]
            if a_ - 2 * b_ + c_ < 0:
                dv = 0.5 * (a_ - c_) / (a_ - 2 * b_ + c_)
        rows.append(380 + v + dv + tpl.shape[0] / 2)
    return (float(np.median(rows)) if rows else None), len(rows)


def fit_bottom(frames, crop):
    """Stretch the crop's height (keeping the timer where it is) until the damage "%" signs sit
    on the same row as in our renders.
    Why: the crop is fitted on the timer, a short strip at the top, which pins the height only
    loosely. On Genesis 7 console footage (nPEj_2fzpAI) the "%" ended up 6 pixels too low, and
    perception misread the damage by +7.8 on average (hand-read check, 45 values); with the
    height corrected to put it on row 428: mean error 3.0, and the port-3 stock count stopped
    flickering (2026-11-01). (A row-brightness check of the bottom HUD tried earlier did not
    work; matching the "%" sign itself does.)
    Returns (crop, measured row before the change, or None)."""
    first = None
    for _ in range(3):                                  # repeat: the measurement moves with the crop
        row, n = percent_row(frames, crop)
        if row is None or n < 10:
            print(f"  damage % found on only {n} sampled frames: height left as fitted", flush=True)
            return crop, first
        first = row if first is None else first
        k = (PERCENT_ROW - TIMER_ROW) / (row - TIMER_ROW)
        if abs(k - 1) < 0.001 or abs(k - 1) > 0.08:    # done, or too far off to trust
            break
        x, y, w, h = crop
        crop = (x, y + TIMER_ROW * h / 480 * (1 - 1 / k), w, h / k)
    print(f"  damage % on row {first:.1f} (renders: {PERCENT_ROW:.0f}) -> height {crop[3]:.1f}", flush=True)
    return tuple(int(round(v)) for v in crop), round(first, 2)


def refine_crop_by_reading(frames, crop, net, device):
    """Final touch to the crop: shift it by fractions of a pixel until the timer reads best.
    Why: the digit reader looks at fixed digit positions and is very sensitive. On Genesis 7
    (nPEj_2fzpAI) only 35 % of in-game frames read validly with the fitted crop, but 100 % with
    the picture moved by one pixel left and up (2026-11-01). The damage display is read the
    same fixed way, so perception gains from the same alignment.
    Score of a shift = share of the sampled frames whose reading is a valid timer value.
    Shifts are in pixels of the 584 x 480 picture: whole pixels -2..2 first, then half pixels
    around the best. Returns (crop, share of valid readings)."""
    lookup = hud_timer.timer_lookup(MAX_FRAME, START_SECONDS)
    x, y, w, h = crop
    sx, sy = w / 584, h / 480                          # video pixels per output pixel

    def score(dx, dy):
        x0, y0 = x + dx * sx, y + dy * sy
        crops = np.stack([hud_timer.timer_crop(np.asarray(small_frame(f, (x0, y0, w, h)))).copy() for f in frames])
        reads = hud_timer.read_timers_net(crops, net, device, min_prob=0.5)
        return float(np.mean([r is not None and r in lookup for r in reads]))

    best = (score(0, 0), 0.0, 0.0)
    print(f"  timer readable on {best[0]:.0%} of sampled frames before the shift search", flush=True)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            v = score(dx, dy)
            if v > best[0] + 0.005:                     # only move for a real gain
                best = (v, float(dx), float(dy))
    _, bx, by = best
    for dy in (by - 0.5, by, by + 0.5):
        for dx in (bx - 0.5, bx, bx + 0.5):
            if (dx, dy) != (bx, by):
                v = score(dx, dy)
                if v > best[0] + 0.005:
                    best = (v, dx, dy)
    v, dx, dy = best
    print(f"  best shift {dx:+.1f}, {dy:+.1f} px: timer readable on {v:.0%}", flush=True)
    return (int(round(x + dx * sx)), int(round(y + dy * sy)), w, h), round(v, 3)


def white_level(frames, crop):
    """How white the timer digits get in this video (0-255; our renders: 255).
    Per sampled gameplay frame (one where a damage "%" sign is found) the 99.5th percentile of
    the timer area's brightness, then the median over frames. Why: console captures often keep
    TV levels (white ~235): on Genesis 7 (nPEj_2fzpAI) the timer reached ~235. Melee's damage
    digits darken as damage rises, so greyer digits were read as more damage; scaling the
    brightness so white is 255 again cut the hand-read damage error from 3.0 to 2.2 (bias
    +2.3 -> +1.0), 2026-11-01. Returns None when too few gameplay frames are found."""
    import cv2
    tpl = percent_template()
    whites = []
    for f in frames:
        # box (area) resampling like shrink()'s ffmpeg scale: bilinear blends the digit edges and
        # reads ~4 lower (measured: renders 255 either way; Genesis 235 box vs 231 bilinear)
        small = small_frame(f, crop, Image.BOX)
        grey = np.asarray(small.convert("L"))
        m = cv2.matchTemplate(grey[380:475].astype(np.float32), tpl, cv2.TM_CCOEFF_NORMED)
        if m.max() < 0.6:
            continue
        timer = hud_timer.timer_crop(np.asarray(small)).mean(axis=2)
        whites.append(np.percentile(timer, 99.5))
    return round(float(np.median(whites)), 1) if len(whites) >= 10 else None


def shrink(source, out, crop, gain=1.0):
    """crop (x, y, w, h) of the source video -> 584 x 480, 60 fps, near-lossless H.264 (CRF 12).
    gain > 1: brightness scaled by it (white_level below 255; see white_level)."""
    x, y, w, h = crop
    # pad first, in case the fitted crop reaches a little past the picture's edge
    # crop's exact=1: without it ffmpeg rounds x and y DOWN TO EVEN numbers for yuv420p video,
    # shifting the picture by up to one source pixel. That was enough to break the timer reader
    # on Genesis 7 (nPEj_2fzpAI: 35 % valid readings vs 100 % when aligned), 2026-11-01.
    vf = f"pad=iw+64:ih+64:32:32,crop={w}:{h}:{x + 32}:{y + 32}:exact=1,scale=584:480:flags=area"
    if gain > 1.0:
        level = f"clip(val*{gain:.4f},0,255)"
        vf += f",format=rgb24,lutrgb=r='{level}':g='{level}':b='{level}'"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", source, "-vf", vf, "-an",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "12", "-pix_fmt", "yuv420p", out],
                   check=True)


def all_timer_crops(video):
    """The timer area of every frame of a 584 x 480 video, uint8 [n, h, w, 3]."""
    container = av.open(video)
    crops = [hud_timer.timer_crop(f.to_ndarray(format="rgb24")).copy() for f in container.decode(video=0)]
    container.close()
    return np.stack(crops)


def frame_map_real(game_frame):
    """Game frame of every video frame from the raw timer readings (-1 = none), for real
    footage. Same idea as hud_timer.frame_map_net, with one rule changed:
    our renders never show a game frame twice, but a stream capture does (measured on
    EAqS1j1oL2o: ~4 % of steps repeat a game frame, ~2 % skip one). So between two readings
    at video frames a < b the game frame may stay the same or go up, by at most 2 * (b - a) + 2.
    A reading that breaks this with a neighbour is dropped (both of the pair, repeated until
    none break it); a misread usually breaks it with both neighbours.
    Then gaps are filled only where certain (a run of video frames between two readings that
    went up by exactly one game frame per video frame).
    Returns (int32 array, number of readings dropped)."""
    fmap = game_frame.astype(np.int32).copy()
    before = int((fmap >= 0).sum())
    while True:
        known = np.where(fmap >= 0)[0]
        dv = np.diff(known)
        dg = np.diff(fmap[known])
        broken = (dg < 0) | (dg > 2 * dv + 2)
        if not broken.any():
            break
        fmap[known[:-1][broken]] = -1
        fmap[known[1:][broken]] = -1
    dropped = before - int((fmap >= 0).sum())
    hud_timer.fill_gaps(fmap)
    return fmap, dropped


def find_games(game_frame, min_readings=600, window=30):
    """Split the video into games from the per-frame timer readings (-1 = no reading).
    A new game starts where the timer jumps back up, i.e. the game frame falls. Single
    misreads must not split a game, so the comparison uses medians: the median of the next
    `window` readings must be more than 600 game frames (10 s) below the median of the
    previous `window` readings.
    Returns a list of (first video frame read, last video frame read)."""
    known = np.where(game_frame >= 0)[0]
    g = game_frame[known]
    cuts, j = [], window
    while j < len(known) - window:
        if np.median(g[j:j + window]) < np.median(g[j - window:j]) - 600:
            # the exact first reading of the new game: first one in the window that is low
            low = np.median(g[j:j + window])
            j += int(np.argmax(np.abs(g[j:j + window] - low) < 600))
            cuts.append(j)
            j += window
        else:
            j += 1
    bounds = [0] + cuts + [len(known)]
    games = [(known[a], known[b - 1]) for a, b in zip(bounds[:-1], bounds[1:])]
    # keep games with enough readings (drops stray misreads on menus etc.)
    return [(a, b) for a, b in games if ((game_frame[a:b + 1] >= 0).sum()) >= min_readings]


def frame0_video(game_frame, a, b):
    """Video frame where game frame 0 would be shown, from the first 100 readings of a game
    (median of video frame - game frame: robust to misreads; repeats and skips shift it a
    little, only a few frames over 100 readings)."""
    known = np.where(game_frame[a:b + 1] >= 0)[0][:100] + a
    return int(np.median(known - game_frame[known]))


def write_game(full, frames, out):
    """Copy video frames frames[0]..frames[1] (inclusive) of `full` into out/video.mp4,
    frame-exact (frames are picked by number, not by time), x264 CRF 23 like our renders."""
    a, b = frames
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", full,
                    "-vf", f"select=between(n\\,{a}\\,{b})", "-fps_mode", "passthrough", "-an",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
                    f"{out}/video.mp4"], check=True)


def prepare(video_id, url="", crop_arg="", net=None, device="cpu"):
    """Download (if needed), fit, shrink, read the timer and cut one video into games."""
    d = f"{REAL}/{video_id}"
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(f"{d}/source.mp4"):
        download(url or f"https://youtu.be/{video_id}", f"{d}/source.mp4")
    if not os.path.exists(f"{d}/full_584.mp4"):
        probe = av.open(f"{d}/source.mp4").streams.video[0]
        W, H = probe.width, probe.height
        # first guess: the middle 4:3 part of a wide (16:9) picture, else the whole picture.
        # (Widescreen games and 4:3 games shown in a 16:9 stream both have their HUD there.)
        guess = (round((W - H * 4 / 3) / 2), 0, round(H * 4 / 3), H) if W / H > 1.4 else (0, 0, W, H)
        if crop_arg:
            crop, fit, readable, percent_before = tuple(int(v) for v in crop_arg.split(",")), None, None, None
        else:
            print(f"fitting the game area (first guess {guess}) ...", flush=True)
            frames = sample_frames(f"{d}/source.mp4")
            crop, fit = fit_crop(frames, guess)
            crop, percent_before = fit_bottom(frames, crop)
            crop, readable = refine_crop_by_reading(frames, crop, net, device)
        print(f"game area in the video: x {crop[0]}, y {crop[1]}, {crop[2]} x {crop[3]} (fit {fit})", flush=True)
        # brightness: stretch so the timer's white is 255 like in our renders (only when clearly
        # below; a fully white timer, e.g. a Slippi stream, is left alone)
        white = white_level(sample_frames(f"{d}/source.mp4", n=100), crop)
        gain = 255.0 / white if white is not None and white < 250 else 1.0
        print(f"timer white level {white}: brightness x {gain:.3f}", flush=True)
        json.dump({"crop_xywh": crop, "fit_correlation": fit, "timer_readable": readable,
                   "percent_row_before_fit_bottom": percent_before, "white_level": white,
                   "brightness_gain": round(gain, 4), "guess": guess},
                  open(f"{d}/crop.json", "w"), indent=1)
        print("shrinking to 584 x 480 ...", flush=True)
        shrink(f"{d}/source.mp4", f"{d}/full_584.mp4", crop, gain)

    print("reading the timer on every frame ...", flush=True)
    crops = all_timer_crops(f"{d}/full_584.mp4")
    # min_prob 0.5 instead of 0.99: on this stream the small centisecond digits are often read
    # with less confidence (compression, an overlay icon touches them) but still right; the
    # consistency rule in frame_map_real() removes misreads. Measured on game 3 of
    # EAqS1j1oL2o: 51 % of frames read at 0.99, 98 % at 0.5 (all valid timer values).
    reads = hud_timer.read_timers_net(crops, net, device, min_prob=0.5)
    lookup = hud_timer.timer_lookup(MAX_FRAME, START_SECONDS)
    game_frame = np.array([lookup.get(r, -1) if r else -1 for r in reads], dtype=np.int32)
    n_read = int((game_frame >= 0).sum())
    print(f"{len(crops)} video frames, {n_read} with a timer reading ({n_read / len(crops):.1%})")

    games = find_games(game_frame) if n_read else []
    summary = []
    prev_end = -1
    for k, (a, b) in enumerate(games, start=1):
        # start at the countdown (game frame -123) when the video has it, a little before
        zero = frame0_video(game_frame, a, b)
        start = max(prev_end + 1, zero - COUNTDOWN - 30, 0)
        end = min(b + 120, len(crops) - 1)
        if k < len(games):
            end = min(end, games[k][0] - 1)     # never into the next game
        prev_end = end
        fmap, unused = frame_map_real(game_frame[start:end + 1])
        if not (fmap >= 0).any():
            # a "game" made only of misreads (seen on fEcj0swS3MI, 2026-11-01): nothing to keep
            print(f"  game {k}: no consistent timer readings, skipped", flush=True)
            continue
        out = f"{d}/game{k}"
        os.makedirs(out, exist_ok=True)
        np.savez_compressed(f"{out}/timer_crops.npz", crops=crops[start:end + 1])
        shown_start = START_SECONDS
        np.save(f"{out}/frame_map.npy", fmap)
        write_game(f"{d}/full_584.mp4", (start, end), out)
        labeled = fmap[fmap >= 0]
        info = {
            "source": f"{d}/source.mp4", "crop_xywh": json.load(open(f"{d}/crop.json"))["crop_xywh"],
            "video_frames": [int(start), int(end)],
            "video_seconds": [round(start / 60, 2), round(end / 60, 2)],
            "first_game_frame_read": int(labeled.min()), "last_game_frame_read": int(labeled.max()),
            # does the video show the last 39 countdown frames (players can move from -39)?
            "has_countdown": bool(start <= zero - 39),
            "timer_start_seconds": int(shown_start),
            "video_frames_labeled": int(len(labeled)), "video_frames_total": int(end - start + 1),
            "readings_dropped": int(unused),
            # game frames between the first and last labelled one that the video never shows
            "game_frames_missing": int(labeled.max() - labeled.min() + 1 - len(np.unique(labeled))),
            "never_decreasing": bool((np.diff(labeled) >= 0).all()),
            # steps between neighbouring video frames that repeat / skip a game frame
            "repeated_frames": int((np.diff(fmap)[(fmap[:-1] >= 0) & (fmap[1:] >= 0)] == 0).sum()),
            "skipped_frames": int((np.diff(fmap)[(fmap[:-1] >= 0) & (fmap[1:] >= 0)] >= 2).sum()),
        }
        json.dump(info, open(f"{out}/info.json", "w"), indent=1)
        summary.append({"game": k, **info})
        print(json.dumps({"game": k, **info}))
    json.dump(summary, open(f"{d}/games.json", "w"), indent=1)
    return len(games), n_read / len(crops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="")
    ap.add_argument("--id", default="", help="name of the folder under data/real/ (the YouTube id)")
    ap.add_argument("--list", default="", help="csv with an 'id' column (real_find_videos.py): do them all")
    ap.add_argument("--widescreen", action="store_true", help="(no longer needed: the first guess is automatic)")
    ap.add_argument("--crop", default="", help="x,y,w,h of the game's 4:3 picture in video pixels (skips the fit)")
    args = ap.parse_args()
    # the timer network is tiny: run it on the CPU and leave the GPU free for training (2026-11-01)
    device = "cpu"
    torch.set_num_threads(4)
    net = hud_timer.load_digit_net(device)
    if not args.list:
        prepare(args.id, args.url, args.crop, net, device)
        return
    ids = pd.read_csv(args.list).id.tolist()
    for k, vid in enumerate(ids, start=1):
        if os.path.exists(f"{REAL}/{vid}/games.json"):
            continue                                   # done before (restartable)
        print(f"=== {k}/{len(ids)} {vid}", flush=True)
        try:
            n_games, share = prepare(vid, net=net, device=device)
            print(f"=== {vid}: {n_games} games, timer read on {share:.1%} of frames", flush=True)
        except Exception as e:                         # one bad video must not stop the batch
            print(f"=== {vid}: ERROR {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
