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
    .venv/bin/python scripts/real_prepare.py --url https://youtu.be/EAqS1j1oL2o --id EAqS1j1oL2o --widescreen
"""

import argparse
import glob
import json
import os
import subprocess

import av
import numpy as np
import torch
from PIL import Image

from vid2slp import hud_timer

REAL = "data/real"
START_SECONDS = 480          # tournament games use the 8-minute timer
MAX_FRAME = START_SECONDS * 60
COUNTDOWN = 123              # the countdown lasts from game frame -123 to 0


def download(url, out):
    """60 fps video only, best quality up to 1080p (we shrink it anyway)."""
    subprocess.run([".venv/bin/yt-dlp", "-f", "bv*[fps>=50][height<=1080]", "-o", out,
                    "--write-info-json", url], check=True)


def sample_frames(path, n=300):
    """n frames spread evenly over the video (seek, take the next frame), RGB arrays."""
    container = av.open(path)
    stream = container.streams.video[0]
    duration = float(stream.duration * stream.time_base)
    frames = []
    for t in np.linspace(0.02, 0.98, n) * duration:
        container.seek(int(t / stream.time_base), stream=stream)
        frames.append(next(container.decode(stream)).to_ndarray(format="rgb24"))
    container.close()
    return frames


def top_mask(frames, crop, rows=120):
    """Average white-pixel mask of the top `rows` rows (the timer) after cropping each frame to
    crop = (x, y, w, h) in video pixels and shrinking it to 584 x 480."""
    x, y, w, h = crop
    acc = np.zeros((rows, 584))
    for f in frames:
        small = np.asarray(Image.fromarray(f).resize((584, 480), Image.BILINEAR, box=(x, y, x + w, y + h)))
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
    renders' (correlation of the two average masks, timer area only: the rest of the top of
    the screen is covered by stream overlays). Coarse search, then a finer one around the best.
    Returns (crop (x, y, w, h) in video pixels, correlation 0..1)."""
    stream, render = top_mask(frames, guess), render_top_mask()
    x0, x1, y0, y1 = 200, 400, 40, 90                 # the timer area in 584 x 480
    r = render[y0:y1, x0:x1] - render[y0:y1, x0:x1].mean()

    def score(a, c, bx, by):
        # output pixel (x, y) takes the guessed-crop pixel (a*x + bx, c*y + by)
        w = np.asarray(Image.fromarray(stream.astype(np.float32)).transform(
            (584, stream.shape[0]), Image.AFFINE, (a, 0, bx, 0, c, by), Image.BILINEAR))[y0:y1, x0:x1]
        w = w - w.mean()
        return float((w * r).sum() / np.sqrt((w * w).sum() * (r * r).sum() + 1e-9))

    best = (-1.0, None)
    for a in np.arange(0.80, 1.06, 0.01):
        for bx in np.arange(-40, 41, 1.0):
            for by in np.arange(-20, 21, 1.0):
                v = score(a, a, bx, by)
                if v > best[0]:
                    best = (v, (a, a, bx, by))
    a0, _, bx0, by0 = best[1]
    for a in np.arange(a0 - 0.015, a0 + 0.016, 0.0025):
        for c in np.arange(a0 - 0.015, a0 + 0.016, 0.0025):
            for bx in np.arange(bx0 - 1.5, bx0 + 1.6, 0.25):
                for by in np.arange(by0 - 1.5, by0 + 1.6, 0.25):
                    v = score(a, c, bx, by)
                    if v > best[0]:
                        best = (v, (a, c, bx, by))
    a, c, bx, by = best[1]
    gx, gy, gw, gh = guess
    # back to video pixels: guessed-crop pixel p is video pixel g0 + p * (guess size / 584 or 480)
    crop = (gx + bx * gw / 584, gy + by * gh / 480, a * gw, c * gh)
    return tuple(int(round(v)) for v in crop), round(best[0], 3)


def shrink(source, out, crop):
    """crop (x, y, w, h) of the source video -> 584 x 480, 60 fps, near-lossless H.264 (CRF 12)."""
    x, y, w, h = crop
    # pad first, in case the fitted crop reaches a little past the picture's edge
    vf = f"pad=iw+64:ih+64:32:32,crop={w}:{h}:{x + 32}:{y + 32},scale=584:480:flags=area"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="")
    ap.add_argument("--id", required=True, help="name of the folder under data/real/")
    ap.add_argument("--widescreen", action="store_true", help="16:9 widescreen game: keep the middle 4:3")
    ap.add_argument("--crop", default="", help="x,y,w,h of the game's 4:3 picture in video pixels (skips the fit)")
    args = ap.parse_args()
    d = f"{REAL}/{args.id}"
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(f"{d}/source.mp4"):
        download(args.url, f"{d}/source.mp4")
    if not os.path.exists(f"{d}/full_584.mp4"):
        probe = av.open(f"{d}/source.mp4").streams.video[0]
        W, H = probe.width, probe.height
        # first guess: the whole picture, or its middle 4:3 part for widescreen
        guess = (round((W - H * 4 / 3) / 2), 0, round(H * 4 / 3), H) if args.widescreen else (0, 0, W, H)
        if args.crop:
            crop, fit = tuple(int(v) for v in args.crop.split(",")), None
        else:
            print(f"fitting the game area (first guess {guess}) ...", flush=True)
            crop, fit = fit_crop(sample_frames(f"{d}/source.mp4"), guess)
        print(f"game area in the video: x {crop[0]}, y {crop[1]}, {crop[2]} x {crop[3]} (fit {fit})", flush=True)
        json.dump({"crop_xywh": crop, "fit_correlation": fit, "guess": guess},
                  open(f"{d}/crop.json", "w"), indent=1)
        print("shrinking to 584 x 480 ...", flush=True)
        shrink(f"{d}/source.mp4", f"{d}/full_584.mp4", crop)

    print("reading the timer on every frame ...", flush=True)
    crops = all_timer_crops(f"{d}/full_584.mp4")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = hud_timer.load_digit_net(device)
    # min_prob 0.5 instead of 0.99: on this stream the small centisecond digits are often read
    # with less confidence (compression, an overlay icon touches them) but still right; the
    # consistency rule in frame_map_real() removes misreads. Measured on game 3 of
    # EAqS1j1oL2o: 51 % of frames read at 0.99, 98 % at 0.5 (all valid timer values).
    reads = hud_timer.read_timers_net(crops, net, device, min_prob=0.5)
    lookup = hud_timer.timer_lookup(MAX_FRAME, START_SECONDS)
    game_frame = np.array([lookup.get(r, -1) if r else -1 for r in reads], dtype=np.int32)
    n_read = int((game_frame >= 0).sum())
    print(f"{len(crops)} video frames, {n_read} with a timer reading ({n_read / len(crops):.1%})")

    games = find_games(game_frame)
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
        out = f"{d}/game{k}"
        os.makedirs(out, exist_ok=True)
        np.savez_compressed(f"{out}/timer_crops.npz", crops=crops[start:end + 1])
        fmap, unused = frame_map_real(game_frame[start:end + 1])
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


if __name__ == "__main__":
    main()
