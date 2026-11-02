"""
real_compare_video.py - side-by-side video: the REAL footage of a game next to our rebuilt
replay of it (scripts/real_rebuild.py) played in Slippi Playback Dolphin.

Left:  the game as cut from the downloaded video (data/real/<id>/game<k>/video.mp4, 584 x 480).
Right: the rebuilt .slp (data/real/<id>/rebuild/<tag>/game<k>.slp) rendered by Slippi Playback
       Dolphin (resync off, exactly as anyone's Slippi would play it).
The two are lined up by GAME FRAME, read from the in-game timer on both sides (the same timer
reader as everywhere else, src/vid2slp/hud_timer.py), so a stream's repeated or skipped frames do
not make them drift apart. The left side keeps the video's own pace; the right side shows the
rendered frame of the same game frame.

Run (after real_rebuild.py):
    .venv/bin/python scripts/real_compare_video.py --id 5njRxIAgS-o --tag v0 --games 1,2,3
Writes data/real/<id>/rebuild/<tag>/compare_game<k>.mp4 (whole game) and
compare_game<k>_60s.mp4 (its first minute, small enough to send around).
"""

import argparse
import os
import shutil
import subprocess
import sys

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3_render import timer_crops_from_video              # noqa: E402  (scripts/m3_render.py)
from vid2slp import hud_timer                              # noqa: E402
from vid2slp.playback import regenerate                    # noqa: E402

REAL = "data/real"
WORK = "data/m3/work"
FONT = "/usr/share/fonts/truetype/noto/NotoSansMono-Bold.ttf"
W, H, BAR = 584, 480, 34


def filled(fmap):
    """Game frame of every video frame, with the frames the timer could not tell (-1, e.g. the
    countdown or a covered timer) counted on from the nearest known frame (one game frame per
    video frame): backwards before the first known frame, forwards after a known one."""
    g = fmap.astype(np.int64).copy()
    known = np.where(g >= 0)[0]
    if len(known) == 0:
        raise SystemExit("no timer readings")
    first = known[0]
    g[:first] = g[first] - (first - np.arange(first))
    last = first
    for i in range(first + 1, len(g)):
        if fmap[i] >= 0:
            last = i
        else:
            g[i] = g[last] + (i - last)
    return g


def render(slp, out_avi, work):
    """Play the .slp in Slippi Playback Dolphin and keep its lossless frame dump."""
    shutil.rmtree(work, ignore_errors=True)
    regenerate(slp, work, timeout=3600, stall_seconds=60, gpu=0,
               dump_video={"codec": "ffv1", "fmt": "avi", "kbps": 25000})
    shutil.move(os.path.join(work, "user", "Dump", "Frames", "framedump0.avi"), out_avi)
    shutil.rmtree(work, ignore_errors=True)


def label(img, text, font):
    """A black bar with white text above a 584 x 480 picture."""
    out = Image.new("RGB", (W, H + BAR), "black")
    out.paste(img, (0, BAR))
    ImageDraw.Draw(out).text((8, 5), text, fill="white", font=font)
    return out


def compose(real_video, real_fmap, render_avi, out_mp4, left_text, right_text, max_frames=None):
    """Write the side-by-side mp4: one output frame per real video frame."""
    font = ImageFont.truetype(FONT, 18)
    small = ImageFont.truetype(FONT, 16)
    real_g = filled(real_fmap)
    crops = timer_crops_from_video(render_avi)
    rmap, _, _ = hud_timer.frame_map_net(crops, int(real_g.max()) + 200, 480, hud_timer.load_digit_net())
    render_g = filled(rmap)

    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{2 * W}x{H + 2 * BAR}", "-r", "60", "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", out_mp4],
                           stdin=subprocess.PIPE)
    rend = av.open(render_avi).decode(video=0)
    r_i, r_img = -1, None
    blank = Image.new("RGB", (W, H), (40, 40, 40))
    for i, frame in enumerate(av.open(real_video).decode(video=0)):
        if max_frames and i >= max_frames:
            break
        g = int(real_g[i])
        # move the render forward to the first of its frames showing game frame g (or the last one)
        while r_i + 1 < len(render_g) and (r_img is None or render_g[r_i] < g):
            r_img = next(rend).to_image()
            r_i += 1
        right = r_img if r_img is not None and render_g[r_i] >= g - 1 else blank
        canvas = Image.new("RGB", (2 * W, H + 2 * BAR), "black")
        canvas.paste(label(frame.to_image(), left_text, font), (0, 0))
        canvas.paste(label(right, right_text, font), (W, 0))
        sec = max(g, 0) / 60
        ImageDraw.Draw(canvas).text(
            (8, H + BAR + 7), f"game frame {g:6d}   ({int(sec // 60)}:{sec % 60:05.2f} into the game)", fill="white", font=small)
        enc.stdin.write(canvas.tobytes())
    enc.stdin.close()
    enc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--games", default="1")
    ap.add_argument("--left", default="Real video (YouTube)")
    args = ap.parse_args()
    d = f"{REAL}/{args.id}/rebuild/{args.tag}"
    for k in [int(g) for g in args.games.split(",")]:
        avi = f"{d}/game{k}_render.avi"
        if not os.path.exists(avi):
            render(f"{d}/game{k}.slp", avi, f"{WORK}/rb_{args.id}_{args.tag}_{k}")
        real = f"{REAL}/{args.id}/game{k}"
        fmap = np.load(f"{real}/frame_map.npy")
        right = "Rebuilt .slp in Slippi Playback"
        compose(f"{real}/video.mp4", fmap, avi, f"{d}/compare_game{k}.mp4", args.left, right)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{d}/compare_game{k}.mp4", "-t", "60", "-c", "copy",
                        f"{d}/compare_game{k}_60s.mp4"], check=True)
        os.remove(avi)                             # the lossless dump is large; the mp4s are what we keep
        print(f"game {k}: {d}/compare_game{k}.mp4", flush=True)


if __name__ == "__main__":
    main()
