"""
m3_compare_video.py - a side-by-side video: game footage, the REAL controller inputs, and the
inputs PREDICTED by the milestone-2 model, plus what the perception model reads from the video.

Left:  the rendered game footage (Playback Dolphin, data/m3/renders/<sha>/video.mp4).
Right: for each player, two controllers side by side:
         REAL      - the inputs stored in the replay;
         PREDICTED - the inputs the milestone-2 model (feat2_v4) predicts from the TRUE game
                     states (not from video: the perception model does not yet give all the
                     states it needs). The real stick position is drawn as a faint grey dot
                     inside the predicted stick, and a predicted button that differs from the
                     real one gets a red ring.
       and one line from the perception model (percep_v1), which DOES only see the video:
       its action state and damage % next to the truth (green = right, red = wrong).

Both models never trained on this game (it is a VALIDATION game). The video is made at full
speed and a half-speed copy is made too.

Run:
    .venv/bin/python scripts/m3_compare_video.py                       # picks a Fox-Falco game
    .venv/bin/python scripts/m3_compare_video.py --sha <sha256> --start-seconds 60 --seconds 20
Writes data/m3/compare/<sha12>_<start>s.mp4 and ..._half_speed.mp4
"""

import argparse
import json
import os
import subprocess

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from vid2slp import id_data
from vid2slp.id_model import build_from_config, predict_split
from vid2slp.matchups import FOX, FALCO, game_characters
from vid2slp.melee_ids import ACTION_NAMES
from vid2slp.perception_data import Game, to_tensors
from vid2slp.perception_model import Perception

M2_RUN = "data/m2/runs/feat2_v4"
M3_RUN = "data/m3/runs/percep_v1"
OUT = "data/m3/compare"
NAMES = {FOX: "Fox", FALCO: "Falco"}
FONT = "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/noto/NotoSansMono-Bold.ttf"
PANEL_W = 700                      # width of the right-hand panel (footage is 584 x 480)
H = 480


def pick_game():
    """A validation Fox-Falco game that is rendered, 8-minute, and in sync."""
    chars = game_characters()
    for r in chars[(chars.split == "validation") & (chars.char_a == FOX) & (chars.char_b == FALCO)].itertuples():
        p = f"data/m3/renders/{r.sha256}/info.json"
        if os.path.exists(p):
            info = json.load(open(p))
            if info["timer_start_seconds"] == 480 and not info.get("playback_ended_early"):
                return r.sha256
    raise SystemExit("no suitable game found")


def m2_predictions(sha, device):
    """(real tgt, real btn, predicted tgt, predicted btn) for every game frame >= 0 of one game."""
    ck = torch.load(f"{M2_RUN}/best.pt", map_location=device, weights_only=False)
    cfg = ck["config"]
    feature_set = "v2" if cfg["data"].endswith("v2") else "v1"
    stage = pd.read_csv("data/index/slp_index.csv").set_index("sha256").loc[sha, "stage_id"]
    num, cat, tgt, btn = id_data.game_arrays(f"data/parsed/{sha}.parquet", stage, feature_set)
    model = build_from_config(cfg, num.shape[-1]).to(device)
    model.load_state_dict(ck["model"])
    split = {"num": num, "cat": cat, "offsets": np.array([0, len(num)])}
    ptgt, pbtn = predict_split(model, split, device, window=cfg["window"])
    return tgt, btn, ptgt, pbtn


def perception_predictions(sha, first, count, device):
    """percep_v1 action + percent for video frames first .. first+count-1 (both players)."""
    ck = torch.load(f"{M3_RUN}/best.pt", map_location=device, weights_only=False)
    cfg = ck["config"]
    size = (cfg["width"], cfg["height"])
    model = Perception(size=size, max_len=cfg["clip"], pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    game = Game(sha)
    clip = cfg["clip"]
    action = np.zeros((count, 2), np.int64)
    percent = np.zeros((count, 2), np.float32)
    with torch.no_grad():
        # consecutive clips of 32 frames covering the whole segment
        for s in range(first, first + count, clip):
            s = min(s, len(game) - clip)
            b = to_tensors(*game.clip(s, clip, size))
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(b["video"][None].to(device))
            a = out["action"][0].argmax(-1).cpu().numpy()
            p = out["reg"][0, :, :, 2].float().cpu().numpy() * 100
            lo, hi = max(s, first), min(s + clip, first + count)
            action[lo - first:hi - first] = a[lo - s:hi - s]
            percent[lo - first:hi - first] = p[lo - s:hi - s]
    return action, percent


# ---- drawing ------------------------------------------------------------------------------
def draw_stick(d, cx, cy, r, value_xy, color, ghost_xy=None):
    """Stick gate (octagon) with the stick position as a dot; values are -1..1."""
    pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.arange(8) * np.pi / 4 + np.pi / 8]
    d.polygon(pts, outline=(150, 150, 150), width=2)
    if ghost_xy is not None:
        gx, gy = cx + ghost_xy[0] * r, cy - ghost_xy[1] * r
        d.ellipse([gx - 6, gy - 6, gx + 6, gy + 6], fill=(110, 110, 110))
    x, y = cx + value_xy[0] * r, cy - value_xy[1] * r
    d.line([cx, cy, x, y], fill=color, width=3)
    d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=color)


def draw_button(d, cx, cy, r, label, pressed, color, font, wrong=False):
    fill = color if pressed else None
    outline = (230, 60, 60) if wrong else color
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=4 if wrong else 2)
    d.text((cx, cy), label, fill=(0, 0, 0) if pressed else color, font=font, anchor="mm")


def draw_controller(d, x0, y0, tgt, btn, font, title, ref_tgt=None, ref_btn=None):
    """One controller. tgt = [stick x, stick y, c x, c y, trigger] in class units, btn = bitmask."""
    d.text((x0 + 150, y0), title, fill=(220, 220, 220), font=font, anchor="mt")
    stick = ((tgt[0] - 80) / 80, (tgt[1] - 80) / 80)
    cst = ((tgt[2] - 80) / 80, (tgt[3] - 80) / 80)
    ghost = None if ref_tgt is None else ((ref_tgt[0] - 80) / 80, (ref_tgt[1] - 80) / 80)
    cghost = None if ref_tgt is None else ((ref_tgt[2] - 80) / 80, (ref_tgt[3] - 80) / 80)
    draw_stick(d, x0 + 55, y0 + 85, 42, stick, (240, 240, 240), ghost)
    draw_stick(d, x0 + 150, y0 + 105, 28, cst, (250, 210, 40), cghost)

    def pressed(mask, name):
        return bool((int(mask) >> id_data.BUTTONS.index(name)) & 1)

    def wrong(name):
        return ref_btn is not None and pressed(btn, name) != pressed(ref_btn, name)

    draw_button(d, x0 + 245, y0 + 90, 20, "A", pressed(btn, "A"), (60, 200, 120), font, wrong("A"))
    draw_button(d, x0 + 210, y0 + 118, 13, "B", pressed(btn, "B"), (230, 80, 80), font, wrong("B"))
    draw_button(d, x0 + 280, y0 + 62, 14, "X/Y", pressed(btn, "JUMP"), (200, 200, 200), font, wrong("JUMP"))
    draw_button(d, x0 + 245, y0 + 48, 13, "Z", pressed(btn, "Z"), (150, 100, 230), font, wrong("Z"))
    # shield: digital L/R click as a button, analog trigger as a bar
    draw_button(d, x0 + 205, y0 + 48, 13, "L/R", pressed(btn, "SHIELD"), (140, 140, 140), font, wrong("SHIELD"))
    trig = tgt[4] / 140
    d.rectangle([x0 + 10, y0 + 140, x0 + 110, y0 + 150], outline=(140, 140, 140))
    d.rectangle([x0 + 10, y0 + 140, x0 + 10 + 100 * trig, y0 + 150], fill=(140, 140, 140))
    d.text((x0 + 115, y0 + 145), "trigger", fill=(140, 140, 140), font=font, anchor="lm")


def action_name(a):
    return ACTION_NAMES.get(int(a), str(a)) if a < 341 else f"special {a}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", default=None)
    ap.add_argument("--start-seconds", type=float, default=60.0, help="where in the game to start")
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    sha = args.sha or pick_game()
    os.makedirs(OUT, exist_ok=True)

    game = Game(sha)
    fmap = game.labels["game_frame"]
    first = int(np.searchsorted(fmap, int(args.start_seconds * 60)))   # first video frame at that game time
    count = min(int(args.seconds * 60), len(game) - first)
    tgt, btn, ptgt, pbtn = m2_predictions(sha, device)
    pact, ppct = perception_predictions(sha, first, count, device)
    chars = [int(game.labels["character"][first, s]) for s in (0, 1)]
    print(f"game {sha[:12]}: {NAMES.get(chars[0])} vs {NAMES.get(chars[1])}, video frames {first}..{first + count - 1}")

    font = ImageFont.truetype(FONT, 13)
    font_big = ImageFont.truetype(FONT_BOLD, 16)
    name = f"{sha[:12]}_{int(args.start_seconds)}s"
    path = f"{OUT}/{name}.mp4"
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{584 + PANEL_W}x{H}", "-r", "60", "-i", "-", "-c:v", "libx264",
                           "-crf", "20", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    agree = {"stick": 0, "buttons": 0, "frames": 0}
    for k in range(0, count, 32):
        n = min(32, count - k)
        video, _ = game.clip(first + k, n, (584, 480))
        for j in range(n):
            v = first + k + j
            g = int(fmap[v])
            canvas = Image.new("RGB", (584 + PANEL_W, H), (24, 24, 28))
            canvas.paste(Image.fromarray(video[j]), (0, 0))
            d = ImageDraw.Draw(canvas)
            if g < 0:   # frame without a label (rare: timer unreadable); show footage only
                d.text((600, 20), "no label for this frame", fill=(200, 200, 200), font=font_big)
                ff.stdin.write(canvas.tobytes())
                continue
            for s in (0, 1):
                y0 = 8 + s * 236
                port_name = f"P{s + 1}  {NAMES.get(chars[s], chars[s])}"
                d.text((600, y0), port_name, fill=(255, 255, 255), font=font_big)
                draw_controller(d, 595, y0 + 22, tgt[g, s], btn[g, s], font, "REAL inputs")
                draw_controller(d, 595 + 345, y0 + 22, ptgt[g, s], pbtn[g, s], font,
                                "PREDICTED  (grey dot = real)", tgt[g, s], btn[g, s])
                # what the perception model reads from the VIDEO
                true_a, pred_a = int(game.labels["action"][v, s]), int(pact[v - first, s])
                true_p, pred_p = float(game.labels["percent"][v, s]), float(ppct[v - first, s])
                ok_a = true_a == pred_a
                ok_p = abs(true_p - pred_p) < 1
                d.text((600, y0 + 200), "video model:", fill=(180, 180, 180), font=font)
                d.text((712, y0 + 200), f"action {action_name(pred_a)}", font=font,
                       fill=(90, 220, 120) if ok_a else (240, 90, 90))
                d.text((712, y0 + 216), f"true   {action_name(true_a)}", font=font, fill=(180, 180, 180))
                d.text((1010, y0 + 200), f"damage {pred_p:5.1f}%", font=font,
                       fill=(90, 220, 120) if ok_p else (240, 90, 90))
                d.text((1010, y0 + 216), f"true   {true_p:5.0f}%", font=font, fill=(180, 180, 180))
                agree["stick"] += int(np.all(np.abs(ptgt[g, s, :2].astype(int) - tgt[g, s, :2]) <= 0))
                agree["buttons"] += int(pbtn[g, s] == btn[g, s])
                agree["frames"] += 1
            d.text((8, 8), f"game frame {g}", fill=(255, 255, 255), font=font)
            d.text((8, 24), "red ring = predicted button wrong", fill=(255, 255, 255), font=font)
            ff.stdin.write(canvas.tobytes())
    ff.stdin.close()
    ff.wait()
    half = f"{OUT}/{name}_half_speed.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-vf", "setpts=2*PTS", "-r", "60",
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", half], check=True)
    f = max(agree["frames"], 1)
    print(f"in this clip: main stick exactly right {agree['stick'] / f:.3f}, all buttons right "
          f"{agree['buttons'] / f:.3f} (player-frames: {agree['frames']})")
    print(f"wrote {path} and {half}")


if __name__ == "__main__":
    main()
