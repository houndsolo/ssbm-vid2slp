"""
m3_timer_digits.py - train the timer digit network (src/vid2slp/hud_timer.py, DigitNet).

Training examples come from rendered games (data/m3/renders/<sha>/timer_crops.npz) at video
frames whose game frame is CERTAIN from the first (template) reader: either read directly,
or filled in between two readings with no skipped frame (hud_timer.fill_gaps). The filled
frames are exactly the hard cases (bright background, a character over the timer), so the
network learns those too. For each certain frame the 6 digits follow from the formula.

Checks printed at the end:
  1. digit and whole-timer accuracy on HELD-OUT games (every 5th game, by sha256 order);
  2. the same, only on held-out frames the template reader could NOT read (the hard ones);
  3. reading the 1,737 frames of the old verified clips (data/m3/timer_calibration).

Run:
    .venv/bin/python scripts/m3_timer_digits.py
Writes src/vid2slp/timer_digitnet.pt (small, kept in git).
"""

import glob
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

from vid2slp import hud_timer

sys.path.insert(0, os.path.dirname(__file__))
import m3_timer_templates  # noqa: E402  (to rebuild the verified calibration frames)

PER_GAME = 150    # frames sampled per game (x ~400 games; more would not fit in memory)
SEED = 0


def certain_examples(sha, rng):
    """(digit boxes [n, 6, 3, 24, 24], digits [n, 6], template_reader_failed [n]) for one game."""
    d = f"data/m3/renders/{sha}"
    info = json.load(open(f"{d}/info.json"))
    crops = np.load(f"{d}/timer_crops.npz")["crops"]
    fmap = np.load(f"{d}/frame_map.npy")
    if not info["strictly_increasing"]:
        return None
    templates = hud_timer.load_templates()
    idx = np.where(fmap >= 1)[0]
    if len(idx) == 0:      # timer never read (should not happen since the start-value fix)
        print(f"{sha[:12]}: no labelled frames, skipped")
        return None
    idx = np.array(sorted(rng.sample(list(idx), min(PER_GAME, len(idx)))))
    digits = np.array([[int(c) for c in hud_timer.frame_to_timer(int(fmap[i]), info["timer_start_seconds"])]
                       for i in idx])
    failed = np.array([hud_timer.read_timer(crops[i], templates)[0] is None for i in idx])
    return hud_timer.digit_boxes(crops[idx]), digits, failed


def main():
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    shas = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob("data/m3/renders/*/info.json"))
    held = set(shas[::5])
    data = {"train": [], "held": []}
    for sha in shas:
        ex = certain_examples(sha, rng)
        if ex is not None:
            data["held" if sha in held else "train"].append(ex)
    cat = {k: [np.concatenate([e[i] for e in v]) for i in range(3)] for k, v in data.items()}
    print(f"games: {len(shas) - len(held)} train, {len(held)} held out; "
          f"frames: {len(cat['train'][0])} train, {len(cat['held'][0])} held out "
          f"({cat['held'][2].sum()} of them unreadable by the template reader)")

    device = "cuda:0"
    net = hud_timer.make_digit_net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.from_numpy(cat["train"][0]).flatten(0, 1)        # every digit box is one example
    y = torch.from_numpy(cat["train"][1]).flatten()
    for epoch in range(4):
        perm = torch.randperm(len(x))
        for a in range(0, len(x), 512):
            b = perm[a:a + 512]
            loss = F.cross_entropy(net(x[b].to(device)), y[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch {epoch}: last batch loss {loss.item():.4f}")
    net.eval().cpu()
    torch.save(net.state_dict(), hud_timer.DIGIT_NET)

    # ---- checks -------------------------------------------------------------------------
    def evaluate(boxes, digits, name):
        with torch.no_grad():
            prob = torch.cat([net(torch.from_numpy(boxes[a:a + 2048]).flatten(0, 1)).softmax(-1)
                              for a in range(0, len(boxes), 2048)]).view(len(boxes), 6, 10)
        p, d = prob.max(-1)
        d, p = d.numpy(), p.numpy()
        confident = (p >= 0.99).all(-1)
        right = (d == digits).all(-1)
        print(f"{name}: {len(boxes)} frames | digits right {np.mean(d == digits):.5f} | "
              f"whole timer right {right.mean():.5f} | confident {confident.mean():.4f} | "
              f"confident but WRONG {np.sum(confident & ~right)}")

    hb, hd, hf = cat["held"]
    evaluate(hb, hd, "held-out games, all certain frames")
    evaluate(hb[hf], hd[hf], "held-out games, frames the template reader could not read")
    boxes, digits = [], []
    for clip in m3_timer_templates.CLIPS:
        ours, known, start_seconds = m3_timer_templates.known_frames(clip, "data/m3/timer_calibration")
        ks = sorted(known)
        boxes.append(hud_timer.digit_boxes(np.stack([hud_timer.timer_crop(ours[k]) for k in ks])))
        digits.append(np.array([[int(c) for c in hud_timer.frame_to_timer(known[k], start_seconds)] for k in ks]))
    evaluate(np.concatenate(boxes), np.concatenate(digits), "old verified clips")
    print(f"wrote {hud_timer.DIGIT_NET}")


if __name__ == "__main__":
    main()
