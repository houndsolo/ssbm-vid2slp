"""
m2_resim_probe.py - which input fields does Slippi Playback actually feed into the game?

A .slp stores each frame's inputs twice: "processed" values (stick/c-stick/trigger floats
and a processed button word, i.e. what the game computed) and "physical" values (raw
button bits, raw L/R analog, raw stick bytes). Before we write predicted inputs into a
replay, we need to know which of these the playback actually uses.

Experiment (first 1200 frames of one test game, inputs of port 1 only):
  identity  - unchanged file                     -> must replay identically
  processed - zero the processed fields          -> if playback uses them, the game diverges
  physical  - zero the physical fields           -> if playback uses them, the game diverges
For each, report the first frame where the regenerated state differs from the original.

Run:
    .venv/bin/python scripts/m2_resim_probe.py              # which fields playback reads
    PROBE=bits .venv/bin/python scripts/m2_resim_probe.py   # which button bits matter
"""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from vid2slp.playback import regenerate
from vid2slp.slp_parse import parse_replay
from vid2slp.slp_write import write_inputs

OUT = "data/resim/probe"
END = 1200
PROCESSED = {"stick_x": 0.0, "stick_y": 0.0, "cstick_x": 0.0, "cstick_y": 0.0, "trigger": 0.0, "buttons": 0}
PHYSICAL = {"buttons_physical": 0, "trigger_l_physical": 0.0, "trigger_r_physical": 0.0,
            "raw_stick_x": 0, "raw_stick_y": 0}


def first_divergence(orig, regen):
    """First frame >= 0 where x, y, action or percent of any port differs (None if never)."""
    o = orig.set_index(["frame", "port"])
    r = regen.set_index(["frame", "port"])
    common = r.index.intersection(o.index)
    diff = np.zeros(len(common), bool)
    for col in ["x", "y", "action", "percent"]:
        diff |= o.loc[common, col].to_numpy() != r.loc[common, col].to_numpy()
    frames = np.array([f for f, _ in common])
    bad = frames[diff & (frames >= 0)]
    return (int(bad.min()) if len(bad) else None), int(frames.max())


def main():
    splits = pd.read_csv("data/index/splits.csv")
    game = splits[(splits.split == "test")].sort_values("sha256").iloc[0]
    print("test game:", game.path, game.characters, game.stage)
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    first_port = int(parse_replay(game.path)[0].port.min())
    variants = {
        "identity": None,
        "processed": PROCESSED,
        "physical": PHYSICAL,
    }
    paths = {}
    for name, fields in variants.items():
        path = f"{OUT}/{name}.slp"
        write_inputs(game.path, path, lambda f, p, v=fields: v if v and p == first_port and f >= 0 else None)
        paths[name] = path

    with ThreadPoolExecutor(3) as pool:
        jobs = {name: pool.submit(regenerate, path, f"{OUT}/{name}_run", -123, END) for name, path in paths.items()}
        results = {name: j.result() for name, j in jobs.items()}

    orig, _ = parse_replay(game.path)
    for name, (regen_path, seconds) in results.items():
        regen, _ = parse_replay(regen_path)
        first, last = first_divergence(orig, regen)
        verdict = "identical to original" if first is None else f"diverges at frame {first}"
        print(f"{name:10s}: {verdict}  (regenerated up to frame {last}, {seconds:.0f} s)")


if __name__ == "__main__" and os.environ.get("PROBE") != "bits":
    main()


# ---- second probe (added after the first one showed playback uses PROCESSED fields only) ----
# Which parts of the processed button word matter? Variants applied to port 1 from frame 0:
#   no_dir_bits   - clear bits 16-23 (stick / c-stick direction flags the game derives)
#   no_bit31      - clear bit 31 (the "trigger is pressed" flag)
#   swap_xy       - move X presses to Y and Y to X        (should be equivalent: both jump)
#   swap_lr       - move L presses to R and R to L        (should be equivalent: both shield)
def probe_button_bits():
    splits = pd.read_csv("data/index/splits.csv")
    game = splits[(splits.split == "test")].sort_values("sha256").iloc[0]
    out = OUT + "_bits"
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    orig, _ = parse_replay(game.path)
    port = int(orig.port.min())
    word = {(int(f), int(p)): int(b) for f, p, b in orig[["frame", "port", "buttons"]].itertuples(index=False)}

    def swap(b, i, j):
        bi, bj = (b >> i) & 1, (b >> j) & 1
        return (b & ~((1 << i) | (1 << j))) | (bi << j) | (bj << i)

    edits = {
        "no_dir_bits": lambda b: b & ~(0xFF << 16),
        "no_bit31": lambda b: b & ~(1 << 31),
        "swap_xy": lambda b: swap(b, 10, 11),
        "swap_lr": lambda b: swap(b, 5, 6),
    }
    paths = {}
    for name, edit in edits.items():
        path = f"{out}/{name}.slp"
        write_inputs(game.path, path,
                     lambda f, p, e=edit: {"buttons": e(word[(f, p)])} if p == port and f >= 0 else None)
        paths[name] = path
    with ThreadPoolExecutor(4) as pool:
        jobs = {name: pool.submit(regenerate, path, f"{out}/{name}_run", -123, END) for name, path in paths.items()}
        results = {name: j.result() for name, j in jobs.items()}
    for name, (regen_path, seconds) in results.items():
        regen, _ = parse_replay(regen_path)
        first, last = first_divergence(orig, regen)
        print(f"{name:12s}: {'identical to original' if first is None else f'diverges at frame {first}'}")


if __name__ == "__main__" and os.environ.get("PROBE") == "bits":
    probe_button_bits()
