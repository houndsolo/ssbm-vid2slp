"""
m4_resync_probe.py - which stored game-state fields does Slippi Playback's RESYNC actually use?

Why: in Stage C the .slp's stored states will come from the perception model (noisy), and
resync snaps the replay to them. Before relying on that, find out what resync reads.

Method: take a real replay (true inputs), change ONE stored field of one player for 10 frames
(600..609), play it back with resync ON, and compare the regenerated game with the original:
  control      nothing changed
  pre_x        pre-frame x + 30        post_x        post-frame x + 30
  pre_action   pre-frame action = 14   post_action   post-frame action = 14 (WAIT)
  pre_percent  pre-frame % + 20        post_percent  post-frame % + 20
If the regenerated game moves/changes at those frames, resync uses that field.

Run:
    .venv/bin/python scripts/m4_resync_probe.py --sha <sha256 of a validation game>
"""

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from vid2slp.playback import regenerate
from vid2slp.slp_parse import parse_replay
from vid2slp.slp_write import write_frames

OUT = "data/m4/resync_probe"
FRAMES = range(600, 610)
END = 900


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", required=True)
    args = ap.parse_args()
    path = pd.read_csv("data/index/splits.csv").set_index("sha256").loc[args.sha, "path"]
    orig, _ = parse_replay(path)
    port = int(sorted(orig.port.unique())[0])            # change the lower port's player
    truth = orig.set_index(["frame", "port"])

    def change(field, kind):
        def func(frame, p):
            if frame not in FRAMES or p != port:
                return None
            if field == "action":
                return {"action": 14}
            # pre-frame state of frame f = post-frame state of f-1 in the original
            f = frame - 1 if kind == "pre" else frame
            return {field: float(truth.loc[(f, port), field]) + (30.0 if field == "x" else 20.0)}
        return func

    variants = {"control": (None, None)}
    for field in ["x", "action", "percent"]:
        variants[f"pre_{field}"] = (change(field, "pre"), None)
        variants[f"post_{field}"] = (None, change(field, "post"))

    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)

    def run(name):
        pre, post = variants[name]
        slp = f"{OUT}/{name}.slp"
        write_frames(path, slp, pre=pre, post=post)
        regen, _ = regenerate(slp, f"{OUT}/{name}_run", -123, END, resync=True)
        table, _ = parse_replay(regen)
        return name, table

    with ThreadPoolExecutor(4) as pool:
        results = dict(pool.map(run, variants))

    o = orig[(orig.frame >= 0) & (orig.frame <= END)].set_index(["frame", "port"]).sort_index()
    print(f"game {args.sha[:12]}, changed port {port} on frames {FRAMES.start}-{FRAMES.stop - 1}, resync ON")
    for name, table in results.items():
        r = table[(table.frame >= 0) & (table.frame <= END)].set_index(["frame", "port"]).sort_index()
        common = o.index.intersection(r.index)
        a, b = o.loc[common], r.loc[common]
        dx = (a.x - b.x).abs()
        bad_x = sorted({f for (f, p), v in dx.items() if v > 0.01})
        bad_a = sorted({f for (f, p), v in (a.action != b.action).items() if v})
        bad_p = sorted({f for (f, p), v in ((a.percent - b.percent).abs() > 0.01).items() if v})
        show = lambda fs: f"{len(fs)} frames" + (f" (first {fs[0]}, last {fs[-1]})" if fs else "")
        print(f"  {name:13s} x differs: {show(bad_x):32s} action differs: {show(bad_a):32s} % differs: {show(bad_p)}"
              f"   max |dx| {dx.max():.2f}")


if __name__ == "__main__":
    main()
