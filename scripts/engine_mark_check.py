"""
engine_mark_check.py - check the engine's second saved state ("mark" / "back", added 2026-11-05).

Plays a replay straight through (the reference), then again with a mark at frame 200: from there
it plays OTHER inputs (random overrides) with ordinary capture/restore in between, goes back to the
mark and plays on. After going back, every state must equal the reference (the random inputs
were sent after the mark, so they are forgotten). Also moves the mark forward and checks again,
and counts the engine processes (they must not pile up).

Run:  .venv/bin/python scripts/engine_mark_check.py [replay.slp]
"""

import subprocess
import sys

import numpy as np

from vid2slp.engine import HEADLESS, Engine

SLP = sys.argv[1] if len(sys.argv) > 1 else "data/real/5njRxIAgS-o/rebuild/v2/game1.slp"


def key(st):
    return (st["match_frame"], st["rng"], tuple((p["action"], p["x"], p["y"], p["percent"]) for p in st["players"]))


def processes():
    """How many engine processes are running with this build (all tests' engines included)."""
    out = subprocess.run(["pgrep", "-fc", HEADLESS], capture_output=True, text=True).stdout.strip()
    return int(out or 0)


def go_to(e, frame):
    while e.state()["match_frame"] < frame:
        e.step()


def mess_about(e, rng):
    """Random inputs for the next 60 frames, with capture/restore in between (as a search does)."""
    f0 = e.state()["match_frame"] + 1
    rows = [(f, p, *rng.uniform(-1, 1, 4), 0.0, int(rng.integers(0, 4096)), 0, 0, 0, 0)
            for f in range(f0, f0 + 60) for p in (0, 1)]
    e.override_many(rows)
    for _ in range(5):
        e.capture()
        e.steps(8)
        e.restore()
        e.steps(12)


def main():
    rng = np.random.default_rng(0)
    with Engine(SLP) as e:
        go_to(e, 200)
        ref = {st["match_frame"]: key(st) for st in e.steps(400)}
    before = processes()
    ok = True
    with Engine(SLP) as e:
        go_to(e, 200)
        e.mark()
        for round_ in range(3):
            mess_about(e, rng)
            st = e.back()
            same = key(st) == ref.get(st["match_frame"], key(st)) and st["match_frame"] == 200
            states = e.steps(150)
            same = same and all(key(s) == ref[s["match_frame"]] for s in states)
            print(f"back to the mark, round {round_ + 1}: frame {st['match_frame']}, next 150 frames same as "
                  f"straight play: {same}; engine processes now {processes() - before}")
            ok = ok and same
        # move the mark to frame 350 (we are there now), mess about, come back
        e.mark()
        mess_about(e, rng)
        st = e.back()
        states = e.steps(200)
        same = st["match_frame"] == 350 and all(key(s) == ref[s["match_frame"]] for s in states)
        print(f"mark moved to 350: back gives frame {st['match_frame']}, next 200 frames same: {same}; "
              f"engine processes now {processes() - before}")
        ok = ok and same
    print("engine processes left after closing:", processes() - before)
    print("ALL OK" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
