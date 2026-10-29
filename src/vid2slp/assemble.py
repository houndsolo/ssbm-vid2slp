"""
assemble.py - Stage C helpers: turn a table of per-frame game states (from perception) into
the values written into a .slp, so that Slippi Playback (resync on) shows that game.

What resync does (read from slippi-ssbm-asm Playback/Core/RestoreGameFrame.asm, 2026-10-28):
every frame, for every player, it overwrites x, y, facing and the action state ID with the
replay's stored START-of-frame (pre-frame) values, and the damage % if it differs. There is no
"only if different" check. So:
  - the stored start-of-frame state must be what the game would have at that moment, or the
    replay is pushed into a slightly wrong state every frame;
  - y must be exactly on the floor when a character stands, or it falls/lands for no reason.

Two fixes, both learned from real replays by scripts/m4_pre_frame_rules.py:
  start_of_frame()  - some action changes (jump squat -> jump, landing -> standing, ...) and most
                      turnarounds already show at the START of the frame; use the new value there.
  snap_to_ground()  - put "on the ground" characters exactly on the nearest real floor/platform
                      height of that stage (per 2-unit x bin).
"""

import json

import numpy as np

RULES = "data/m4/pre_frame_rules.json"
GROUND = "data/m4/ground_heights.json"
SNAP_DISTANCE = 4.0     # only snap when a real floor height is this close (world units)


def load_rules():
    rules = json.load(open(RULES))
    early = {tuple(int(v) for v in k.split(",")): p > 0.5 for k, p in rules["action_early"].items()}
    return early, rules["facing_early"] > 0.5


def snap_to_ground(table, stage_id):
    """Copy of table with y moved onto the nearest floor height where airborne == 0."""
    data = json.load(open(GROUND))
    heights = data["heights"].get(str(int(stage_id)), {})
    out = table.copy()
    y = out.y.to_numpy().copy()
    xb = np.floor(out.x.to_numpy() / data["x_bin"]).astype(int)
    grounded = out.airborne.to_numpy() == 0
    snapped = 0
    for i in np.where(grounded)[0]:
        candidates = heights.get(str(xb[i]))
        if not candidates:
            continue
        best = min(candidates, key=lambda h: abs(h - y[i]))
        if abs(best - y[i]) <= SNAP_DISTANCE:
            y[i] = best
            snapped += 1
    out["y"] = y.astype(np.float32)
    return out, snapped


def start_of_frame(prev, now, early_actions, facing_early):
    """Start-of-frame state of a frame, from the end-of-frame states of the previous frame
    (prev) and this frame (now). Both are dicts with action, x, y, facing, percent."""
    action = prev["action"]
    if now["action"] != prev["action"] and early_actions.get((int(prev["action"]), int(now["action"])), False):
        action = now["action"]
    facing = prev["facing"]
    if now["facing"] != prev["facing"] and facing_early:
        facing = now["facing"]
    return {"action": int(action), "x": float(prev["x"]), "y": float(prev["y"]),
            "facing": float(facing), "percent": float(prev["percent"])}
