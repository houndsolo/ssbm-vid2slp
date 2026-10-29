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

Fixes, all learned from real replays (scripts/m4_pre_frame_rules.py, scripts/m4_blast_zones.py):
  start_of_frame()  - some action changes (jump squat -> jump, landing -> standing, ...) and most
                      turnarounds already show at the START of the frame; use the new value there.
  snap_to_ground()  - put "on the ground" characters exactly on the nearest real floor/platform
                      height of that stage (per 2-unit x bin). Tested: slightly worse, not used.
  place_deaths()    - resync does NOT force the stock count: the game only takes a stock when it
                      sees a character cross a blast zone. So put a dying character just past the
                      line right before the death, and keep everyone else safely inside the lines.
"""

import json

import numpy as np

RULES = "data/m4/pre_frame_rules.json"
GROUND = "data/m4/ground_heights.json"
BLAST = "data/m4/blast_zones.json"
SNAP_DISTANCE = 4.0     # only snap when a real floor height is this close (world units)
LAST_DEATH_ACTION = 10  # actions 0..10 are the death animations (0 bottom, 1 left, 2 right, 3-10 top)
MIN_DEATH_RUN = 5       # a death must last at least this many frames (ignore 1-frame mix-ups)
PAST_LINE = 2.0         # put a dying character this far past the blast zone
SAFE_MARGIN = 5.0       # keep living characters at least this far inside (real ones: 0.09% of frames)


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


def place_deaths(table, stage_id):
    """Copy of table (end-of-frame states) in which deaths happen exactly where perception sees them.

    1. Living characters (not in a death action) are kept SAFE_MARGIN inside the left, right and
       bottom blast zones, so a small position error can't kill them. (Not the top line: crossing it
       only kills while being launched, and respawning characters start above it.)
    2. A death = a run of >= MIN_DEATH_RUN frames in a death action, with the stock count (read
       from the HUD, very reliable) dropping within 10 frames of the run. The death frame is the
       earlier of the run's first frame and the stock drop (the death action is sometimes seen one
       frame late; for star KOs the stock only drops at the end of the animation). On the frame
       before the death frame, the character is put PAST_LINE past the blast zone the death action
       points to. The next frame's start-of-frame position is then outside the line, and the game
       itself takes the stock on that frame.
    Returns (new table, number of deaths placed).
    """
    zone = json.load(open(BLAST))[str(int(stage_id))]
    out = table.copy()
    deaths = 0
    for port in out.port.unique():
        rows = np.where(out.port.to_numpy() == port)[0]
        rows = rows[np.argsort(out.frame.to_numpy()[rows])]
        action = out.action.to_numpy()[rows].astype(int)
        stocks = out.stocks.to_numpy()[rows].astype(int)
        x = out.x.to_numpy()[rows].astype(np.float32).copy()
        y = out.y.to_numpy()[rows].astype(np.float32).copy()
        dead = action <= LAST_DEATH_ACTION

        # 1. keep living characters inside the lines
        x[~dead] = np.clip(x[~dead], zone["left"] + SAFE_MARGIN, zone["right"] - SAFE_MARGIN)
        y[~dead] = np.maximum(y[~dead], zone["bottom"] + SAFE_MARGIN)

        # 2. find the death runs and push the character over the line just before each one
        i = 1
        while i < len(rows):
            if dead[i] and not dead[i - 1]:
                end = i
                while end < len(rows) and dead[end]:
                    end += 1
                lo, hi = max(i - 10, 1), min(end + 10, len(rows))
                drops = np.where(stocks[lo:hi] < stocks[lo - 1:hi - 1])[0] + lo   # frames where stocks fall
                if end - i >= MIN_DEATH_RUN and len(drops):
                    f = min(i, drops[0])            # death frame; f - 1 = last frame alive
                    if action[i] == 0:
                        y[f - 1] = zone["bottom"] - PAST_LINE
                    elif action[i] == 1:
                        x[f - 1] = zone["left"] - PAST_LINE
                    elif action[i] == 2:
                        x[f - 1] = zone["right"] + PAST_LINE
                    else:
                        y[f - 1] = zone["top"] + PAST_LINE
                    deaths += 1
                i = end
            else:
                i += 1
        out.loc[out.index[rows], "x"] = x
        out.loc[out.index[rows], "y"] = y
    return out, deaths
