"""
slp_parse.py - turn one .slp replay into a simple per-frame table.

The table has one row per (frame, port): what the player pressed that frame
("pre-frame" inputs) and what state their character ended up in ("post-frame" state).
This is the training data for the inverse-dynamics model (state -> inputs) and the
label source for the perception model (video -> state).

Uses peppi-py, a fast Rust .slp parser.

IMPORTANT GOTCHA (found in the audit): frame numbers inside a .slp are not always
unique. With rollback netcode, Slippi can write the same frame number more than once;
the LAST copy is the one that actually happened. `last_copy_rows` handles this.
"""

import numpy as np
import pandas as pd
import peppi_py

# Melee's first frame number in every replay (the countdown before "GO").
FIRST_FRAME = -123


def last_copy_rows(frame_ids):
    """Return row indices that keep only the LAST copy of each frame number, in frame order.

    Example: frame_ids [1, 2, 3, 2, 3, 4] -> rows [0, 3, 4, 5] (frames 1, 2, 3, 4).
    """
    ids = np.asarray(frame_ids)
    # np.unique on the reversed array finds the first occurrence from the end, i.e. the last copy.
    _, first_from_end = np.unique(ids[::-1], return_index=True)
    rows = len(ids) - 1 - first_from_end
    return rows  # np.unique returns ids sorted ascending, so rows are already in frame order


def _arr(value, rows):
    """Convert one peppi (pyarrow) column to numpy and pick `rows`. Missing column -> None."""
    if value is None:
        return None
    return value.to_numpy(zero_copy_only=False)[rows]


def parse_replay(path):
    """Parse a .slp file. Returns (table, info).

    table: pandas DataFrame, one row per (frame, port), sorted by frame then port.
           Columns that the replay's Slippi version does not record are filled with NaN.
    info:  dict with frame-number facts (duplicates removed, gaps found).
    """
    game = peppi_py.read_slippi(path)
    ids = game.frames.id.to_numpy()
    rows = last_copy_rows(ids)
    frames = ids[rows]

    # Ports in game.frames.ports are listed in the same order as game.start.players.
    tables = []
    for player, port_data in zip(game.start.players, game.frames.ports):
        pre = port_data.leader.pre    # inputs for this frame
        post = port_data.leader.post  # resulting character state
        cols = {
            "frame": frames.astype(np.int32),
            "port": np.full(len(rows), int(player.port), np.int8),
            # --- inputs ("effective" = after the game processes the raw controller) ---
            "buttons": _arr(pre.buttons, rows),              # bitmask of logical buttons
            "buttons_physical": _arr(pre.buttons_physical, rows),
            "stick_x": _arr(pre.joystick.x, rows),           # -1..1
            "stick_y": _arr(pre.joystick.y, rows),
            "cstick_x": _arr(pre.cstick.x, rows),
            "cstick_y": _arr(pre.cstick.y, rows),
            "trigger": _arr(pre.triggers, rows),             # 0..1, max of L/R as the game sees it
            "trigger_l_physical": _arr(pre.triggers_physical.l, rows),
            "trigger_r_physical": _arr(pre.triggers_physical.r, rows),
            "raw_stick_x": _arr(pre.raw_analog_x, rows),     # raw controller byte (newer replays)
            "raw_stick_y": _arr(pre.raw_analog_y, rows),     # only recorded since Slippi 3.15
            # --- state after this frame ---
            "character": _arr(post.character, rows),         # internal character id
            "action": _arr(post.state, rows),                # action state id
            "action_frame": _arr(post.state_age, rows),      # frames into the current action
            "x": _arr(post.position.x, rows),
            "y": _arr(post.position.y, rows),
            "facing": _arr(post.direction, rows),            # +1 right, -1 left
            "percent": _arr(post.percent, rows),
            "shield": _arr(post.shield, rows),
            "stocks": _arr(post.stocks, rows),
            "airborne": _arr(post.airborne, rows),
            "ground": _arr(post.ground, rows),               # id of the platform stood on
            "jumps": _arr(post.jumps, rows),                 # jumps remaining
            "hitlag": _arr(post.hitlag, rows),
            "hurtbox_state": _arr(post.hurtbox_state, rows),
            "l_cancel": _arr(post.l_cancel, rows),
            "last_attack_landed": _arr(post.last_attack_landed, rows),
            "last_hit_by": _arr(post.last_hit_by, rows),
            "combo_count": _arr(post.combo_count, rows),
            "misc_as": _arr(post.misc_as, rows),
            "vel_air_x": _arr(post.velocities.self_x_air, rows) if post.velocities else None,
            "vel_ground_x": _arr(post.velocities.self_x_ground, rows) if post.velocities else None,
            "vel_y": _arr(post.velocities.self_y, rows) if post.velocities else None,
            "vel_knockback_x": _arr(post.velocities.knockback_x, rows) if post.velocities else None,
            "vel_knockback_y": _arr(post.velocities.knockback_y, rows) if post.velocities else None,
        }
        # Columns this Slippi version doesn't record become all-NaN float columns.
        cols = {k: (v if v is not None else np.full(len(rows), np.nan, np.float32)) for k, v in cols.items()}
        tables.append(pd.DataFrame(cols))

    table = pd.concat(tables, ignore_index=True).sort_values(["frame", "port"], kind="stable")
    table = table.reset_index(drop=True)

    expected = np.arange(FIRST_FRAME, frames[-1] + 1) if len(frames) else np.array([])
    info = {
        "rows_in_file": int(len(ids)),
        "duplicate_frame_ids": int(len(ids) - len(rows)),
        "first_frame": int(frames[0]) if len(frames) else None,
        "last_frame": int(frames[-1]) if len(frames) else None,
        "missing_frames": int(len(np.setdiff1d(expected, frames))),
    }
    return table, info
