"""
id_data.py - turn parsed games into arrays for the inverse-dynamics model (milestone 2).

"Inverse dynamics" = given what the characters DID (game state over time), work out what
was PRESSED (controller inputs). This file defines exactly which state values the model
sees (features) and which inputs it must predict (targets).

Layout: for one split, all games are stacked frame after frame into big arrays:
    num[N, 2, NUM_FEATURES]  float32  numeric state of both players (port order: lower port first)
    cat[N, 2, CAT_FEATURES]  int16    categorical state (action state id, character, ...)
    tgt[N, 2, 5]             int16    stick x, stick y, c-stick x, c-stick y as -80..80 shifted to 0..160,
                                      and trigger as 0..140 (steps of 1/140)
    btn[N, 2]                uint8    bitmask of logical buttons (see BUTTONS)
    offsets[G + 1]           int64    game g occupies rows offsets[g] : offsets[g+1]
Only frames >= 0 are kept (frames -123..-1 are the countdown where nobody can act).
"""

import os

import numpy as np
import pandas as pd

# ---- targets ---------------------------------------------------------------------------
# Logical buttons. X and Y both jump, and L and R (digital click) both shield / air dodge,
# so each pair is merged into "any of the pair is held" (JUMP, SHIELD).
# BUT merging alone loses information: pressing R while L is already held is a NEW press
# (e.g. wavedash out of shield: hold L, jump, press R to air dodge), yet "any held" stays 1.
# Found 2026-10-27 when true inputs, merged, failed to replay identically. So each pair also
# gets an EDGE flag: "some button of the pair went down on this frame".
# DPAD_UP is the taunt. Rare (0.02% of frames) but used in ~15% of games, and dropping it
# makes the replay diverge (found 2026-10-27 in the same oracle check).
BUTTONS = ["A", "B", "Z", "JUMP", "SHIELD", "JUMP_EDGE", "SHIELD_EDGE", "DPAD_UP"]
_PHYS_BITS = {"A": [8], "B": [9], "Z": [4], "JUMP": [10, 11], "SHIELD": [5, 6],
              "DPAD_UP": [3]}  # .slp physical bit numbers
_EDGE_OF = {"JUMP_EDGE": "JUMP", "SHIELD_EDGE": "SHIELD"}
AXES = ["stick_x", "stick_y", "cstick_x", "cstick_y"]
AXIS_CLASSES = 161     # -80..80
TRIGGER_CLASSES = 141  # 0..140 (value * 140)
# RAW main-stick bytes (-128..127, stored +128 as 0..255). The game derives the processed stick
# from them (stick_from_raw), but UCF 0.84 also reads the raw bytes themselves (e.g. a stick pushed
# PAST the rim on a cardinal counts as exactly 1.0), so the closed loop needs them (2026-10-30).
RAW_CLASSES = 256
RAW_MISSING = -100     # older replays don't store raw stick y: ignored in the loss

# ---- features --------------------------------------------------------------------------
# (column, scale): the model sees value / scale, so most features land roughly in -3..3.
# Two versions exist so older checkpoints stay usable:
#   v1 - the first feature set (data/m2)
#   v2 - adds frame-to-frame VELOCITY CHANGES at a fine scale (data/m2_v2). Why: replay tests
#        showed most desyncs come from the exact analog stick value (e.g. air drift changes
#        velocity by 0.08 x stick for Fox), which is visible in velocity changes of ~0.01.
NUMERIC_V1 = [
    ("x", 100.0), ("y", 100.0), ("dx", 3.0), ("dy", 3.0),
    ("vel_air_x", 3.0), ("vel_ground_x", 3.0), ("vel_y", 3.0),
    ("vel_knockback_x", 3.0), ("vel_knockback_y", 3.0),
    ("percent", 100.0), ("shield", 60.0), ("stocks", 4.0), ("hitlag", 10.0),
    ("action_frame", 30.0), ("jumps", 2.0), ("airborne", 1.0), ("facing", 1.0),
]
NUMERIC_V2 = NUMERIC_V1 + [
    ("dvel_air_x", 0.1), ("dvel_ground_x", 0.1), ("dvel_y", 0.1),
    ("dvel_knockback_x", 0.1), ("dvel_knockback_y", 0.1), ("dx_fine", 0.3), ("dy_fine", 0.3),
]
FEATURE_SETS = {"v1": NUMERIC_V1, "v2": NUMERIC_V2}
NUMERIC = NUMERIC_V1  # default, kept for older code
# (column, vocabulary size): values >= size are clipped to size - 1.
CATEGORICAL = [
    ("action", 512), ("character", 40), ("ground", 256), ("hurtbox_state", 4),
    ("l_cancel", 4), ("last_attack_landed", 256), ("stage", 8),
]
NUM_FEATURES = len(NUMERIC)
CAT_FEATURES = len(CATEGORICAL)

# Legal stage id -> small index for the "stage" feature.
STAGE_INDEX = {2: 0, 3: 1, 8: 2, 28: 3, 31: 4, 32: 5}


def game_arrays(parquet_path, stage_id, feature_set="v1"):
    """Load one parsed game and return (num, cat, tgt, btn) for frames >= 0."""
    full = pd.read_parquet(parquet_path)
    # Button edges need the previous frame, so compute them on the full table first
    # (frame 0's "previous frame" is frame -1, from the countdown).
    phys_full = full.buttons_physical.to_numpy().astype(np.int64).reshape(-1, 2)
    t = full[full.frame >= 0]
    n = len(t) // 2
    # Each frame has exactly two rows (two ports), sorted by frame then port,
    # so a reshape gives [frame, port]. Check that assumption instead of trusting it.
    frames = t.frame.to_numpy().reshape(n, 2)
    ports = t.port.to_numpy().reshape(n, 2)
    assert len(t) == 2 * n and (frames[:, 0] == frames[:, 1]).all() and (ports[:, 0] < ports[:, 1]).all()
    assert (np.diff(frames[:, 0]) == 1).all(), "frames are not contiguous"

    def col(name):
        return np.nan_to_num(t[name].to_numpy(dtype=np.float32)).reshape(n, 2)

    # Movement since the previous frame (first frame gets 0).
    x, y = col("x"), col("y")
    dx = np.vstack([np.zeros((1, 2), np.float32), np.diff(x, axis=0)])
    dy = np.vstack([np.zeros((1, 2), np.float32), np.diff(y, axis=0)])
    extra = {"dx": dx, "dy": dy, "dx_fine": dx, "dy_fine": dy}
    for name in ["vel_air_x", "vel_ground_x", "vel_y", "vel_knockback_x", "vel_knockback_y"]:
        v = col(name)
        extra["d" + name] = np.vstack([np.zeros((1, 2), np.float32), np.diff(v, axis=0)])

    num = np.stack([(extra[c] if c in extra else col(c)) / s for c, s in FEATURE_SETS[feature_set]], axis=-1)
    num = np.clip(num, -50, 50).astype(np.float32)  # guards against rare garbage values

    stage = np.full((n, 2), STAGE_INDEX.get(int(stage_id), 7), np.float32)
    cat = np.stack([np.clip(stage if c == "stage" else col(c), 0, size - 1) for c, size in CATEGORICAL],
                   axis=-1).astype(np.int16)

    tgt = np.stack([np.round(col(a) * 80) + 80 for a in AXES] + [np.round(col("trigger") * 140)], axis=-1)
    tgt = np.clip(tgt, 0, [160, 160, 160, 160, 140]).astype(np.int16)

    first = len(phys_full) - n          # row of frame 0 in the full table
    phys, prev = phys_full[first:], phys_full[first - 1:-1]
    btn = np.zeros((n, 2), np.uint8)
    for i, name in enumerate(BUTTONS):
        if name in _EDGE_OF:
            # any button of the pair that is down now but was up on the previous frame
            bits = _PHYS_BITS[_EDGE_OF[name]]
            flag = np.zeros((n, 2), bool)
            for bit in bits:
                flag |= (((phys >> bit) & 1) & ~((prev >> bit) & 1)).astype(bool)
        else:
            flag = np.zeros((n, 2), bool)
            for bit in _PHYS_BITS[name]:
                flag |= ((phys >> bit) & 1).astype(bool)
        btn |= (flag.astype(np.uint8) << i)
    return num, cat, tgt, btn


def stick_from_raw(raw_x, raw_y):
    """Raw main-stick bytes -> processed stick values as the game computes them, in -80..80
    (divide by 80 for the .slp float). Checked on 426,224 frames of real UCF 0.84 games: exact.
    The raw point is pulled onto the circle of radius 80 if it is outside, truncated toward
    zero, and then any axis below 23 is set to 0 (the dead zone)."""
    rx = np.asarray(raw_x, np.float64)
    ry = np.asarray(raw_y, np.float64)
    m = np.sqrt(rx ** 2 + ry ** 2)
    scale = np.where(m > 80, 80 / np.maximum(m, 1e-9), 1.0)
    px, py = np.trunc(rx * scale), np.trunc(ry * scale)
    px = np.where(np.abs(px) < 23, 0, px)
    py = np.where(np.abs(py) < 23, 0, py)
    return px.astype(np.int64), py.astype(np.int64)


def raw_arrays(parquet_path):
    """Raw main-stick bytes of one parsed game, frames >= 0: int16 [n, 2, 2] (x, y) as 0..255,
    RAW_MISSING where the replay doesn't store them."""
    t = pd.read_parquet(parquet_path, columns=["frame", "port", "raw_stick_x", "raw_stick_y"])
    t = t[t.frame >= 0]
    n = len(t) // 2
    out = []
    for c in ("raw_stick_x", "raw_stick_y"):
        v = t[c].to_numpy(dtype=np.float32).reshape(n, 2)
        out.append(np.where(np.isnan(v), RAW_MISSING, np.nan_to_num(v) + 128))
    return np.stack(out, axis=-1).astype(np.int16)


def load_split(directory, split, mmap=True):
    """Load the arrays saved by scripts/m2_build_dataset.py for one split."""
    mode = "r" if mmap else None
    out = {k: np.load(f"{directory}/{split}_{k}.npy", mmap_mode=mode) for k in ("num", "cat", "tgt", "btn")}
    out["offsets"] = np.load(f"{directory}/{split}_offsets.npy")
    if os.path.exists(f"{directory}/{split}_raw.npy"):      # scripts/m2_add_raw.py
        out["raw"] = np.load(f"{directory}/{split}_raw.npy", mmap_mode=mode)
    out["sha256"] = open(f"{directory}/{split}_games.txt").read().split()
    return out


def arrays_from_states(table, stage_id, feature_set="v2"):
    """(num, cat) for the inverse-dynamics model from a game-state table that has ONLY the
    fields perception can see (Stage C): character, action, x, y, facing, percent, stocks,
    shield, airborne. Fields it cannot see (velocities, hitlag, jumps, ground id, ...) are 0;
    the "visible features" models (m2_train.py --zero-features/--zero-cats) never look at them.

    table: rows (frame, port) for frames 0..T-1, two ports, like data/m3/pred/<run>/<sha>.parquet.
    """
    t = table[table.frame >= 0].sort_values(["frame", "port"])
    n = len(t) // 2

    def col(name):
        if name not in t.columns:
            return np.zeros((n, 2), np.float32)
        return np.nan_to_num(t[name].to_numpy(dtype=np.float32)).reshape(n, 2)

    x, y = col("x"), col("y")
    dx = np.vstack([np.zeros((1, 2), np.float32), np.diff(x, axis=0)])
    dy = np.vstack([np.zeros((1, 2), np.float32), np.diff(y, axis=0)])
    extra = {"dx": dx, "dy": dy, "dx_fine": dx, "dy_fine": dy}
    num = np.stack([(extra[c] if c in extra else col(c)) / s for c, s in FEATURE_SETS[feature_set]], axis=-1)
    num = np.clip(num, -50, 50).astype(np.float32)
    stage = np.full((n, 2), STAGE_INDEX.get(int(stage_id), 7), np.float32)
    cat = np.stack([np.clip(stage if c == "stage" else col(c), 0, size - 1) for c, size in CATEGORICAL],
                   axis=-1).astype(np.int16)
    return num, cat
