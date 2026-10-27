"""
id_metrics.py - how we score predicted controller inputs against the real ones.

Every model and baseline in milestone 2 is scored with `score()` so numbers are comparable.

Why several metrics: many different inputs give the SAME behaviour in Melee (e.g. the
exact stick value inside a walk, or pressing A during landing lag), so "exact match" is
too strict, while "fraction of frames correct" is too easy (most frames are neutral).
We therefore report:

  stick_x / stick_y exact     - exact value (out of 161 possible) matches
  stick_*_within2             - within 2/80 of the real value
  stick_region_joint          - the stick is in the same Melee-relevant zone on both axes:
                                per axis: neutral (|v| < 23/80 = 0.2875, the deadzone),
                                tilt (< 64/80 = 0.8), or smash (>= 0.8), with direction
  cstick_dir                  - same C-stick direction (neutral / left / right / up / down)
  trigger_cat                 - same trigger category (none / light-shield analog / full)
  <button>_held_f1            - F1 of "button is held" per frame (A, B, Z, JUMP=X|Y, SHIELD=L|R,
                                and the JUMP_EDGE / SHIELD_EDGE "new press of the pair" flags)
  <button>_press_f1           - F1 of the exact frame the button goes down
  <button>_press_f1_tol2      - same, but a press counts if it is within 2 frames
  press_f1_tol2_mean          - average press F1 (+-2 frames) over the 5 press events the game
                                reacts to: A, B, Z, JUMP_EDGE, SHIELD_EDGE (headline number)
  frame_all_correct           - stick zones, C-stick direction, trigger category and all
                                button flags ALL match on that frame (strict headline number)

F1 = harmonic mean of precision (how many predicted presses were real) and recall
(how many real presses were found); 1.0 is perfect, 0 means none found.
"""

import numpy as np

from vid2slp.id_data import BUTTONS

DEADZONE = 23   # in 1/80 units
PRESS_EVENTS = ["A", "B", "Z", "JUMP_EDGE", "SHIELD_EDGE"]
SMASH = 64


def axis_zone(v):
    """-80..80 integer stick value -> zone -2..2 (smash left/down ... smash right/up)."""
    a = np.abs(v)
    zone = np.where(a < DEADZONE, 0, np.where(a < SMASH, 1, 2))
    return zone * np.sign(v)


def cstick_dir(x, y):
    """0 neutral, 1 right, 2 left, 3 up, 4 down (by the larger axis)."""
    neutral = np.maximum(np.abs(x), np.abs(y)) < DEADZONE
    horiz = np.abs(x) >= np.abs(y)
    d = np.where(horiz, np.where(x > 0, 1, 2), np.where(y > 0, 3, 4))
    return np.where(neutral, 0, d)


def trigger_cat(t):
    """0..140 -> 0 none, 1 analog (light press), 2 full."""
    return np.where(t == 0, 0, np.where(t >= 140, 2, 1))


def _f1(tp_p, n_pred, tp_t, n_true):
    precision = tp_p / n_pred if n_pred else 0.0
    recall = tp_t / n_true if n_true else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _dilate(mask, k):
    """True wherever mask is True within +-k positions."""
    if k == 0:
        return mask
    c = np.concatenate([[0], np.cumsum(mask)])
    idx = np.arange(len(mask))
    lo, hi = np.clip(idx - k, 0, len(mask)), np.clip(idx + k + 1, 0, len(mask))
    return (c[hi] - c[lo]) > 0


def score(true_tgt, pred_tgt, true_btn, pred_btn, offsets):
    """Score one split.

    true_tgt, pred_tgt: int arrays [N, 2, 5] (stick x, y, c-stick x, y as 0..160; trigger 0..140)
    true_btn, pred_btn: uint8 arrays [N, 2] (bitmask in BUTTONS order)
    offsets: game boundaries [G + 1], used so a press is never counted across two games.
    Returns a dict of metric name -> float.
    """
    # Flatten to one long sequence per port: [2 * N]. Each game/port is a contiguous run.
    tt = np.concatenate([true_tgt[:, 0], true_tgt[:, 1]]).astype(np.int32)
    pt = np.concatenate([pred_tgt[:, 0], pred_tgt[:, 1]]).astype(np.int32)
    tb = np.concatenate([true_btn[:, 0], true_btn[:, 1]])
    pb = np.concatenate([pred_btn[:, 0], pred_btn[:, 1]])
    n = len(true_tgt)
    first_frame = np.zeros(2 * n, bool)
    first_frame[np.concatenate([offsets[:-1], offsets[:-1] + n])] = True

    tx, ty, tcx, tcy = (tt[:, i] - 80 for i in range(4))
    px, py, pcx, pcy = (pt[:, i] - 80 for i in range(4))
    m = {
        "frames": int(2 * n),
        "stick_x_exact": np.mean(tx == px),
        "stick_y_exact": np.mean(ty == py),
        "stick_x_within2": np.mean(np.abs(tx - px) <= 2),
        "stick_y_within2": np.mean(np.abs(ty - py) <= 2),
    }
    joint_t = (axis_zone(tx) == axis_zone(px)) & (axis_zone(ty) == axis_zone(py))
    cdir_t = cstick_dir(tcx, tcy) == cstick_dir(pcx, pcy)
    trig_t = trigger_cat(tt[:, 4]) == trigger_cat(pt[:, 4])
    m["stick_region_joint"] = np.mean(joint_t)
    m["cstick_dir"] = np.mean(cdir_t)
    m["trigger_cat"] = np.mean(trig_t)

    press_f1s = []
    for i, name in enumerate(BUTTONS):
        th, ph = ((tb >> i) & 1).astype(bool), ((pb >> i) & 1).astype(bool)
        m[f"{name}_held_f1"] = _f1((th & ph).sum(), ph.sum(), (th & ph).sum(), th.sum())
        # Press = held now and not held on the previous frame of the same game.
        t_on = th & ~np.concatenate([[False], th[:-1]]) & ~first_frame
        p_on = ph & ~np.concatenate([[False], ph[:-1]]) & ~first_frame
        m[f"{name}_press_f1"] = _f1((t_on & p_on).sum(), p_on.sum(), (t_on & p_on).sum(), t_on.sum())
        f1_tol = _f1((p_on & _dilate(t_on, 2)).sum(), p_on.sum(), (t_on & _dilate(p_on, 2)).sum(), t_on.sum())
        m[f"{name}_press_f1_tol2"] = f1_tol
        if name in PRESS_EVENTS:
            press_f1s.append(f1_tol)
    m["press_f1_tol2_mean"] = float(np.mean(press_f1s))
    m["buttons_all_exact"] = np.mean(tb == pb)
    m["frame_all_correct"] = np.mean(joint_t & cdir_t & trig_t & (tb == pb))
    return {k: float(v) for k, v in m.items()}


HEADLINE = ["stick_x_exact", "stick_region_joint", "cstick_dir", "trigger_cat",
            "press_f1_tol2_mean", "frame_all_correct"]


def format_scores(m, keys=None):
    keys = keys or HEADLINE
    return "  ".join(f"{k} {m[k]:.4f}" for k in keys)
