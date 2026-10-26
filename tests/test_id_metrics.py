"""Checks for the milestone-2 scoring code (src/vid2slp/id_metrics.py)."""

import numpy as np

from vid2slp.id_metrics import axis_zone, cstick_dir, score, trigger_cat


def test_zones():
    v = np.array([-80, -64, -63, -23, -22, 0, 22, 23, 63, 64, 80])
    assert list(axis_zone(v)) == [-2, -2, -1, -1, 0, 0, 0, 1, 1, 2, 2]
    assert list(cstick_dir(np.array([0, 80, -80, 0, 0]), np.array([0, 0, 0, 80, -80]))) == [0, 1, 2, 3, 4]
    assert list(trigger_cat(np.array([0, 43, 139, 140]))) == [0, 1, 1, 2]


def _one_game(n=20):
    tgt = np.full((n, 2, 5), 80, np.int16)
    tgt[..., 4] = 0
    btn = np.zeros((n, 2), np.uint8)
    return tgt, btn, np.array([0, n])


def test_perfect_prediction_scores_one():
    tgt, btn, off = _one_game()
    btn[5:8, 0] = 1  # A held for 3 frames on port 0
    m = score(tgt, tgt.copy(), btn, btn.copy(), off)
    assert m["frame_all_correct"] == 1.0 and m["A_press_f1"] == 1.0 and m["A_held_f1"] == 1.0


def test_press_tolerance():
    tgt, btn, off = _one_game()
    true_b = btn.copy(); true_b[5:8, 0] = 1
    pred_b = btn.copy(); pred_b[7:9, 0] = 1  # pressed 2 frames late
    m = score(tgt, tgt, true_b, pred_b, off)
    assert m["A_press_f1"] == 0.0          # wrong exact frame
    assert m["A_press_f1_tol2"] == 1.0     # but within 2 frames


def test_press_not_counted_across_games():
    # Port 0 holds A at the end of game 1 and the start of game 2: the start of
    # game 2 is not a "press" because it's the first frame of that game.
    tgt = np.full((10, 2, 5), 80, np.int16); tgt[..., 4] = 0
    b = np.zeros((10, 2), np.uint8); b[3:7, 0] = 1
    m = score(tgt, tgt, b, np.zeros_like(b), np.array([0, 5, 10]))
    assert m["A_press_f1"] == 0.0 and m["A_held_f1"] == 0.0
