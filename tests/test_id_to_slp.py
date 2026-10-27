"""Checks for turning predicted targets into .slp input fields (src/vid2slp/id_to_slp.py)."""

import os

import numpy as np
import pytest

from vid2slp.id_data import BUTTONS
from vid2slp.id_to_slp import BIT, pair_bits, processed_sequence
from vid2slp.slp_write import pre_frame_events, read_field, write_inputs

L, R = BIT["L"], BIT["R"]


def test_pair_new_press_while_held_switches_button():
    # shield held for 4 frames, with a second press (e.g. wavedash out of shield) on frame 2
    held = np.array([1, 1, 1, 1, 0], bool)
    edge = np.array([1, 0, 1, 0, 0], bool)
    bits = pair_bits(held, edge, L, R)
    assert list(bits) == [1 << L, 1 << L, 1 << R, 1 << R, 0]
    # the game sees a NEW press on frame 2: R goes down while it was up on frame 1
    assert (bits[2] & ~bits[1]) != 0


def test_pair_release_then_press_uses_first_button_again():
    bits = pair_bits(np.array([1, 0, 1], bool), np.array([1, 0, 1], bool), L, R)
    assert list(bits) == [1 << L, 0, 1 << L]


def _btn(**flags):
    return sum(int(flags.get(name, 0)) << i for i, name in enumerate(BUTTONS))


def test_processed_word_rules():
    tgt = np.array([[80, 80, 80, 80, 0],      # Z only
                    [160, 80, 80, 80, 140],   # stick right, full trigger, shield held
                    [80, 80, 80, 80, 0]])     # taunt
    btn = np.array([_btn(Z=1), _btn(SHIELD=1, SHIELD_EDGE=1), _btn(DPAD_UP=1)])
    seq = processed_sequence(tgt, btn)
    z, full, taunt = (s["buttons"] for s in seq)
    assert z == (1 << BIT["Z"]) | (1 << BIT["A"]) | (1 << BIT["TRIGGER"])  # Z also sets A + trigger flag
    assert abs(seq[0]["trigger"] - 0.35) < 1e-9                            # Z gives a 0.35 trigger
    assert full == (1 << L) | (1 << BIT["TRIGGER"]) and seq[1]["stick_x"] == 1.0
    assert taunt == 1 << BIT["DPAD_UP"]


REPLAY = "/home/prompter/ssbm-vid2slp/data/recordings/examples/Game_20260425T023703.slp"


@pytest.mark.skipif(not os.path.exists(REPLAY), reason="replay library not present")
def test_write_inputs_changes_only_requested_fields(tmp_path):
    out = tmp_path / "x.slp"
    write_inputs(REPLAY, out, lambda f, p: {"stick_x": 0.5} if f == 100 and p == 0 else None)
    a, b = open(REPLAY, "rb").read(), open(out, "rb").read()
    assert len(a) == len(b)
    changed = [i for i in range(len(a)) if a[i] != b[i]]
    pos = [pos for pos, f, p in pre_frame_events(a) if f == 100 and p == 0]
    assert read_field(b, pos[-1], "stick_x") == 0.5
    assert all(pos[0] + 0x19 <= i < pos[0] + 0x1D or pos[-1] + 0x19 <= i < pos[-1] + 0x1D for i in changed)
