"""
id_to_slp.py - turn predicted inputs (milestone-2 targets) into a playable .slp.

Slippi Playback feeds the game the PROCESSED input fields of each pre-frame event
(stick/c-stick/trigger floats + processed button word); the physical fields are ignored.
Measured with scripts/m2_resim_probe.py on 2026-10-26, which also showed:
  - the stick/c-stick direction flags (bits 16-23) are recomputed by the game -> leave 0
  - bit 31 ("trigger pressed") matters, and in real data it is exactly (trigger > 0)
  - X vs Y and L vs R are interchangeable as long as every NEW press is reproduced
  - when Z is held the processed trigger is always 49/140 (0.35) and the A bit is also set
  - D-pad up is the taunt and must be written (a later oracle check found this)

Jump (X/Y) and shield (L/R) pairs: the targets say "any of the pair held" plus "a button of
the pair went down this frame" (EDGE). The game reacts to a button going down, so to
reproduce a new press while the pair is already held, we switch to the OTHER button of the
pair (e.g. L held, new press -> release L, press R). Only one button of a pair is ever down.
"""

import numpy as np

from vid2slp.id_data import BUTTONS
from vid2slp.slp_write import write_inputs

# processed button word bit numbers
BIT = {"DPAD_UP": 3, "Z": 4, "R": 5, "L": 6, "A": 8, "B": 9, "X": 10, "Y": 11, "TRIGGER": 31}
_I = {name: i for i, name in enumerate(BUTTONS)}


def pair_bits(held, edge, first_bit, second_bit):
    """Per-frame bits for a button pair from 'any held' and 'new press' sequences (one player).

    held, edge: bool arrays [T]. Returns an int array [T] with first_bit or second_bit set.
    """
    out = np.zeros(len(held), np.int64)
    current = None
    for t in range(len(held)):
        down = bool(held[t] or edge[t])  # a press implies the pair is down this frame
        if not down:
            current = None
        elif current is None:
            current = first_bit          # pair was up: any button gives a new press
        elif edge[t]:
            current = second_bit if current == first_bit else first_bit  # switch = new press
        if current is not None:
            out[t] = 1 << current
    return out


def processed_sequence(tgt, btn):
    """One player's whole game: tgt [T, 5], btn [T] -> list of per-frame .slp field dicts."""
    btn = np.asarray(btn).astype(np.int64)
    bit = lambda name: ((btn >> _I[name]) & 1).astype(bool)
    jump = pair_bits(bit("JUMP"), bit("JUMP_EDGE"), BIT["X"], BIT["Y"])
    shield = pair_bits(bit("SHIELD"), bit("SHIELD_EDGE"), BIT["L"], BIT["R"])
    a, b, z, dpad_up = bit("A"), bit("B"), bit("Z"), bit("DPAD_UP")
    trigger = np.where(z, 49, np.asarray(tgt)[:, 4]).astype(np.int64)  # Z always gives 0.35
    on = lambda flags, name: flags.astype(np.int64) << BIT[name]
    words = (on(a | z, "A") | on(b, "B") | on(z, "Z") | on(dpad_up, "DPAD_UP") | jump | shield
             | on(trigger > 0, "TRIGGER"))
    tgt = np.asarray(tgt).astype(np.int64)
    return [{
        "stick_x": (tgt[t, 0] - 80) / 80, "stick_y": (tgt[t, 1] - 80) / 80,
        "cstick_x": (tgt[t, 2] - 80) / 80, "cstick_y": (tgt[t, 3] - 80) / 80,
        "trigger": trigger[t] / 140, "buttons": int(words[t]),
    } for t in range(len(tgt))]


def write_predicted_slp(template_path, out_path, tgt, btn, ports):
    """Write a copy of template_path whose inputs for frames >= 0 come from tgt/btn.

    tgt [T, 2, 5], btn [T, 2]: row t = frame t; column 0/1 = the lower/higher of `ports`.
    Frames < 0 (countdown, nobody can act) keep the template's inputs.
    """
    col = {int(p): i for i, p in enumerate(sorted(ports))}
    seqs = [processed_sequence(np.asarray(tgt)[:, i], np.asarray(btn)[:, i]) for i in range(2)]

    def values(frame, port):
        if frame < 0 or frame >= len(seqs[0]):
            return None
        return seqs[col[int(port)]][frame]

    return write_inputs(template_path, out_path, values)
