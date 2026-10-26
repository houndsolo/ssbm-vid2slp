"""Checks for the milestone-2 model helpers (src/vid2slp/id_model.py)."""

import numpy as np
import torch

from vid2slp.id_model import InverseDynamics, decode, window_jobs


def test_windows_cover_every_frame_exactly_once():
    # games of awkward lengths: exact multiples of the stride, one frame over, etc.
    lengths = [256, 257, 383, 384, 385, 1800, 2049]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    covered = np.zeros(offsets[-1], int)
    for s, keep_from, keep_to in window_jobs(offsets, 256):
        assert s <= keep_from < keep_to <= s + 256          # kept rows lie inside the window
        covered[keep_from:keep_to] += 1
    assert (covered == 1).all()
    # windows never cross a game boundary
    for s, _, _ in window_jobs(offsets, 256):
        g = np.searchsorted(offsets, s, side="right") - 1
        assert s + 256 <= offsets[g + 1]


def test_model_shapes_and_decode():
    model = InverseDynamics(d_model=32, layers=1, heads=2, window=16)
    num = torch.randn(3, 16, 2, 17)
    cat = torch.zeros(3, 16, 2, 7, dtype=torch.int16)
    out = model(num, cat)
    tgt, btn = decode(out)
    assert tgt.shape == (3, 16, 5) and btn.shape == (3, 16)
    assert int(tgt[..., :4].max()) <= 160 and int(tgt[..., 4].max()) <= 140
