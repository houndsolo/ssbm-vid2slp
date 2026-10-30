"""stick_from_raw must reproduce the game's processed stick from raw bytes (checked on real data
2026-10-30; these are hand-picked cases from that data)."""

from vid2slp.id_data import stick_from_raw


def test_examples_from_real_games():
    cases = [
        ((0, 0), (0, 0)),
        ((22, -22), (0, 0)),        # dead zone
        ((80, 0), (80, 0)),         # exactly on the rim
        ((98, -1), (79, 0)),        # past the rim: pulled onto the circle, truncated
        ((94, 11), (79, 0)),        # small y counts for the length but is then dead-zoned
        ((28, -124), (0, -78)),     # x shrinks below 23 after scaling -> 0
        ((57, -55), (57, -55)),     # inside the circle: unchanged
    ]
    for (rx, ry), (px, py) in cases:
        got = stick_from_raw(rx, ry)
        assert (int(got[0]), int(got[1])) == (px, py), (rx, ry, got)
