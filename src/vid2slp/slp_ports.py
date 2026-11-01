"""
slp_ports.py - move the players of a .slp replay to other controller ports.

Why: every replay in our library uses ports 1 and 2 (netplay always does), but at tournaments
players pick any port, and the damage/stock display of a player is drawn at a place that
depends on the port (seen on Genesis 7 footage: Leffen on port 3). Perception has only ever
seen ports 1 and 2. Rewriting some replays to other ports and rendering them gives labelled
training footage with the other layouts. (Added 2026-11-01.)

What changes (Slippi replay spec, project-slippi/slippi-wiki SPEC.md; offsets counted from the
event's command byte):
  Game Start (0x36): the 4 player blocks (0x65 + 0x24 * i, 0x24 bytes each), and the per-port
      arrays: controller fixes (0x141, 8 bytes each), nametags (0x161, 16), display names
      (0x1A5, 31), connect codes (0x221, 10), Slippi user IDs (0x249, 29).
  Pre-Frame (0x37) and Post-Frame (0x38): player index (0x05).
  Post-Frame: "last hit by" (0x20), when it is a player index.
  Item Update (0x3B): owner (0x2A).
  Game End (0x39): LRAS initiator (0x02) and the placements array (0x03, one byte per port).
The metadata block at the end (UBJSON) is left alone; playback does not use it.

The players keep their order (the lower port stays the lower port), so "slot 0 = lower port"
labels from data/parsed/ stay right.
"""

import struct

GAME_START, PRE, POST, GAME_END, ITEM = 0x36, 0x37, 0x38, 0x39, 0x3B
# (offset, size of one port's entry) of every per-port array in Game Start
GAME_START_ARRAYS = [(0x65, 0x24), (0x141, 8), (0x161, 0x10), (0x1A5, 0x1F), (0x221, 0xA), (0x249, 0x1D)]


def permutation(new_ports):
    """new_ports: the new port numbers (1-4) of the players now on ports 1 and 2, e.g. (1, 3).
    Returns a full mapping old index -> new index for indexes 0-3 (the empty ports fill the rest)."""
    p, q = new_ports
    assert 1 <= p < q <= 4, "ports must be 1-4 and keep their order"
    mapping = {0: p - 1, 1: q - 1}
    free_new = [i for i in range(4) if i not in mapping.values()]
    for old, new in zip([2, 3], free_new):
        mapping[old] = new
    return mapping


def move_ports(data, new_ports):
    """.slp bytes -> .slp bytes with the two players on new_ports (see permutation())."""
    mapping = permutation(new_ports)
    out = bytearray(data)
    i = data.find(b"raw[$U#l") + 12          # first byte of the raw event stream (0x35)
    raw_size = struct.unpack(">I", data[i - 4:i])[0]
    n = data[i + 1]
    sizes = {data[i + 2 + 3 * k]: struct.unpack(">H", data[i + 3 + 3 * k:i + 5 + 3 * k])[0]
             for k in range((n - 1) // 3)}
    end = i + raw_size if raw_size else len(data)
    pos = i + 1 + n
    while pos < end:
        command = data[pos]
        if command not in sizes:
            break
        size = sizes[command]
        if command == GAME_START:
            for offset, length in GAME_START_ARRAYS:
                if offset + 4 * length > 1 + size:
                    continue                        # older replay without this array
                old = [bytes(data[pos + offset + length * k:pos + offset + length * (k + 1)]) for k in range(4)]
                for k in range(4):
                    a = pos + offset + length * mapping[k]
                    out[a:a + length] = old[k]
        elif command in (PRE, POST):
            out[pos + 5] = mapping[data[pos + 5]]
            if command == POST and size >= 0x20 and data[pos + 0x20] in mapping:
                out[pos + 0x20] = mapping[data[pos + 0x20]]
        elif command == ITEM and size >= 0x2A and data[pos + 0x2A] in mapping:
            out[pos + 0x2A] = mapping[data[pos + 0x2A]]
        elif command == GAME_END:
            if size >= 0x02 and data[pos + 2] in mapping:
                out[pos + 2] = mapping[data[pos + 2]]
            if size >= 0x06:
                old = bytes(data[pos + 3:pos + 7])
                for k in range(4):
                    out[pos + 3 + mapping[k]] = old[k]
        pos += 1 + size
    return bytes(out)
