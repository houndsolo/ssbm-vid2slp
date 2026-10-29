"""
slp_codes.py - read the Gecko code list a .slp carries (the game modifications that were active,
e.g. UCF), straight from the file's raw events.

Why: replays recorded with different Slippi versions carry different code versions. The native
engine (src/vid2slp/engine.py) implements UCF 0.84 only; ~54% of our Fox games (Slippi 3.13-3.16)
were played with the older UCF 0.8, which behaves slightly differently (e.g. no "1.0 cardinals"),
so the engine can't replay those exactly (measured 2026-10-29, scripts/native_fidelity.py).

The code list is stored in Message Splitter events (0x10, 512 bytes of data each) that carry
command 0x3D (Gecko list), or in one 0x3D event in older files.
"""

import struct

UCF084_PAD_BUFFER = 0x8006B460   # only UCF 0.84 has this code (pad buffer + 1.0 cardinals)
UCF_DASHBACK = 0x800C9A44        # every UCF version has this one


def raw_events(data):
    """Yield (command, payload) of the events before the first frame."""
    i = data.find(b"raw[$U#l") + 12
    if data[i] != 0x35:
        raise ValueError("no event payload table")
    n = data[i + 1]
    sizes = {data[i + 2 + 3 * k]: struct.unpack(">H", data[i + 3 + 3 * k:i + 5 + 3 * k])[0]
             for k in range((n - 1) // 3)}
    pos = i + 1 + n
    while pos < len(data):
        command = data[pos]
        pos += 1
        if command not in sizes:
            break
        yield command, data[pos:pos + sizes[command]]
        pos += sizes[command]
        if command == 0x3A:      # Frame Start: the code list always comes before the first frame
            break


def gecko_list(path):
    """The raw Gecko code list bytes stored in a .slp (b"" if none)."""
    data = open(path, "rb").read()
    out = b""
    for command, payload in raw_events(data):
        if command == 0x10 and len(payload) >= 516 and payload[514] == 0x3D:
            size = struct.unpack(">H", payload[512:514])[0]
            out += payload[:size]
        elif command == 0x3D:
            out += payload
    return out


def codes(gecko):
    """Split a Gecko list into (address, type, bytes) entries."""
    idx = 0
    while idx + 8 <= len(gecko):
        kind = gecko[idx] & 0xFE
        address = (struct.unpack(">I", gecko[idx:idx + 4])[0] & 0x01FFFFFF) | 0x80000000
        word = struct.unpack(">I", gecko[idx + 4:idx + 8])[0]
        length = 8
        if kind in (0xC0, 0xC2):
            length = 8 + word * 8
        elif kind == 0x08:
            length = 16
        elif kind == 0x06:
            length = 8 + ((word + 7) & ~7)
        yield address, kind, gecko[idx:idx + length]
        idx += length


def ucf_version(path):
    """'0.84', 'old' (UCF without the 0.84 pad buffer), or 'none'."""
    addresses = {a for a, _, _ in codes(gecko_list(path))}
    if UCF084_PAD_BUFFER in addresses:
        return "0.84"
    if UCF_DASHBACK in addresses:
        return "old"
    return "none"
