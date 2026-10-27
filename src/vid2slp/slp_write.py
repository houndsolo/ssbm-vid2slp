"""
slp_write.py - write controller inputs into a copy of a .slp replay.

We take a real .slp as a template (it already has the correct game settings, stage,
characters, random seeds, ...) and overwrite the INPUT fields of its per-frame
"pre-frame update" events. Everything else is copied byte for byte.

.slp layout (Slippi spec, https://github.com/project-slippi/slippi-wiki/blob/master/SPEC.md):
  - The file is UBJSON: {"raw": [bytes...], "metadata": {...}}. The "raw" array is a stream
    of events: 1 command byte + a payload whose size is declared by the first event (0x35).
  - Pre-frame update (command 0x37), offsets from the command byte, big-endian:
        0x01 frame s32   0x05 port u8   0x06 is_follower u8
        0x19 joystick x f32   0x1D joystick y f32   0x21 c-stick x f32   0x25 c-stick y f32
        0x29 trigger f32      0x2D processed buttons u32   0x31 physical buttons u16
        0x33 physical L f32   0x37 physical R f32   0x3B raw stick x s8   0x40 raw stick y s8
"""

import struct

PRE_FRAME = 0x37
PAYLOAD_SIZES = 0x35
# field name -> (offset from command byte, struct format)
FIELDS = {
    "stick_x": (0x19, ">f"), "stick_y": (0x1D, ">f"),
    "cstick_x": (0x21, ">f"), "cstick_y": (0x25, ">f"),
    "trigger": (0x29, ">f"),
    "buttons": (0x2D, ">I"),
    "buttons_physical": (0x31, ">H"),
    "trigger_l_physical": (0x33, ">f"), "trigger_r_physical": (0x37, ">f"),
    "raw_stick_x": (0x3B, ">b"), "raw_stick_y": (0x40, ">b"),
}


def _raw_bounds(data):
    """Return (start, end) byte positions of the event stream inside the UBJSON file."""
    head = b"{U\x03raw[$U#l"
    assert data[:len(head)] == head, "not a .slp file (unexpected header)"
    length = struct.unpack(">I", data[len(head):len(head) + 4])[0]
    start = len(head) + 4
    return start, start + length


def pre_frame_events(data):
    """Yield (byte position, frame, port) for every LEADER pre-frame event."""
    start, end = _raw_bounds(data)
    assert data[start] == PAYLOAD_SIZES
    info_size = data[start + 1]
    sizes = {}
    for i in range(start + 2, start + 1 + info_size, 3):
        sizes[data[i]] = struct.unpack(">H", data[i + 1:i + 3])[0]
    pos = start
    while pos < end:
        cmd = data[pos]
        if cmd == PAYLOAD_SIZES:
            pos += 1 + info_size
            continue
        if cmd == PRE_FRAME and data[pos + 6] == 0:  # leader only (not Ice Climbers' Nana)
            frame = struct.unpack(">i", data[pos + 1:pos + 5])[0]
            yield pos, frame, data[pos + 5]
        pos += 1 + sizes[cmd]


def read_field(data, pos, name):
    off, fmt = FIELDS[name]
    return struct.unpack(fmt, data[pos + off:pos + off + struct.calcsize(fmt)])[0]


def write_inputs(template_path, out_path, values, fields=None, payload_size_check=True):
    """Copy template_path to out_path with some input fields replaced.

    values: function (frame, port) -> dict {field name: value} or None to leave that frame alone.
            Called for every leader pre-frame event, INCLUDING rollback duplicates of a frame
            (all copies get the same value, so playback sees consistent inputs).
    fields: optional allow-list of field names that may be written.
    Returns the number of events modified.
    """
    data = bytearray(open(template_path, "rb").read())
    start, _ = _raw_bounds(data)
    info_size = data[start + 1]
    pre_size = None
    for i in range(start + 2, start + 1 + info_size, 3):
        if data[i] == PRE_FRAME:
            pre_size = struct.unpack(">H", data[i + 1:i + 3])[0]
    changed = 0
    for pos, frame, port in list(pre_frame_events(bytes(data))):
        new = values(frame, port)
        if not new:
            continue
        for name, value in new.items():
            if fields is not None and name not in fields:
                continue
            off, fmt = FIELDS[name]
            if payload_size_check and off + struct.calcsize(fmt) > pre_size + 1:
                continue  # this replay's Slippi version doesn't have that field
            data[pos + off:pos + off + struct.calcsize(fmt)] = struct.pack(fmt, value)
        changed += 1
    open(out_path, "wb").write(bytes(data))
    return changed
