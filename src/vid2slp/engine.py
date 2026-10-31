"""
engine.py - drive the native Melee engine (Melee Unlocked's headless Linux host) from Python.

Why: Stage C needs a SIMULATOR. Slippi Playback (Dolphin) plays a .slp in real time and can't
go back. The native engine runs the same game code ~20x faster and can save its state
("capture") and go back to it ("restore") in ~2 ms. vid2slp added Slippi replay playback to its
headless host (data/native/melee-unlocked/port/linux/replay_source.cpp, built by
scripts/native_build.sh): it plays a .slp like Slippi Playback does (same seeds, same gameplay
codes, optional resync), and lets us OVERRIDE any frame's inputs or stored states before the game
reaches that frame.

How it talks: the engine runs as a child process; we talk over a socket pair with one text line
per command (protocol "MUCONTROL 2", port/linux/retrace_control.cpp):
    step <40 numbers>  advance one frame (the 40 numbers are pad values, unused in replay mode)
    capture            save the state (one checkpoint; a new capture replaces it)
    restore            go back to the saved state
    override ...       replace a frame's inputs       (replay_source.h)
    override-state ... replace a frame's stored state (used with resync)
    clear-overrides
    override-seed      force a frame's random seed
    frame-seeds 0|1    0 = don't force the file's per-frame random seeds
    record             the played game so far, as a .slp file
After every step / capture / restore the engine answers with a STATE line:
    STATE retraces,scene_major,scene,match_frame,rng, then for each of 4 ports:
          present,stocks,action,anim_frame,x,y,z,vel_x,vel_y,vel_z,percent,facing
    (the float fields are sent as their 32-bit patterns)

Usage:
    with Engine("data/m4/x.slp") as e:
        s = e.state()               # after Slippi frame s["match_frame"] (= 1 at the start)
        e.override(s["match_frame"] + 1, port, ...)   # inputs of the NEXT frame
        e.capture()
        s = e.step()
"""

import os
import socket
import struct
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HEADLESS = f"{ROOT}/data/native/build/headless/melee_headless"
GAME = f"{ROOT}/data/tools/native-engine/build/game/melee_game.so"
ISO = f"{ROOT}/data/tools/native-engine/assets/melee.iso"
SYS = f"{ROOT}/data/tools/playback/squashfs-root/usr/bin/Sys"   # Slippi Playback's Sys folder
NEUTRAL_STEP = "step " + " ".join(["0 0 0 0 0 0 0 0 0 0"] * 4)

FIELDS = ["present", "stocks", "action", "anim_frame", "x", "y", "z", "vel_x", "vel_y", "vel_z", "percent", "facing"]
FLOATS = {"anim_frame", "x", "y", "z", "vel_x", "vel_y", "vel_z", "percent", "facing"}


def bits_to_float(value):
    return struct.unpack("<f", struct.pack("<I", int(value) & 0xFFFFFFFF))[0]


def parse_state(line):
    """'STATE a,b,c,...' -> dict with match_frame, rng and players[4] (dicts of FIELDS).
    Float fields arrive as their 32-bit patterns; all 4 ports are converted in one struct call
    (2026-10-30: ~4x faster than one call per value; same values)."""
    values = [int(v) for v in line.split(" ", 1)[1].split(",")]
    out = {"retraces": values[0], "scene_major": values[1], "scene": values[2],
           "match_frame": values[3], "rng": values[4] & 0xFFFFFFFF, "players": []}
    body = values[5:5 + 12 * 4]
    floats = _FLOATS48.unpack(_UINTS48.pack(*[v & 0xFFFFFFFF for v in body]))
    for port in range(4):
        k = 12 * port
        out["players"].append({name: (floats[k + j] if is_float else body[k + j])
                               for j, (name, is_float) in enumerate(_FIELD_KINDS)})
    return out


_FIELD_KINDS = [(name, name in FLOATS) for name in FIELDS]
_UINTS48, _FLOATS48 = struct.Struct("<48I"), struct.Struct("<48f")


def trim_to_complete_frames(data):
    """Cut a recorded .slp after its last COMPLETE frame (the last Frame Bookend event, 0x3C).

    A recording taken in the middle of a game ends with a frame whose events are not all
    written yet; replay parsers (peppi) reject that. The raw block's length field is fixed up.
    """
    i = data.find(b"raw[$U#l") + 12          # first byte of the raw event stream (0x35)
    raw_size = struct.unpack(">I", data[i - 4:i])[0]
    n = data[i + 1]
    sizes = {data[i + 2 + 3 * k]: struct.unpack(">H", data[i + 3 + 3 * k:i + 5 + 3 * k])[0]
             for k in range((n - 1) // 3)}
    end = i + raw_size if raw_size else len(data)
    pos, cut = i + 1 + n, None
    while pos < end:
        command = data[pos]
        if command not in sizes:
            break
        pos += 1 + sizes[command]
        if command == 0x3C:                  # Frame Bookend: everything up to here is complete
            cut = pos
    if cut is None or cut == end:
        return data
    return data[:i - 4] + struct.pack(">I", cut - i) + data[i:cut] + data[end:]


class Engine:
    def __init__(self, slp, resync=False, log_file=None, boot_limit=6000, codes=None, preroll=False,
                 frame_offset=0):
        """slp: the replay to play. codes (optional): a file with a raw Gecko code list to use
        instead of the replay's own, e.g. data/m4/ucf084_gecko.bin (UCF 0.84, the only UCF the
        engine implements); the recording then carries that list too.

        preroll: hand over control at the start of the countdown (Slippi frame -123) instead of
        at frame 1. Players can already move from frame -39. The game's own frame counter is 0
        during the countdown, so the frame number is counted here (measured 2026-10-31: the
        hand-over state is frame -123, and each step is one frame; it agrees with the game's
        counter from frame 1 on and with the recorded frame numbers).
        frame_offset: every frame number going in (overrides, seeds) and coming out ("match_frame"
        of a state) is the Slippi frame + frame_offset. With 39 the first frame players can move
        is 0, so arrays starting there can be indexed by frame."""
        self.preroll, self.offset = preroll, frame_offset
        self._count = -123                   # Slippi frame of the current state while counting
        ours, theirs = socket.socketpair()
        cmd = [HEADLESS, "--iso", ISO, "--game-library", GAME, "--headless", "--fast",
               "--replay", os.path.abspath(slp), "--replay-sys", SYS,
               "--control-fd", str(theirs.fileno()), "--boot-limit", str(boot_limit)]
        if resync:
            cmd.append("--replay-resync")
        if codes:
            cmd += ["--replay-codes", os.path.abspath(codes)]
        if preroll:
            cmd.append("--control-preroll")
        if log_file:
            cmd += ["--log-file", log_file]
        self.proc = subprocess.Popen(cmd, pass_fds=[theirs.fileno()], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE)
        theirs.close()
        self.sock = ours
        self.reader = ours.makefile("r", encoding="ascii", newline="\n")
        hello = self._line()
        if hello != "MUCONTROL 2":
            raise RuntimeError(f"engine said {hello!r}: {self.proc.stderr.read().decode()[-2000:] if self.proc.poll() is not None else ''}")
        self._state = self._fix(parse_state(self._line()))
        self._saved_count = self._count

    def _line(self):
        line = self.reader.readline()
        if not line:
            err = self.proc.stderr.read().decode()[-2000:] if self.proc.poll() is not None else ""
            raise RuntimeError(f"engine closed the connection. {err}")
        return line.rstrip("\n")

    def _fix(self, st):
        """Put the Slippi frame (+ offset) into st["match_frame"] (see __init__)."""
        if self.preroll and st["match_frame"] == 0:
            st["match_frame"] = self._count
        st["match_frame"] += self.offset
        return st

    def _send(self, text):
        self.sock.sendall((text + "\n").encode("ascii"))

    def state(self):
        """The state at the current boundary (before the next step)."""
        return self._state

    def step(self):
        self._send(NEUTRAL_STEP)
        self._count += 1
        self._state = self._fix(parse_state(self._line()))
        return self._state

    def capture(self):
        self._send("capture")
        reply = self._line()
        if reply not in ("CAPTURED", "RESTORED"):
            raise RuntimeError(f"capture failed: {reply}")
        self._saved_count = self._count
        self._state = self._fix(parse_state(self._line()))
        return self._state

    def restore(self):
        self._send("restore")
        reply = self._line()
        if reply != "RESTORED":
            raise RuntimeError(f"restore failed: {reply}")
        self._count = self._saved_count
        self._state = self._fix(parse_state(self._line()))
        return self._state

    def override(self, frame, port, stick_x, stick_y, cstick_x, cstick_y, trigger, buttons,
                 raw_x, raw_y, raw_cx, raw_cy):
        """Replace the inputs of (Slippi) frame `frame`, port `port` (processed values)."""
        self._send(f"override {frame - self.offset} {port} {stick_x:.9g} {stick_y:.9g} {cstick_x:.9g} {cstick_y:.9g} "
                   f"{trigger:.9g} {int(buttons)} {int(raw_x)} {int(raw_y)} {int(raw_cx)} {int(raw_cy)}")
        self._expect_ok()

    def override_state(self, frame, port, x, y, facing, action, percent):
        """Replace the stored start-of-frame state that resync forces on frame `frame`."""
        self._send(f"override-state {frame - self.offset} {port} {x:.9g} {y:.9g} {facing:.9g} {int(action)} {percent:.9g}")
        self._expect_ok()

    def override_many(self, rows):
        """Several overrides at once (sent together, then the replies are read: much faster than
        one round trip each). rows: list of argument tuples for override()."""
        lines = [f"override {f - self.offset} {p} {sx:.9g} {sy:.9g} {cx:.9g} {cy:.9g} {tr:.9g} {int(b)} "
                 f"{int(rx)} {int(ry)} {int(rcx)} {int(rcy)}"
                 for f, p, sx, sy, cx, cy, tr, b, rx, ry, rcx, rcy in rows]
        # In batches: every "OK" reply takes ~1 KB of socket buffer on the engine's side, so
        # thousands of unread replies would block the engine while we are still sending (deadlock).
        for k in range(0, len(lines), 100):
            batch = lines[k:k + 100]
            self._send("\n".join(batch))
            for _ in batch:
                self._expect_ok()

    def steps(self, n):
        """Advance n frames; returns the n states (sent together, like override_many)."""
        self._send("\n".join([NEUTRAL_STEP] * n))
        out = []
        for _ in range(n):
            self._count += 1
            out.append(self._fix(parse_state(self._line())))
        self._state = out[-1]
        return out

    def override_seed(self, frame, seed):
        """Force random seed `seed` (32-bit) at the start of `frame`. The recording stores it and
        Slippi Playback forces it too, so a rebuilt game may choose its own seeds."""
        self._send(f"override-seed {int(frame) - self.offset} {int(seed) & 0xFFFFFFFF}")
        self._expect_ok()

    def frame_seeds(self, on):
        """False: stop forcing the replay's per-frame random seeds (a rebuilt game must not use
        the original game's seeds; the game's own RNG runs on and the recording stores it)."""
        self._send(f"frame-seeds {1 if on else 0}")
        self._expect_ok()

    def clear_overrides(self):
        self._send("clear-overrides")
        self._expect_ok()

    def _expect_ok(self):
        reply = self._line()
        if reply != "OK":
            raise RuntimeError(reply)

    def record(self, trim=True):
        """The game played so far, as .slp bytes (cut after the last complete frame)."""
        self._send("record")
        head = self._line()
        if not head.startswith("REPLAY "):
            raise RuntimeError(head)
        size = int(head.split()[1])
        chunks = []
        while True:
            line = self._line()
            if line == "RECORDED":
                break
            chunks.append(line)
        data = bytes.fromhex("".join(chunks))
        assert len(data) == size
        return trim_to_complete_frames(data) if trim else data

    def close(self):
        try:
            self._send("quit")
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
