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
    record             the played game so far, as a .slp file
After every step / capture / restore the engine answers with a STATE line:
    STATE retraces,scene_major,scene,match_frame,rng, then for each of 4 ports:
          present,stocks,action,anim_frame,x,y,z,vel_x,vel_y,vel_z,percent,facing
    (the float fields are sent as their 32-bit patterns)

Usage:
    with Engine("data/m4/x.slp") as e:
        s = e.state()               # first frame of the match
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
    """'STATE a,b,c,...' -> dict with match_frame, rng and players[4] (dicts of FIELDS)."""
    values = line.split(" ", 1)[1].split(",")
    out = {"retraces": int(values[0]), "scene_major": int(values[1]), "scene": int(values[2]),
           "match_frame": int(values[3]), "rng": int(values[4]) & 0xFFFFFFFF, "players": []}
    for port in range(4):
        raw = values[5 + 12 * port: 5 + 12 * (port + 1)]
        player = {}
        for name, v in zip(FIELDS, raw):
            player[name] = bits_to_float(v) if name in FLOATS else int(v)
        out["players"].append(player)
    return out


class Engine:
    def __init__(self, slp, resync=False, log_file=None, boot_limit=6000):
        ours, theirs = socket.socketpair()
        cmd = [HEADLESS, "--iso", ISO, "--game-library", GAME, "--headless", "--fast",
               "--replay", os.path.abspath(slp), "--replay-sys", SYS,
               "--control-fd", str(theirs.fileno()), "--boot-limit", str(boot_limit)]
        if resync:
            cmd.append("--replay-resync")
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
        self._state = parse_state(self._line())

    def _line(self):
        line = self.reader.readline()
        if not line:
            err = self.proc.stderr.read().decode()[-2000:] if self.proc.poll() is not None else ""
            raise RuntimeError(f"engine closed the connection. {err}")
        return line.rstrip("\n")

    def _send(self, text):
        self.sock.sendall((text + "\n").encode("ascii"))

    def state(self):
        """The state at the current boundary (before the next step)."""
        return self._state

    def step(self):
        self._send(NEUTRAL_STEP)
        self._state = parse_state(self._line())
        return self._state

    def capture(self):
        self._send("capture")
        reply = self._line()
        if reply not in ("CAPTURED", "RESTORED"):
            raise RuntimeError(f"capture failed: {reply}")
        self._state = parse_state(self._line())
        return self._state

    def restore(self):
        self._send("restore")
        reply = self._line()
        if reply != "RESTORED":
            raise RuntimeError(f"restore failed: {reply}")
        self._state = parse_state(self._line())
        return self._state

    def override(self, frame, port, stick_x, stick_y, cstick_x, cstick_y, trigger, buttons,
                 raw_x, raw_y, raw_cx, raw_cy):
        """Replace the inputs of (Slippi) frame `frame`, port `port` (processed values)."""
        self._send(f"override {frame} {port} {stick_x:.9g} {stick_y:.9g} {cstick_x:.9g} {cstick_y:.9g} "
                   f"{trigger:.9g} {int(buttons)} {int(raw_x)} {int(raw_y)} {int(raw_cx)} {int(raw_cy)}")
        self._expect_ok()

    def override_state(self, frame, port, x, y, facing, action, percent):
        """Replace the stored start-of-frame state that resync forces on frame `frame`."""
        self._send(f"override-state {frame} {port} {x:.9g} {y:.9g} {facing:.9g} {int(action)} {percent:.9g}")
        self._expect_ok()

    def clear_overrides(self):
        self._send("clear-overrides")
        self._expect_ok()

    def _expect_ok(self):
        reply = self._line()
        if reply != "OK":
            raise RuntimeError(reply)

    def record(self):
        """The game played so far, as .slp bytes."""
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
        return data

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
