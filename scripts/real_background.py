"""
real_background.py - make Battlefield's background in a rebuilt replay change when and how it
changes in the real video.

Why: Battlefield's background is not fixed. It starts as background 1 (dark purple space); after a
random wait of 2400-3599 frames (40-60 s) a "tunnel" of grid lines plays for 400 frames, then a
random OTHER background is picked (2 = green glow, 4 = red glow, 1), the old one fades out for 200
frames, a new random wait is drawn, and so on (decomp: src/melee/gr/grbattle.c,
grBattle_BG_Callback2). The random numbers come from the game's random number generator, so our
rebuilt replay (its own random numbers) changed backgrounds at other times and to other ones than
the video (seen on Genesis 7 set 1, 2026-11-04).

The background does not change the gameplay. Its three random draws per change can be steered:
  - the first wait is drawn while the stage loads, from the replay's game start seed
    (Game Start event offset 0x13D): we try seeds until the wait matches;
  - each later choice and wait is drawn in a known frame (the choice 400 frames after the tunnel
    starts, the next wait 200 frames after that): the engine can force that frame's random seed
    (Engine.override_seed; the recording stores it, Slippi Playback forces it too). We try seeds
    until the draw is what the video shows, and keep only seeds after which the players' states
    stay exactly the same as before (the gameplay must not change).
The background's state is read straight from the engine's memory (/proc/<pid>/mem) at addresses
from the engine's debug info (data/tools/native-engine/build/game/melee_game.map and .dbg):
stage_info.map_gobjs[3] -> its Ground -> u.battle_bg {state, curr, prev, timer}.

What the video shows is measured per game frame on the cut game video:
  - tint: green minus red of the dark background pixels (background 2 is greener, 4 is redder);
  - tunnel score: fine line detail in the dark background (the tunnel's grid lines).
A background change is where the tint settles on a new background; its start is found from the
tunnel score's peak before that.

Run (after real_rebuild.py; Battlefield games only, others are skipped):
    .venv/bin/python scripts/real_background.py --id 5njRxIAgS-o --games 2,3 --tag v1 --out v2
Writes data/real/<id>/rebuild/<out>/game<k>.slp (and background.json with what was matched).
"""

import argparse
import json
import os
import shutil
import struct
import sys

import av
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_compare_video import filled                    # noqa: E402  (scripts/real_compare_video.py)
from vid2slp.engine import Engine, blank_player_info     # noqa: E402
from vid2slp.slp_parse import parse_replay               # noqa: E402

REAL = "data/real"
# engine memory layout (lp64 build of the decomp, from melee_game.map and the DWARF debug info)
STAGE_INFO = 0x82FD50E0          # address of stage_info (the game library is loaded at a fixed place)
MAP_GOBJS = 408                  # offset of stage_info.map_gobjs (8-byte pointers)
BG_GOBJ = 3                      # Battlefield's background is map object 3
USER_DATA = 72                   # offset of HSD_GObj.user_data (the Ground)
GROUND_U = 256                   # offset of Ground.u (here u.battle_bg: state, curr, prev, timer)
TUNNEL, FADE = 400, 200          # frames from the tunnel's start to the choice, and to the next wait
# tint (green - red) of the three backgrounds, measured on our renders: about -4 (1), +9 (2), -17 (4)
TINT_2, TINT_4 = 3.0, -9.0       # above TINT_2: background 2; below TINT_4: background 4; else 1
PEAK_AFTER_START = 175           # the smoothed tunnel score peaks this long after a change starts
                                 # (measured on our replay, where the true start is known)
PLAYER_FIELDS = ("present", "stocks", "action", "x", "y", "percent", "facing")


# ---- the video ---------------------------------------------------------------------------

def video_features(video, frame_map):
    """Per game frame: tint (green - red of the dark background) and tunnel score.
    Uses rows 80-330 of the 584 x 480 picture (below the timer, above the HUD)."""
    game = filled(frame_map)
    rows = []
    for i, frame in enumerate(av.open(video).decode(video=0)):
        if i >= len(game):
            break
        a = frame.to_ndarray(format="rgb24")[80:330]
        px = a.reshape(-1, 3).astype(np.int16)
        bright = px.max(1)
        dark = px[(bright < 140) & (bright > 12)]
        tint = float(dark[:, 1].mean() - dark[:, 0].mean()) if len(dark) else 0.0
        grey = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(grey, (3, 3), 0), cv2.CV_32F))
        away = cv2.dilate((grey > 90).astype(np.uint8), np.ones((7, 7), np.uint8)) == 0
        rows.append((game[i], tint, float(lap[away].mean()) if away.any() else 0.0))
    t = pd.DataFrame(rows, columns=["frame", "tint", "tunnel"]).groupby("frame").mean()
    return t.reindex(range(int(t.index.min()), int(t.index.max()) + 1)).interpolate()


def video_changes(t, min_stable=600):
    """Background changes seen in the video: [(start frame of the tunnel, new background)]."""
    tint = t.tint.rolling(61, center=True, min_periods=1).median()
    level = np.where(tint > TINT_2, 2, np.where(tint < TINT_4, 4, 1))
    level = pd.Series(level, index=t.index).rolling(301, center=True, min_periods=1).median().round().astype(int)
    # runs of one background, keeping only the long (stable) ones
    runs, start = [], level.index[0]
    for f, prev, cur in zip(level.index[1:], level.to_numpy()[:-1], level.to_numpy()[1:]):
        if cur != prev:
            runs.append((start, f - 1, int(prev)))
            start = f
    runs.append((start, level.index[-1], int(level.iloc[-1])))
    stable = [r for r in runs if r[1] - r[0] + 1 >= min_stable]
    tunnel = t.tunnel.rolling(101, center=True, min_periods=1).mean()
    changes = []
    for (a0, a1, bg_a), (b0, b1, bg_b) in zip(stable[:-1], stable[1:]):
        if bg_a == bg_b:
            continue
        # the tunnel plays just before the new background shows: its peak in the 1200 frames before
        window = tunnel.loc[max(a0, b0 - 1200):b0 + 200]
        changes.append((int(window.idxmax()) - PEAK_AFTER_START, bg_b))
    return changes, stable


# ---- the engine --------------------------------------------------------------------------

class Background:
    """Reads Battlefield's background state from a running engine's memory.

    The engine's checkpoint (capture/restore) works by forking: the process that captured stays
    frozen and the game goes on in a child, and every restore starts a new child
    (data/native/melee-unlocked/port/linux/process_checkpoint.cpp). So after a capture or restore
    call refresh(): it finds the live process again (the one of this engine's processes that has
    no child of its own)."""

    def __init__(self, engine):
        self.engine = engine
        # forked copies share the command line (replay path, control socket number, ...)
        self.key = open(f"/proc/{engine.proc.pid}/cmdline", "rb").read()
        self.mem = None
        self.refresh()

    def refresh(self):
        procs = {}
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                if open(f"/proc/{d}/cmdline", "rb").read() != self.key:
                    continue
                stat = open(f"/proc/{d}/stat").read()
            except OSError:
                continue
            procs[int(d)] = int(stat[stat.rindex(")") + 2:].split()[1])      # pid -> parent pid
        parents = set(procs.values())
        live = [pid for pid in procs if pid not in parents]
        if len(live) != 1:
            raise RuntimeError(f"cannot tell the live engine process: {procs}")
        if self.mem:
            self.mem.close()
        self.mem = open(f"/proc/{live[0]}/mem", "rb", buffering=0)

    def _read(self, addr, n):
        self.mem.seek(addr)
        return self.mem.read(n)

    def state(self):
        """(state, current background, previous background, wait timer); state 0 = waiting,
        1 = tunnel playing, 2 = old background fading out. None before the stage exists."""
        gobj = struct.unpack("<Q", self._read(STAGE_INFO + MAP_GOBJS + 8 * BG_GOBJ, 8))[0]
        if not gobj:
            return None
        ground = struct.unpack("<Q", self._read(gobj + USER_DATA, 8))[0]
        return struct.unpack("<4i", self._read(ground + GROUND_U, 16))


def players(st, ports):
    return [tuple(st["players"][p][c] for c in PLAYER_FIELDS) for p in ports]


def with_start_seed(data, seed):
    """.slp bytes with the game start event's random seed (offset 0x13D) replaced."""
    out = bytearray(data)
    i = data.find(b"raw[$U#l") + 12
    pos = i + 1 + data[i + 1]
    assert data[pos] == 0x36, "Game Start not where expected"
    out[pos + 0x13D:pos + 0x141] = struct.pack(">I", seed & 0xFFFFFFFF)
    return bytes(out)


def first_wait(path, codes):
    """The background's wait timer at the first frame (Slippi frame -123) for this file."""
    with Engine(path, codes=codes, preroll=True) as e:
        return Background(e).state()[3]


def try_seeds(e, bg, frame, ports, want, rng, limit=30000):
    """At the boundary before `frame` (captured), find a random seed for `frame` after which
    want(background state) is true and the players' states are the same as with the file's own
    seed. Returns (seed, state after the frame) or (None, None)."""
    e.capture()
    bg.refresh()
    ref = e.step()                                   # the frame with the file's own seed
    ref_players = players(ref, ports)
    for n in range(limit):
        e.restore()
        bg.refresh()
        seed = int(rng.integers(0, 2**32))
        e.override_seed(frame, seed)
        st = e.step()
        if want(bg.state()) and players(st, ports) == ref_players:
            return seed, st
    e.restore()
    bg.refresh()
    return None, None


def adopt_orphans():
    """Make this process a "child subreaper": when the engine's checkpoint ends an old frozen
    copy, its live child is handed to us instead of to init. Linux (Yama ptrace_scope 1) lets a
    process read the memory of its own descendants only, so this keeps /proc/<pid>/mem readable."""
    import ctypes
    PR_SET_CHILD_SUBREAPER = 36
    if ctypes.CDLL(None, use_errno=True).prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_CHILD_SUBREAPER) failed")


def main():
    adopt_orphans()
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--games", default="1")
    ap.add_argument("--tag", required=True, help="the rebuild to start from (data/real/<id>/rebuild/<tag>)")
    ap.add_argument("--out", required=True, help="where to write the result (data/real/<id>/rebuild/<out>)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src, dst = f"{REAL}/{args.id}/rebuild/{args.tag}", f"{REAL}/{args.id}/rebuild/{args.out}"
    os.makedirs(dst, exist_ok=True)
    config = json.load(open(f"{src}/config.json"))
    report = {}
    rng = np.random.default_rng(args.seed)
    for k in [int(g) for g in args.games.split(",")]:
        name = f"game{k}"
        game = config["games"][name]
        codes = f"{src}/codes_{game.get('spawns', 'neutral')}.bin"
        ports = [p - 1 for p in game["video_ports"]]
        if game["stage"] != "Battlefield":
            shutil.copy(f"{src}/{name}.slp", f"{dst}/{name}.slp")
            print(f"{name}: {game['stage']}, nothing to do (copied)", flush=True)
            continue
        t = video_features(f"{REAL}/{args.id}/{name}/video.mp4", np.load(f"{REAL}/{args.id}/{name}/frame_map.npy"))
        changes, stable = video_changes(t)
        print(f"{name}: background in the video {[(a, b, bg) for a, b, bg in stable]}, changes {changes}", flush=True)
        data = open(f"{src}/{name}.slp", "rb").read()
        last = int(parse_replay(f"{src}/{name}.slp")[0].frame.max())
        work = f"{dst}/{name}_work.slp"

        # 1) the first wait: game start seeds until the tunnel would start at the video's frame
        if changes:
            want_first = changes[0][0] + 123 - 2         # tunnel start = -123 + wait + 2
            best = (None, 10**9)
            for n in range(400):
                seed = int(rng.integers(0, 2**32))
                open(work, "wb").write(with_start_seed(data, seed))
                wait = first_wait(work, codes)
                if abs(wait - want_first) < abs(best[1] - want_first):
                    best = (seed, wait)
                if abs(wait - want_first) <= 5:
                    break
            data = with_start_seed(data, best[0])
            print(f"  first wait {best[1]} (wanted {want_first}) after {n + 1} tries", flush=True)
        open(work, "wb").write(data)

        # 2) each change: the choice and the next wait, seed by seed
        chosen, events = {}, []
        with Engine(work, codes=codes, preroll=True) as e:
            bg = Background(e)
            st = e.state()
            for i, (start, new_bg) in enumerate(changes):
                # play on until the tunnel starts in our game
                while bg.state()[0] != 1:
                    st = e.step()
                ours = st["match_frame"]
                choice = ours + TUNNEL                    # the frame in which the choice is drawn
                while st["match_frame"] < choice - 1:
                    st = e.step()
                seed, st = try_seeds(e, bg, choice, ports, lambda s: s[0] == 2 and s[1] == new_bg, rng)
                if seed is None:
                    raise SystemExit(f"no seed gives background {new_bg} at frame {choice}")
                chosen[choice] = seed
                e.override_seed(choice, seed)
                # the next wait: 200 frames later; make the next tunnel start at the video's frame
                wait_frame = choice + FADE
                while st["match_frame"] < wait_frame - 1:
                    st = e.step()
                if i + 1 < len(changes):
                    want_wait = changes[i + 1][0] - wait_frame - 2
                    want_wait = min(max(want_wait, 2400), 3599)
                    seed, st = try_seeds(e, bg, wait_frame, ports, lambda s: s[0] == 0 and s[3] == want_wait, rng)
                    if seed is None:
                        raise SystemExit(f"no seed gives wait {want_wait} at frame {wait_frame}")
                    chosen[wait_frame] = seed
                    e.override_seed(wait_frame, seed)
                events.append({"video_start": start, "our_start": int(ours), "background": new_bg})
                print(f"  change {i + 1}: tunnel at {ours} (video {start}), background {new_bg}", flush=True)
            while st["match_frame"] < last:
                try:
                    st = e.steps(min(200, last - st["match_frame"]))[-1]
                except RuntimeError:
                    break
            out = blank_player_info(e.record())
        open(f"{dst}/{name}.slp", "wb").write(out)
        os.remove(work)

        # 3) the gameplay must be exactly the same as before
        before, _ = parse_replay(f"{src}/{name}.slp")
        after, _ = parse_replay(f"{dst}/{name}.slp")
        cols = ["action", "x", "y", "percent", "stocks"]
        a = before.set_index(["frame", "port"])[cols].sort_index()
        b = after.set_index(["frame", "port"])[cols].sort_index()
        common = a.index.intersection(b.index)
        same = float((a.loc[common].to_numpy() == b.loc[common].to_numpy()).all(1).mean())
        print(f"  gameplay identical on {same:.4f} of {len(common)} player-frames", flush=True)
        report[name] = {"video_changes": changes, "events": events, "seeds": {str(f): s for f, s in chosen.items()},
                        "gameplay_identical": same}
    json.dump({"args": vars(args), "games": report}, open(f"{dst}/background.json", "w"), indent=1)


if __name__ == "__main__":
    main()
