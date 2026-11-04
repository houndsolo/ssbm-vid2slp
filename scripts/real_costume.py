"""
real_costume.py - which costume (colour) each player wears in a real video, from the stock
icons drawn above their damage.

Why: our starting files are other people's netplay games, so the rebuilt replay showed THEIR
costumes (e.g. a red Fox where the video has a blue one). The costume is one byte per player in
the replay's game start event (offset 3 of the player block); it does not change the gameplay.

How:
  1. Reference icons (made once, data/m3/costume_ref/icons.npz): short Slippi Playback renders
     of a game start with Fox / Falco in each of their 4 costumes on each of the 4 ports. For each
     character, port and costume we keep the stock icon area (a small box left of the "%" sign),
     as the median over the countdown frames.
     Only the pixels that differ between the 4 costumes count (the "mask"): the coloured parts of
     the icons. Outlines and whatever stage art sits behind the icons are the same in all 4.
  2. In the cut game video, the same box for the player's port, as the median over the first
     frames where perception says the player still has 4 stocks (the median removes characters
     and effects passing behind the HUD).
  3. The costume whose reference is closest on the mask pixels wins. The margin (2nd best
     distance - best) says how sure it is.

Run (the reference renders take a few minutes the first time):
    .venv/bin/python scripts/real_costume.py --id 5njRxIAgS-o --games 1,2,3
    .venv/bin/python scripts/real_costume.py --id 5njRxIAgS-o --games 1,2,3 --apply v2   # into rebuilt replays
Used by scripts/real_rebuild.py (costumes()).
"""

import argparse
import os
import shutil

import av
import numpy as np
import pandas as pd

from vid2slp.playback import regenerate
from vid2slp.slp_ports import move_ports

REF = "data/m3/costume_ref"
# Centre column of each port's damage "%" sign in a 584 x 480 game picture, measured on our
# renders with players on every port pair (data/m3/renders_ports, 2026-11-02).
PERCENT_X = {1: 133, 2: 273, 3: 409, 4: 549}
TEMPLATE = "data/recordings/examples/Game_20260116T171327.slp"   # a Fox ditto on Battlefield
CHARACTERS = {2: "Fox", 20: "Falco"}      # external character ids (as in the replay)
# the stock icon box relative to a port's "%" sign centre, in the 584 x 480 game picture
BOX_Y = (358, 386)
BOX_X = (-118, -14)
# renders: (ports, costumes of the two players) so that every costume appears on every port
RENDERS = [((1, 3), (0, 1)), ((1, 3), (1, 0)), ((1, 3), (2, 3)), ((1, 3), (3, 2)),
           ((2, 4), (0, 1)), ((2, 4), (1, 0)), ((2, 4), (2, 3)), ((2, 4), (3, 2))]


def set_players(data, players):
    """.slp bytes with player blocks changed: players = {index 0-3: (character, costume)}."""
    out = bytearray(data)
    i = data.find(b"raw[$U#l") + 12
    pos = i + 1 + data[i + 1]
    for p, (char, costume) in players.items():
        out[pos + 0x65 + 0x24 * p] = char
        out[pos + 0x65 + 0x24 * p + 3] = costume
    return bytes(out)


def set_costumes(data, costumes):
    """.slp bytes with the costume byte of some players set: costumes = {index 0-3: costume}."""
    out = bytearray(data)
    i = data.find(b"raw[$U#l") + 12
    pos = i + 1 + data[i + 1]
    for p, costume in costumes.items():
        out[pos + 0x65 + 0x24 * p + 3] = costume
    return bytes(out)


def box(frame, port):
    """The stock icon box of `port` in a 584 x 480 RGB picture."""
    x = PERCENT_X[port]
    return frame[BOX_Y[0]:BOX_Y[1], x + BOX_X[0]:x + BOX_X[1]].astype(np.float32)


def make_references():
    """Render every (character, costume, port) once and keep the icon boxes."""
    os.makedirs(REF, exist_ok=True)
    work = "data/m3/work/costume_ref"
    icons = {}
    for char in CHARACTERS:
        for ports, costumes in RENDERS:
            data = set_players(open(TEMPLATE, "rb").read(), {0: (char, costumes[0]), 1: (char, costumes[1])})
            slp = f"{REF}/tmp.slp"
            open(slp, "wb").write(move_ports(data, ports))
            shutil.rmtree(work, ignore_errors=True)
            # the first 80 frames of the countdown: the HUD is up, nobody has lost a stock
            regenerate(slp, work, -123, -44, gpu=0, dump_video={"codec": "ffv1", "fmt": "avi", "kbps": 25000})
            frames = [f.to_ndarray(format="rgb24") for f in
                      av.open(f"{work}/user/Dump/Frames/framedump0.avi").decode(video=0)]
            frames = frames[20:]                      # skip the fade in
            for port, costume in zip(ports, costumes):
                icons[(char, costume, port)] = np.median([box(f, port) for f in frames], axis=0)
            print(CHARACTERS[char], ports, costumes, len(frames), "frames", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    os.remove(f"{REF}/tmp.slp")
    np.savez_compressed(f"{REF}/icons.npz", keys=np.array(list(icons.keys())), boxes=np.stack(list(icons.values())))


def references():
    if not os.path.exists(f"{REF}/icons.npz"):
        make_references()
    z = np.load(f"{REF}/icons.npz")
    return {tuple(int(v) for v in k): b for k, b in zip(z["keys"], z["boxes"])}


def video_box(video, frames, port):
    """Median icon box of `port` over the given video frame numbers."""
    wanted, crops = set(frames), []
    container = av.open(video)
    for i, frame in enumerate(container.decode(video=0)):
        if i > max(wanted):
            break
        if i in wanted:
            crops.append(box(frame.to_ndarray(format="rgb24"), port))
    container.close()
    return np.median(crops, axis=0)


def costume_of(refs, char, port, seen_box):
    """(best costume, margin, distances) of one player."""
    ref = np.stack([refs[(char, c, port)] for c in range(4)])
    mask = ref.std(0).max(-1) > 20               # pixels where the 4 costumes differ
    dist = [float(np.abs(seen_box[mask] - ref[c][mask]).mean()) for c in range(4)]
    order = np.argsort(dist)
    return int(order[0]), dist[order[1]] - dist[order[0]], [round(d, 1) for d in dist]


def costumes(video, frame_map, perceived, chars, ports, n=200):
    """Costume of each player in a cut game video. perceived: perception rows with Slippi frames
    (port = slot 0/1, consistent stocks); chars: external character ids per slot; ports: the
    video's ports. Returns [(costume, margin, distances)] per slot."""
    refs = references()
    out = []
    for slot, (char, port) in enumerate(zip(chars, ports)):
        full = perceived[(perceived.port == slot) & (perceived.stocks == 4) & (perceived.frame >= 0)].frame
        game_frames = set(full.to_numpy().tolist())
        video_frames = [i for i, g in enumerate(frame_map) if g in game_frames]
        step = max(1, len(video_frames) // n)
        out.append(costume_of(refs, char, port, video_box(video, video_frames[::step][:n], port)))
    return out


def main():
    from real_rebuild import consistent, video_ports       # here: real_rebuild imports this file
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--games", default="1")
    ap.add_argument("--perception", default="percep_v6soft")
    ap.add_argument("--apply", default=None,
                    help="also write the costumes into the rebuilt replays data/real/<id>/rebuild/<APPLY>/game<k>.slp "
                         "(in place; looks only, the gameplay stays the same)")
    args = ap.parse_args()
    to_external = {1: 2, 22: 20}
    for k in [int(g) for g in args.games.split(",")]:
        d = f"data/real/{args.id}/game{k}"
        seen = consistent(pd.read_parquet(f"data/real/{args.id}/pred_{args.perception}_countdown/game{k}.parquet"))
        chars = [to_external[int(seen[seen.port == s].character.iloc[0])] for s in (0, 1)]
        ports, _ = video_ports(f"{d}/video.mp4")
        res = costumes(f"{d}/video.mp4", np.load(f"{d}/frame_map.npy"), seen, chars, ports)
        print(f"game {k}: ports {ports}, " + ", ".join(
            f"P{p} {CHARACTERS[c]} costume {r[0]} (margin {r[1]:.1f}, distances {r[2]})"
            for p, c, r in zip(ports, chars, res)), flush=True)
        if args.apply:
            slp = f"data/real/{args.id}/rebuild/{args.apply}/game{k}.slp"
            data = set_costumes(open(slp, "rb").read(), {p - 1: r[0] for p, r in zip(ports, res)})
            open(slp, "wb").write(data)
            print(f"  wrote the costumes into {slp}", flush=True)


if __name__ == "__main__":
    main()
