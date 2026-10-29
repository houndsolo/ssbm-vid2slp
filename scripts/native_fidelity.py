"""
native_fidelity.py - does the native engine's replay playback reproduce real games exactly?

For each chosen game: play the original .slp in the native engine (src/vid2slp/engine.py,
replay playback added by vid2slp; resync OFF, so the game is simulated only from the stored
inputs and seeds, exactly like Slippi Playback without resync), record the played game, and
compare it with the original frame by frame (action, x, y, percent, stocks of both players).

A game "matches" if every compared value is identical. If not, we report the first frame
where anything differs and the share of player-frames whose action still matches.

Run:
    .venv/bin/python scripts/native_fidelity.py --games 20 --split validation
Output: data/native/fidelity/<tag>.csv
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from vid2slp.engine import Engine
from vid2slp.matchups import wanted_games
from vid2slp.slp_codes import ucf_version
from vid2slp.slp_parse import parse_replay

OUT = "data/native/fidelity"


def play(path, resync=False):
    """Play a whole replay in the engine; return (recorded .slp bytes, seconds, frames stepped)."""
    t0 = time.time()
    steps = 0
    with Engine(path, resync=resync) as e:
        while True:
            s = e.state()
            # the match is over when the replay scene ends (or after a generous cap)
            if not (s["scene_major"] == 0x0E and s["scene"] == 1) or steps > 30000:
                break
            try:
                e.step()
            except RuntimeError:          # the engine exits when the replay has finished
                break
            steps += 1
        data = e.record()
    return data, time.time() - t0, steps


def compare(orig, regen):
    cols = ["action", "x", "y", "percent", "stocks"]
    o = orig[orig.frame >= -123].set_index(["frame", "port"])[cols].sort_index()
    r = regen[regen.frame >= -123].set_index(["frame", "port"])[cols].sort_index()
    common = o.index.intersection(r.index)
    o, r = o.loc[common], r.loc[common]
    same = ((o.action.to_numpy() == r.action.to_numpy()) & (o.x.to_numpy() == r.x.to_numpy())
            & (o.y.to_numpy() == r.y.to_numpy()) & (o.percent.to_numpy() == r.percent.to_numpy())
            & (o.stocks.to_numpy() == r.stocks.to_numpy()))
    frames = np.array([f for f, p in common])
    first_diff = int(frames[~same].min()) if (~same).any() else None
    late = ~same & (frames > -100)       # ignore the known 1-ULP pre-match differences
    first_late = int(frames[late].min()) if late.any() else None
    return {"orig_frames": int(orig.frame.max()), "compared_frames": int(len(set(frames))),
            "exact": float(same.mean()), "action_match": float((o.action.to_numpy() == r.action.to_numpy()).mean()),
            "first_diff_frame": first_diff, "first_diff_after_-100": first_late}


def one(job):
    sha, path = job
    try:
        data, seconds, steps = play(path)
        out = f"{OUT}/slp/{sha[:12]}.slp"
        open(out, "wb").write(data)
        orig, _ = parse_replay(path)
        regen, _ = parse_replay(out)
        return {"game": sha[:12], "ucf": ucf_version(path), "seconds": round(seconds, 1), "steps": steps,
                **compare(orig, regen)}
    except Exception as e:
        return {"game": sha[:12], "error": f"{type(e).__name__}: {str(e)[:300]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="replay_v1")
    ap.add_argument("--ucf", default="", help="only games with this UCF version (0.84 / old)")
    ap.add_argument("--fox", action="store_true", help="only Fox-Fox / Fox-Falco games")
    ap.add_argument("--game-list", default="", help="file with sha256 prefixes, one per line (instead of a random sample)")
    args = ap.parse_args()
    os.makedirs(f"{OUT}/slp", exist_ok=True)
    splits = pd.read_csv("data/index/splits.csv")
    if args.game_list:
        wanted = open(args.game_list).read().split()
        games = splits[splits.sha256.apply(lambda s: any(s.startswith(w) for w in wanted))]
    else:
        games = splits[splits.split == args.split]
        if args.fox:
            games = games[games.sha256.isin(wanted_games())]
        if args.ucf:
            games = games[[ucf_version(p) == args.ucf for p in games.path]]
        games = games.sample(min(args.games, len(games)), random_state=args.seed)
    jobs = list(zip(games.sha256, games.path))
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one, jobs))
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/{args.tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))
    ok = res[res["error"].isna()] if "error" in res else res
    # Known tiny differences before the match (1 float ULP at frames -118/-113, listed in
    # Melee Unlocked's docs/replay-validation.md) are not counted as divergence here.
    diverged = ok["first_diff_after_-100"].notna()
    print(f"\n{len(ok)}/{len(res)} games ran; same from frame -100 to the end: {(~diverged).sum()}; "
          f"mean exact share {ok.exact.mean():.4f}; mean action match {ok.action_match.mean():.4f}")


if __name__ == "__main__":
    main()
