"""
m4_closed_loop.py - Stage C with the game engine IN THE LOOP ("closed loop").

The open-loop pipeline (scripts/m4_end_to_end.py) guesses every frame's inputs from the video
once and writes them into a .slp. Small input mistakes then add up: the replayed game drifts
away from the video, and Slippi's resync can only hide part of it (it forces positions and
action numbers but not ledge hangs, hitstun, animation timing, stocks).

Here the native engine (src/vid2slp/engine.py) plays the game WHILE we choose the inputs:
  1. The input model (M2) gives, for every frame, a probability for every stick / button
     value, computed from the PERCEIVED states (what the video shows; percep_v3 output).
  2. The engine plays the game a few frames at a time ("chunk", --chunk frames). At the start
     of each chunk it saves its state (capture). Then for each player it tries several
     candidate input sequences for the chunk (restore, play chunk + a short look-ahead,
     measure how far the simulated states are from the perceived states) and keeps the best.
     Candidates: the model's most likely inputs; the same shifted 1-2 frames earlier/later
     (timing mistakes); and samples from the model's probabilities (other stick values,
     other button presses).
  3. The chosen inputs are played for real, and the next chunk starts from there.
  4. At the end the engine's recording IS the output .slp: its stored states are the
     engine's own, so it replays the same with or without resync.

Information the rebuilt game must not borrow from the original game (honesty rules):
  - inputs: model only (the template .slp's inputs are overwritten from frame 0);
  - states: the engine's own (resync is OFF, the original's stored states are never used);
  - per-frame random seeds: NOT forced (Engine.frame_seeds(False)); the RNG runs on from the
    game start seed. Still borrowed for now (known shortcut, also in m4_end_to_end.py):
    stage, characters, costumes, ports, the game start seed, frames before 2 (countdown).
  - Gecko codes: our fixed UCF 0.84 list (data/m4/ucf084_gecko.bin), not the original's.

Scores: same as m4_end_to_end.py (src/vid2slp/assemble.py replay_scores), frames 0..--end, on
the same validation games (--same-games-as). With --verify N, the first N outputs are also
played in Slippi Playback (resync off) to check they replay exactly like the engine played them.

Run (in tmux):
    .venv/bin/python scripts/m4_closed_loop.py --same-games-as v2_perc_m2 --tag cl_v0
    .venv/bin/python scripts/m4_closed_loop.py --same-games-as v2_perc_m2 --open-loop --tag ol_v0
Output: data/m4/closed_loop/<tag>/ (results.csv, config.json, <game>.slp)
"""

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from vid2slp.assemble import replay_scores
from vid2slp.engine import Engine
from vid2slp.id_data import BUTTONS, FEATURE_SETS, arrays_from_states
from vid2slp.id_model import build_from_config, predict_probs
from vid2slp.id_to_slp import processed_sequence
from vid2slp.playback import regenerate
from vid2slp.slp_parse import parse_replay
from vid2slp.slp_write import write_inputs

OUT = "data/m4/closed_loop"
CODES = "data/m4/ucf084_gecko.bin"
HISTORY = 600   # frames of committed inputs re-processed before a chunk (X/Y and L/R alternation)


# ---- inputs ------------------------------------------------------------------------------

def most_likely(probs):
    """Model probabilities -> (tgt [T, 2, 5], btn [T, 2]) most likely values (like decode())."""
    tgt = np.concatenate([probs["axes"].argmax(-1), probs["trigger"].argmax(-1)[..., None]], axis=-1)
    bits = probs["buttons"] > 0.5
    btn = sum(bits[..., i].astype(np.uint8) << i for i in range(len(BUTTONS)))
    return tgt.astype(np.int64), btn.astype(np.uint8)


def candidates(probs, base_tgt, base_btn, player, t0, n, samples, rng):
    """Candidate inputs of one player for frames t0 .. t0+n-1: list of (name, tgt [n, 5], btn [n])."""
    T = len(base_tgt)
    rows = np.arange(t0, t0 + n)
    out = [("base", base_tgt[rows, player], base_btn[rows, player])]
    # timing: the whole chunk's inputs taken from d frames later (d > 0: everything happens
    # d frames EARLIER) or earlier (d < 0)
    for d in (1, -1, 2, -2):
        src = np.clip(rows + d, 0, T - 1)
        out.append((f"shift{d:+d}", base_tgt[src, player], base_btn[src, player]))
    # samples: one random number per input per chunk, used as a QUANTILE on every frame.
    # Stick values are ordered (-80..80), so a low number gives "a bit more left/down" on every
    # frame of the chunk instead of independent per-frame noise. A button is pressed on the
    # frames where its probability is above its number.
    axes = probs["axes"][rows, player].astype(np.float32)        # [n, 4, 161]
    trig = probs["trigger"][rows, player].astype(np.float32)     # [n, 141]
    bprob = probs["buttons"][rows, player].astype(np.float32)    # [n, 8]
    for k in range(samples):
        kind = ("stick", "buttons", "all")[k % 3]
        tgt = base_tgt[rows, player].copy()
        btn = base_btn[rows, player].copy()
        if kind in ("stick", "all"):
            u = rng.random(5)
            for a in range(4):
                cdf = np.cumsum(axes[:, a], -1)
                tgt[:, a] = (cdf < u[a] * cdf[:, -1:]).sum(-1)
            cdf = np.cumsum(trig, -1)
            tgt[:, 4] = (cdf < u[4] * cdf[:, -1:]).sum(-1)
            tgt = np.clip(tgt, 0, [160, 160, 160, 160, 140])
        if kind in ("buttons", "all"):
            u = rng.random(len(BUTTONS))
            bits = bprob > u[None, :]
            btn = sum(bits[:, i].astype(np.uint8) << i for i in range(len(BUTTONS))).astype(np.uint8)
        out.append((f"sample_{kind}", tgt, btn))
    return out


def override_rows(cur_tgt, cur_btn, t0, end, ports):
    """Engine override rows for frames t0 .. end-1, both players, from the input arrays.

    The arrays are turned into .slp input values with processed_sequence (which alternates
    X/Y and L/R for new presses, so it is run over HISTORY frames before t0 as well)."""
    a = max(0, t0 - HISTORY)
    rows = []
    for i, port in enumerate(ports):
        seq = processed_sequence(cur_tgt[a:end, i], cur_btn[a:end, i])
        for f in range(t0, end):
            v = seq[f - a]
            raw = cur_tgt[f, i, :4] - 80        # raw stick bytes, consistent with the processed values
            rows.append((f, int(port), v["stick_x"], v["stick_y"], v["cstick_x"], v["cstick_y"],
                         v["trigger"], v["buttons"], raw[0], raw[1], raw[2], raw[3]))
    return rows


# ---- distance between a simulated stretch and the perceived states -------------------------

def cost(states, t0, per, ports):
    """Sum over frames t0.. and both players of how different the engine's states are from the
    perceived ones. Units: 1 = one frame with the wrong action (10 world units of position
    error, or 10 % damage error, cost as much; a wrong stock count costs 3; wrong facing 0.5).
    Frames where perception did not see the character count 0.2."""
    total = 0.0
    for k, st in enumerate(states):
        f = t0 + k
        if f >= len(per["action"]):
            break
        for i, port in enumerate(ports):
            p = st["players"][port]
            c = float(p["action"] != per["action"][f, i])
            c += min(abs(p["x"] - per["x"][f, i]) + abs(p["y"] - per["y"][f, i]), 30.0) / 10.0
            c += 0.5 * float((p["facing"] > 0) != (per["facing"][f, i] > 0))
            c += min(abs(p["percent"] - per["percent"][f, i]), 20.0) / 10.0
            c += 3.0 * float(p["stocks"] != per["stocks"][f, i])
            total += c * (1.0 if per["seen"][f, i] else 0.2)
    return total


def in_match(state):
    return state["scene_major"] == 0x0E and state["scene"] == 1


# ---- one game ----------------------------------------------------------------------------

def run_game(job):
    """Closed-loop (or open-loop) reconstruction of one game in the engine."""
    sha, path, probs, per, ports, end, args, out_dir = job
    t_start = time.time()
    rng = np.random.default_rng(args["seed"])
    base_tgt, base_btn = most_likely(probs)
    cur_tgt, cur_btn = base_tgt.copy(), base_btn.copy()
    T = len(base_tgt)

    # Starting file: the original as a template (settings, stage, characters, start seed) with
    # the model's most likely inputs from frame 0 on, so even frames played before we take
    # control (the engine hands over at frame 1) use model inputs.
    seqs = [processed_sequence(base_tgt[:, i], base_btn[:, i]) for i in range(2)]
    col = {int(p): i for i, p in enumerate(ports)}

    def values(frame, port):
        if frame < 0 or frame >= T:
            return None
        v = dict(seqs[col[int(port)]][frame])
        v["raw_stick_x"] = int(base_tgt[frame, col[int(port)], 0] - 80)
        v["raw_stick_y"] = int(base_tgt[frame, col[int(port)], 1] - 80)
        return v
    base_slp = f"{out_dir}/{sha[:12]}_start.slp"
    write_inputs(path, base_slp, values)

    H, L = args["chunk"], args["lookahead"]
    choices = {}
    gain = []
    with Engine(base_slp, codes=CODES) as e:
        e.frame_seeds(False)
        st = e.state()
        while in_match(st) and st["match_frame"] < end:
            t0 = st["match_frame"] + 1           # the next frame the engine plays
            n = min(H, end - st["match_frame"])
            span_end = min(t0 + n + L, T)
            if not args["open_loop"]:
                e.capture()
                for i in range(2):               # player by player (the other keeps its best so far)
                    scored = []
                    for name, tgt, btn in candidates(probs, base_tgt, base_btn, i, t0, n, args["samples"], rng):
                        trial_tgt, trial_btn = cur_tgt.copy(), cur_btn.copy()
                        trial_tgt[t0:t0 + n, i], trial_btn[t0:t0 + n, i] = tgt, btn
                        e.restore()
                        e.override_many(override_rows(trial_tgt, trial_btn, t0, span_end, ports))
                        states = e.steps(span_end - t0)
                        scored.append((cost(states, t0, per, ports), name, tgt, btn))
                    best = min(scored, key=lambda s: s[0])
                    gain.append(scored[0][0] - best[0])   # scored[0] is "base"
                    choices[best[1]] = choices.get(best[1], 0) + 1
                    cur_tgt[t0:t0 + n, i], cur_btn[t0:t0 + n, i] = best[2], best[3]
                e.restore()
            e.override_many(override_rows(cur_tgt, cur_btn, t0, span_end, ports))
            try:
                st = e.steps(n)[-1]
            except RuntimeError:                 # the engine stopped (the replay file ended)
                break
        data = e.record()
    out = f"{out_dir}/{sha[:12]}.slp"
    open(out, "wb").write(data)
    os.remove(base_slp)
    return {"game": sha[:12], "seconds": round(time.time() - t_start, 1), "last_frame": int(st["match_frame"]),
            "choices": json.dumps(choices), "mean_gain": float(np.mean(gain)) if gain else 0.0}


# ---- main --------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perception", default="percep_v3")
    ap.add_argument("--m2", default="perc_v1")
    ap.add_argument("--same-games-as", default="v2_perc_m2", help="games of an m4_end_to_end run (its --tag)")
    ap.add_argument("--end", type=int, default=3600, help="last frame rebuilt and compared")
    ap.add_argument("--chunk", type=int, default=4, help="frames chosen at a time")
    ap.add_argument("--lookahead", type=int, default=10, help="extra frames played to judge a candidate")
    ap.add_argument("--samples", type=int, default=8, help="sampled candidates per player and chunk")
    ap.add_argument("--open-loop", action="store_true", help="no search: the model's most likely inputs")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--verify", type=int, default=2, help="replay this many outputs in Slippi Playback")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-games", type=int, default=0, help="only the first N games (quick tests)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    out_dir = f"{OUT}/{args.tag}"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)

    games = json.load(open(f"data/m4/end_to_end/{args.same_games_as}/config.json"))["games"]
    if args.max_games:
        games = games[:args.max_games]
    splits = pd.read_csv("data/index/splits.csv").set_index("sha256")
    stage_ids = pd.read_csv("data/index/slp_index.csv").drop_duplicates("sha256").set_index("sha256").stage_id

    # The input model's probabilities for every game (GPU, in this process).
    device = "cuda:0"
    ck = torch.load(f"data/m2/runs/{args.m2}/best.pt", map_location=device)
    cfg = ck["config"]
    version = open(f"{cfg['data']}/features.txt").read().strip()
    model = build_from_config(cfg, len(FEATURE_SETS[version])).to(device)
    model.load_state_dict(ck["model"])
    jobs, tables = [], {}
    for sha in games:
        path = splits.loc[sha, "path"]
        perceived = pd.read_parquet(f"data/m3/pred/{args.perception}/{sha}.parquet")
        perceived = perceived[perceived.frame >= 0].sort_values(["frame", "port"])
        ports = sorted(int(p) for p in perceived.port.unique())
        num, cat = arrays_from_states(perceived, int(stage_ids[sha]), version)
        probs = predict_probs(model, num, cat, device, cfg["window"])
        T = len(num)
        per = {c: np.nan_to_num(perceived[c].to_numpy().astype(np.float32)).reshape(T, 2) for c in ("action", "x", "y", "facing", "percent", "stocks", "seen")}
        end = min(args.end, T - args.lookahead - 5)
        # only the frames needed (keeps the data sent to the worker processes small)
        keep = min(T, end + args.lookahead + 10)
        probs = {k: v[:keep] for k, v in probs.items()}
        jobs.append((sha, path, probs, per, ports, end, vars(args), out_dir))
        tables[sha] = perceived
    del model
    torch.cuda.empty_cache()
    print(f"{len(jobs)} games; M2 probabilities done; starting the engine search", flush=True)

    with ProcessPoolExecutor(args.workers) as pool:
        runs = list(pool.map(run_game, jobs))

    rows = []
    for (sha, path, *_), run in zip(jobs, runs):
        orig, _ = parse_replay(path)
        regen, _ = parse_replay(f"{out_dir}/{sha[:12]}.slp")
        end = min(args.end, run["last_frame"])
        rows.append({**run, **replay_scores(orig, regen, tables[sha], end)})
    res = pd.DataFrame(rows)

    # Does Slippi Playback replay the output exactly like the engine played it?
    for sha, *_ in jobs[:args.verify]:
        slp = f"{out_dir}/{sha[:12]}.slp"
        mine, _ = parse_replay(slp)
        last = int(mine.frame.max())
        regen, _ = regenerate(slp, f"{out_dir}/{sha[:12]}_playback", -123, last - 1, resync=False)
        theirs, _ = parse_replay(regen)
        cols = ["action", "x", "y", "percent", "stocks"]
        a = mine[mine.frame <= last - 1].set_index(["frame", "port"])[cols].sort_index()
        b = theirs[theirs.frame <= last - 1].set_index(["frame", "port"])[cols].sort_index()
        common = a.index.intersection(b.index)
        same = (a.loc[common].to_numpy() == b.loc[common].to_numpy()).all(1).mean()
        res.loc[res.game == sha[:12], "playback_identical"] = float(same)
        shutil.rmtree(f"{out_dir}/{sha[:12]}_playback", ignore_errors=True)

    res.to_csv(f"{out_dir}/results.csv", index=False)
    pd.set_option("display.width", 250)
    show = ["game", "seconds", "last_frame", "mean_gain", "action_match", "action_anim_match", "airborne_match",
            "x_err", "percent_err", "stocks_match", "follows_stored_action"]
    if "playback_identical" in res:
        show.append("playback_identical")
    print(res[show].round(3).to_string(index=False))
    print("\nmean:", res[["action_match", "action_anim_match", "airborne_match", "x_err", "y_err", "percent_err",
                         "stocks_match", "follows_stored_action"]].mean().round(3).to_dict())
    total = {}
    for c in res.choices:
        for k, v in json.loads(c).items():
            total[k] = total.get(k, 0) + v
    print("chosen candidates:", total)
    json.dump({**vars(args), "m2_step": ck["step"], "games": games, "codes": CODES}, open(f"{out_dir}/config.json", "w"), indent=1)
    print(f"wrote {out_dir}/results.csv")


if __name__ == "__main__":
    main()
