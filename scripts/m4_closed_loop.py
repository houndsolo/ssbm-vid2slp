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
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from vid2slp.assemble import replay_scores
from vid2slp.engine import Engine
from vid2slp.id_data import BUTTONS, FEATURE_SETS, arrays_from_states, game_arrays, stick_from_raw
from vid2slp.id_model import build_from_config, predict_probs
from vid2slp.id_to_slp import processed_sequence
from vid2slp.playback import regenerate
from vid2slp.slp_parse import parse_replay
from vid2slp.slp_write import write_inputs

OUT = "data/m4/closed_loop"
CODES = "data/m4/ucf084_gecko.bin"
AIRBORNE = "data/m4/airborne_by_action.json"   # scripts/m4_airborne_table.py
HISTORY = 600   # frames of committed inputs re-processed before a chunk (X/Y and L/R alternation)


# ---- inputs ------------------------------------------------------------------------------

def decode_bits(bprob, threshold=0.5):
    """Button probabilities [..., 8] -> bitmask [...] of the buttons above the threshold."""
    bits = bprob > threshold
    return sum(bits[..., i].astype(np.uint8) << i for i in range(len(BUTTONS))).astype(np.uint8)


def match_raw(tgt, raw):
    """Make the processed main stick (tgt[..., 0:2], 0..160) agree with the raw bytes
    (raw[..., 0:2], -128..127), as the game computes it (stick_from_raw)."""
    tgt = tgt.copy()
    px, py = stick_from_raw(raw[..., 0], raw[..., 1])
    tgt[..., 0], tgt[..., 1] = px + 80, py + 80
    return tgt


def most_likely(probs):
    """Model probabilities -> (tgt [T, 2, 5], btn [T, 2], raw [T, 2, 2]) most likely values.
    With raw heads, the raw stick bytes are the model's most likely raw values and the processed
    main stick follows from them; otherwise raw = processed value * 80."""
    tgt = np.concatenate([probs["axes"].argmax(-1), probs["trigger"].argmax(-1)[..., None]], axis=-1).astype(np.int64)
    btn = decode_bits(probs["buttons"])
    if "raw" in probs:
        raw = probs["raw"].argmax(-1).astype(np.int64) - 128
        return match_raw(tgt, raw), btn, raw
    return tgt, btn, tgt[..., :2] - 80


def quantile_sample(p, tgt, btn, raw, kind, rng):
    """One sampled input sequence for a chunk. One random number per input per chunk is used as
    a QUANTILE on every frame. Stick values are ordered (-80..80, raw -128..127), so a low number
    gives "a bit more left/down" on every frame of the chunk instead of independent per-frame
    noise. A button is pressed on the frames where its probability is above its number.
    p: probabilities for the chunk {"axes" [n, 4, 161], "trigger" [n, 141], "buttons" [n, 8],
    optional "raw" [n, 2, 256]}; tgt [n, 5], btn [n], raw [n, 2]: the values kept for the parts
    not sampled (kind = "stick", "buttons" or "all"). Returns (tgt, btn, raw)."""
    tgt, btn, raw = tgt.copy(), btn.copy(), raw.copy()

    def draw(prob, u):
        cdf = np.cumsum(prob.astype(np.float32), -1)
        return (cdf < u * cdf[:, -1:]).sum(-1)

    if kind in ("stick", "all"):
        u = rng.random(5)
        for a in range(4):
            tgt[:, a] = np.clip(draw(p["axes"][:, a], u[a]), 0, 160)
        tgt[:, 4] = np.clip(draw(p["trigger"], u[4]), 0, 140)
        if "raw" in p:                    # main stick from sampled raw bytes (same quantiles)
            raw = np.stack([draw(p["raw"][:, 0], u[0]), draw(p["raw"][:, 1], u[1])], -1).clip(0, 255) - 128
            tgt = match_raw(tgt, raw)
        else:
            raw = tgt[:, :2] - 80
    if kind in ("buttons", "all"):
        u = rng.random(len(BUTTONS))
        bits = p["buttons"] > u[None, :]
        btn = sum(bits[:, i].astype(np.uint8) << i for i in range(len(BUTTONS))).astype(np.uint8)
    return tgt, btn, raw


def candidates(probs, base, player, t0, n, samples, rng, oracle=None, replan=None, replan_samples=0):
    """Candidate inputs of one player for frames t0 .. t0+n-1.
    base: (tgt [T, 2, 5], btn [T, 2], raw [T, 2, 2]) the model's most likely inputs.
    replan: the model's probabilities for these frames computed from the ENGINE's recent states
    (--replan), {"axes" [n, 4, 161], "trigger" [n, 141], "buttons" [n, 8], maybe "raw" [n, 2, 256]}.
    oracle: (true tgt, true btn, true raw) of the original game, DIAGNOSTIC ONLY (--oracle).
    Returns (name, tgt [n, 5], btn [n], raw [n, 2]) entries; raw = the raw main-stick bytes."""
    base_tgt, base_btn, base_raw = base
    T = len(base_tgt)
    rows = np.arange(t0, t0 + n)
    out = [("base", base_tgt[rows, player], base_btn[rows, player], base_raw[rows, player])]
    if oracle is not None:
        out.append(("TRUE", oracle[0][rows, player].astype(np.int64), oracle[1][rows, player], oracle[2][rows, player]))
    # timing: the whole chunk's inputs taken from d frames later (d > 0: everything happens
    # d frames EARLIER) or earlier (d < 0)
    for d in (1, -1, 2, -2):
        src = np.clip(rows + d, 0, T - 1)
        out.append((f"shift{d:+d}", base_tgt[src, player], base_btn[src, player], base_raw[src, player]))
    # samples from the model's probabilities (see quantile_sample)
    mine = {k: v[rows, player] for k, v in probs.items()}
    for k in range(samples):
        kind = ("stick", "buttons", "all")[k % 3]
        out.append((f"sample_{kind}", *quantile_sample(mine, base_tgt[rows, player], base_btn[rows, player],
                                                         base_raw[rows, player], kind, rng)))
    if replan is not None:
        r_tgt, r_btn, r_raw = most_likely(replan)
        out.append(("replan", r_tgt, r_btn, r_raw))
        for k in range(replan_samples):
            kind = ("stick", "buttons", "all")[k % 3]
            out.append((f"replan_sample_{kind}", *quantile_sample(replan, r_tgt, r_btn, r_raw, kind, rng)))
    return out


class Replanner:
    """The input model run on a window whose PAST is the engine's own states (what really
    happened in the rebuilt game so far) and whose FUTURE is the perceived states (what the
    video shows). When the rebuilt game has drifted, the inputs it suggests start from where the
    characters really are. Loaded once per worker process (needs the GPU)."""

    def __init__(self, m2, perceived, stage, device="cuda:0"):
        ck = torch.load(f"data/m2/runs/{m2}/best.pt", map_location=device)
        self.cfg = ck["config"]
        self.version = open(f"{self.cfg['data']}/features.txt").read().strip()
        self.model = build_from_config(self.cfg, len(FEATURE_SETS[self.version])).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.device = device
        self.stage = stage
        self.airborne = json.load(open(AIRBORNE))
        # perceived table as arrays [T, 2] (character / shield are not in the engine's state line)
        p = perceived.sort_values(["frame", "port"])
        self.T = len(p) // 2
        self.per = {c: p[c].to_numpy().reshape(self.T, 2).copy() for c in
                    ("character", "action", "x", "y", "facing", "percent", "stocks", "shield", "airborne")}
        self.sim = {c: self.per[c].copy() for c in ("action", "x", "y", "facing", "percent", "stocks", "airborne")}

    def record(self, frame, state, ports):
        """Remember the engine's state after `frame`."""
        if frame >= self.T:
            return
        for i, port in enumerate(ports):
            pl = state["players"][port]
            for c in ("action", "x", "y", "facing", "percent", "stocks"):
                self.sim[c][frame, i] = pl[c]
            ch = int(self.per["character"][frame, i])
            self.sim["airborne"][frame, i] = self.airborne.get(f"{ch},{pl['action']}", 0)

    @torch.no_grad()
    def probs(self, t0, n):
        """Probabilities for frames t0 .. t0+n-1, both players: {"axes" [n, 2, 4, 161], ...}."""
        window = self.cfg["window"]
        ws = max(0, min(t0 - (3 * window) // 4, self.T - window))
        rows = np.arange(ws, ws + window)
        past = rows < t0
        cols = {}
        for c in self.per:
            v = self.per[c][rows].copy()
            if c in self.sim:
                v[past] = self.sim[c][rows][past]
            cols[c] = v.reshape(-1)
        table = pd.DataFrame({"frame": np.repeat(rows - ws, 2), "port": np.tile([0, 1], window), **cols})
        num, cat = arrays_from_states(table, self.stage, self.version)
        num = torch.from_numpy(num).to(self.device)
        cat = torch.from_numpy(cat).to(self.device)
        # both players as "self" in one batch (the model expects self first)
        num = torch.stack([num, num[:, [1, 0]]])
        cat = torch.stack([cat, cat[:, [1, 0]]])
        with torch.autocast("cuda", dtype=torch.float16):
            out = self.model(num, cat)
        a, b = t0 - ws, t0 - ws + n
        axes = torch.stack([o.float().softmax(-1) for o in out["axes"]], dim=2)[:, a:b]   # [2, n, 4, 161]
        res = {"axes": axes.permute(1, 0, 2, 3).cpu().numpy(),
               "trigger": out["trigger"].float().softmax(-1)[:, a:b].permute(1, 0, 2).cpu().numpy(),
               "buttons": torch.sigmoid(out["buttons"].float())[:, a:b].permute(1, 0, 2).cpu().numpy()}
        if "raw" in out:
            raw = torch.stack([o.float().softmax(-1) for o in out["raw"]], dim=2)[:, a:b]     # [2, n, 2, 256]
            res["raw"] = raw.permute(1, 0, 2, 3).cpu().numpy()
        return res


def override_rows(cur_tgt, cur_btn, cur_raw, t0, end, ports):
    """Engine override rows for frames t0 .. end-1, both players, from the input arrays.

    The arrays are turned into .slp input values with processed_sequence (which alternates
    X/Y and L/R for new presses, so it is run over HISTORY frames before t0 as well)."""
    a = max(0, t0 - HISTORY)
    rows = []
    for i, port in enumerate(ports):
        seq = processed_sequence(cur_tgt[a:end, i], cur_btn[a:end, i])
        for f in range(t0, end):
            v = seq[f - a]
            raw_c = cur_tgt[f, i, 2:4] - 80      # c-stick raw bytes: the processed value * 80
            rows.append((f, int(port), v["stick_x"], v["stick_y"], v["cstick_x"], v["cstick_y"],
                         v["trigger"], v["buttons"], cur_raw[f, i, 0], cur_raw[f, i, 1], raw_c[0], raw_c[1]))
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
    sha, path, probs, per, ports, end, args, out_dir, oracle, perceived, stage = job
    t_start = time.time()
    rng = np.random.default_rng(args["seed"])
    base_tgt, base_btn, base_raw = most_likely(probs)
    if args["true_inputs"]:                      # DIAGNOSTIC: start from the true inputs
        T0 = len(base_tgt)
        base_tgt, base_btn, base_raw = oracle[0][:T0].astype(np.int64), oracle[1][:T0], oracle[2][:T0]
    cur_raw = base_raw.copy()                    # raw main-stick bytes of every frame (UCF 0.84 reads them)
    cur_tgt, cur_btn = base_tgt.copy(), base_btn.copy()
    T = len(base_tgt)

    # Starting file: the original as a template (settings, stage, characters, start seed) with
    # the model's most likely inputs from frame 0 on, so even frames played before we take
    # control (the engine hands over at frame 1) use model inputs.
    first_tgt, first_btn, first_raw = base_tgt, base_btn, cur_raw
    if oracle is not None:                       # DIAGNOSTIC: frames played before we take control
        T0 = len(base_tgt)
        first_tgt, first_btn, first_raw = oracle[0][:T0].astype(np.int64), oracle[1][:T0], oracle[2][:T0]
    seqs = [processed_sequence(first_tgt[:, i], first_btn[:, i]) for i in range(2)]
    col = {int(p): i for i, p in enumerate(ports)}

    def values(frame, port):
        if frame < 0 or frame >= T:
            return None
        v = dict(seqs[col[int(port)]][frame])
        v["raw_stick_x"] = int(first_raw[frame, col[int(port)], 0])
        v["raw_stick_y"] = int(first_raw[frame, col[int(port)], 1])
        return v
    base_slp = f"{out_dir}/{sha[:12]}_start.slp"
    write_inputs(path, base_slp, values)

    H, L = args["chunk"], args["lookahead"]
    replanner = Replanner(args["m2"], perceived, stage) if args["replan"] else None
    choices = {}
    gain = []
    seeds = {}                                   # frame -> random seed chosen there (--seeds)
    with Engine(base_slp, codes=CODES) as e:
        e.frame_seeds(False)
        st = e.state()
        while in_match(st) and st["match_frame"] < end:
            t0 = st["match_frame"] + 1           # the next frame the engine plays
            n = min(H, end - st["match_frame"])
            span_end = min(t0 + n + L, T)
            long_end = min(t0 + n + args["long_lookahead"], T)
            if not args["open_loop"]:
                e.capture()
                replan = replanner.probs(t0, n) if replanner else None
                def trial(i, tgt, btn, raw, stop):
                    """Cost of playing these inputs for player i in this chunk, judged up to frame stop."""
                    trial_tgt, trial_btn, trial_raw = cur_tgt.copy(), cur_btn.copy(), cur_raw.copy()
                    trial_tgt[t0:t0 + n, i], trial_btn[t0:t0 + n, i], trial_raw[t0:t0 + n, i] = tgt, btn, raw
                    e.restore()
                    e.override_many(override_rows(trial_tgt, trial_btn, trial_raw, t0, stop, ports))
                    return cost(e.steps(stop - t0), t0, per, ports)

                for i in range(2):               # player by player (the other keeps its best so far)
                    mine = {k: v[:, i] for k, v in replan.items()} if replan else None
                    scored = [(trial(i, tgt, btn, raw, span_end), name, tgt, btn, raw) for name, tgt, btn, raw in
                              candidates(probs, (base_tgt, base_btn, base_raw), i, t0, n, args["samples"], rng,
                                         oracle, mine, args["replan_samples"])]
                    if long_end > span_end:
                        # Two stages: the few best candidates are judged again over a LONGER stretch
                        # (some input differences only show up 10-30 frames later; measured
                        # 2026-10-30: look-ahead 30 instead of 10 raised the oracle check 0.81 -> 0.89).
                        finalists = sorted(range(len(scored)), key=lambda k: scored[k][0])[:args["finalists"]]
                        if 0 not in finalists:
                            finalists.append(0)          # always keep "base" as a finalist
                        scored = [(trial(i, *scored[k][2:], long_end), *scored[k][1:]) for k in finalists]
                    best = min(scored, key=lambda s: s[0])
                    gain.append(next(c for c, name, *_ in scored if name == "base") - best[0])
                    choices[best[1]] = choices.get(best[1], 0) + 1
                    cur_tgt[t0:t0 + n, i], cur_btn[t0:t0 + n, i], cur_raw[t0:t0 + n, i] = best[2], best[3], best[4]
                if args["seeds"]:
                    # The random seed is also ours to choose (it decides e.g. which tumble animation a
                    # launched character gets, Dreamland's wind, Fountain of Dreams' platforms). Keep
                    # the game's own seed unless a random new one clearly fits the video better.
                    rows = override_rows(cur_tgt, cur_btn, cur_raw, t0, span_end, ports)
                    e.restore()
                    e.override_many(rows)
                    keep_cost = cost(e.steps(span_end - t0), t0, per, ports)
                    trials = []
                    for seed in rng.integers(0, 2**32, args["seeds"]):
                        e.restore()
                        e.override_many(rows)
                        e.override_seed(t0, int(seed))
                        trials.append((cost(e.steps(span_end - t0), t0, per, ports), int(seed)))
                    best_seed = min(trials)
                    if best_seed[0] < keep_cost - 0.5:
                        seeds[t0] = best_seed[1]
                        choices["seed"] = choices.get("seed", 0) + 1
                e.restore()
                if t0 in seeds:
                    e.override_seed(t0, seeds[t0])
            e.override_many(override_rows(cur_tgt, cur_btn, cur_raw, t0, span_end, ports))
            try:
                played = e.steps(n)
            except RuntimeError:                 # the engine stopped (the replay file ended)
                break
            if replanner:
                for k, state in enumerate(played):
                    replanner.record(t0 + k, state, ports)
            st = played[-1]
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
    ap.add_argument("--game-list", default="", help="file with game sha256s (instead of --same-games-as)")
    ap.add_argument("--end", type=int, default=3600, help="last frame rebuilt and compared")
    ap.add_argument("--chunk", type=int, default=4, help="frames chosen at a time")
    ap.add_argument("--lookahead", type=int, default=10, help="extra frames played to judge a candidate")
    ap.add_argument("--samples", type=int, default=8, help="sampled candidates per player and chunk")
    ap.add_argument("--long-lookahead", type=int, default=0,
                    help="judge the best --finalists candidates again over this many frames (0 = off)")
    ap.add_argument("--finalists", type=int, default=3)
    ap.add_argument("--open-loop", action="store_true", help="no search: the model's most likely inputs")
    ap.add_argument("--replan", action="store_true",
                    help="also try the model's inputs computed from the engine's own recent states")
    ap.add_argument("--replan-samples", type=int, default=4, help="sampled candidates from the replanned probabilities")
    ap.add_argument("--seeds", type=int, default=0, help="random seeds tried per chunk (0 = never choose seeds)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--verify", type=int, default=2, help="replay this many outputs in Slippi Playback")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-games", type=int, default=0, help="only the first N games (quick tests)")
    # DIAGNOSTICS (use information from the original game; never part of the real pipeline):
    ap.add_argument("--oracle", action="store_true", help="add the TRUE inputs as a candidate")
    ap.add_argument("--true-inputs", action="store_true", help="the TRUE inputs replace the model's most likely ones")
    ap.add_argument("--true-raw", action="store_true", help="--oracle/--true-inputs also use the TRUE raw stick bytes")
    ap.add_argument("--target", default="perceived", choices=["perceived", "truth"],
                    help="truth: judge candidates against the TRUE states instead of perception's")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    out_dir = f"{OUT}/{args.tag}"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)

    if args.game_list:
        games = open(args.game_list).read().split()
    else:
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
        oracle = None
        if args.target == "truth" or args.oracle or args.true_inputs:
            _, _, true_tgt, true_btn = game_arrays(f"data/parsed/{sha}.parquet", int(stage_ids[sha]), version)
            truth = parse_replay(path)[0]
            truth = truth[(truth.frame >= 0) & (truth.frame < T)].sort_values(["frame", "port"])
            if args.target == "truth":
                per = {c: truth[c].to_numpy().astype(np.float32).reshape(T, 2) for c in ("action", "x", "y", "facing", "percent", "stocks")}
                per["seen"] = np.ones((T, 2), np.float32)
            if args.oracle or args.true_inputs:
                # true raw stick bytes; older replays don't store raw y: then processed * 80
                raw = np.stack([truth.raw_stick_x.to_numpy(np.float32).reshape(T, 2),
                                truth.raw_stick_y.to_numpy(np.float32).reshape(T, 2)], axis=-1)
                fallback = true_tgt[:T, :, :2].astype(np.float32) - 80
                raw = np.where(np.isnan(raw), fallback, raw).astype(np.int64)
                if not args.true_raw:
                    raw = fallback.astype(np.int64)
                oracle = (true_tgt, true_btn, raw)
        end = min(args.end, T - args.lookahead - 5)
        # only the frames needed (keeps the data sent to the worker processes small)
        keep = min(T, end + args.lookahead + 10)
        probs = {k: v[:keep] for k, v in probs.items()}
        jobs.append((sha, path, probs, per, ports, end, vars(args), out_dir, oracle, perceived, int(stage_ids[sha])))
        tables[sha] = perceived
    del model
    torch.cuda.empty_cache()
    print(f"{len(jobs)} games; M2 probabilities done; starting the engine search", flush=True)

    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
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
