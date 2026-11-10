"""
engine_pool.py - several copies of the native engine that play the SAME game side by side, so
the closed-loop search (scripts/m4_closed_loop.py) can try its candidate inputs on all of them at
once, one per CPU core.

Why (measured 2026-11-05, scripts profiled on a 600-frame real game): one closed-loop rebuild
runs on ONE core and spends ~80 % of its time waiting for the engine to play candidate inputs
(each 4-frame chunk: ~12-15 candidates per player, each played for ~64 frames from the same
saved state), ~15 % in Python around it (input rows, reading states, cost) and ~1 % on the GPU.
The candidates of a chunk don't depend on each other, so they can be played at the same time.

How:
  - Each copy runs in its own helper process (Python + its engine), so the Python work around
    a candidate (input rows, reading the engine's state lines, the cost) is spread too.
  - Everything that changes the game (capture, restore, override, steps, seeds) is sent to EVERY
    copy, so all copies are always in the same state. The pool therefore looks like one Engine
    to the search code. After each capture the copies' states are compared (frame, random seed,
    positions); if they ever differ the run stops with an error instead of giving a wrong result.
  - trials(): the candidates of one chunk are handed out one at a time to whichever copy is free;
    each plays it (restore, override, play, cost) and the costs come back in the original order.
    The copies share the best cost found so far for the chunk, so a candidate that is already
    worse than it stops early (as with one engine, which stops at the best seen so far).

Same result as one engine: every candidate gets the same cost, except candidates that were
stopped early, whose (partial) cost is already above a candidate's full cost, so they can't be
the best anyway. The chosen inputs are the same and the output replay is byte-identical (checked
with scripts/m4_pool_check.py: 2026-11-05, 600 frames, 1-16 engines).
"""

import multiprocessing
import multiprocessing.connection

import numpy as np

from vid2slp.engine import Engine
from vid2slp.id_to_slp import processed_sequence

HISTORY = 600   # frames of committed inputs re-processed before a chunk (X/Y and L/R alternation)
EARLY_STOP_PIECE = 8   # a candidate is simulated in pieces of this many frames (see play_trials)


def override_rows(cur_tgt, cur_btn, cur_raw, t0, end, ports, first=0):
    """Engine override rows for frames t0 .. end-1, both players, from the input arrays.

    The arrays are turned into .slp input values with processed_sequence (which alternates
    X/Y and L/R for new presses, so it is run over HISTORY frames before t0 as well).
    first: the frame number of the arrays' first row (0 = whole game; the pool's helpers get
    only the part from t0 - HISTORY on)."""
    a = max(0, t0 - HISTORY)
    rows = []
    for i, port in enumerate(ports):
        seq = processed_sequence(cur_tgt[a:end, i], cur_btn[a:end, i])
        for f in range(t0, end):
            v = seq[f - a]
            raw_c = cur_tgt[f, i, 2:4] - 80      # c-stick raw bytes: the processed value * 80
            rows.append((f + first, int(port), v["stick_x"], v["stick_y"], v["cstick_x"], v["cstick_y"],
                         v["trigger"], v["buttons"], cur_raw[f, i, 0], cur_raw[f, i, 1], raw_c[0], raw_c[1]))
    return rows


def state_cost(states, t0, per, ports, pos_tail=0.0):
    """Sum over frames t0.. and both players of how different the engine's states are from the
    perceived ones. Units: 1 = one frame with the wrong action (10 world units of position
    error, or 10 % damage error, cost as much; a wrong stock count costs 3; wrong facing 0.5).
    Frames where perception did not see the character count 0.2.
    Position errors beyond 30 units add pos_tail per extra unit (--pos-tail; 0 = the old flat cap).
    Why (2026-10-30): with a flat cap, once a rebuilt character is 30+ units off, every candidate
    costs the same and nothing pulls it back (game 6cd8 drifted offstage and lost a stock)."""
    total = 0.0
    for k, st in enumerate(states):
        f = t0 + k
        if f >= len(per["action"]):
            break
        for i, port in enumerate(ports):
            p = st["players"][port]
            c = float(p["action"] != per["action"][f, i])
            d = abs(p["x"] - per["x"][f, i]) + abs(p["y"] - per["y"][f, i])
            c += min(d, 30.0) / 10.0 + pos_tail * max(d - 30.0, 0.0)
            c += 0.5 * float((p["facing"] > 0) != (per["facing"][f, i] > 0))
            c += min(abs(p["percent"] - per["percent"][f, i]), 20.0) / 10.0
            c += 3.0 * float(p["stocks"] != per["stocks"][f, i])
            total += c * (1.0 if per["seen"][f, i] else 0.2)
    return total


def play_trial(e, window, cand, full, best, per, ports, pos_tail):
    """In a helper: play one candidate from the engine's saved state and return its cost.
    window = (t0, n, end_frame, first, cur_tgt, cur_btn, cur_raw, early_stop): the current inputs
    from frame `first` on; cand = (player, tgt, btn, raw) for frames t0 .. t0+n-1.
    Like m4_closed_loop's trial(), the candidate stops early once its cost is ABOVE the best cost
    any copy has found for this chunk so far (`best`, shared by all helpers). Strictly above, not
    "reaches": a candidate stopped at exactly the best cost would tie with it, and the search keeps
    the first of tied candidates, which might then differ from playing them one by one. full: never
    stop early (the search's "base" candidate: its full cost goes into the statistics)."""
    t0, n, end_frame, first, cur_tgt, cur_btn, cur_raw, early_stop = window
    i, tgt, btn, raw = cand
    a, b = t0 - first, t0 - first + n                 # the chunk's rows in the arrays
    t_tgt, t_btn, t_raw = cur_tgt.copy(), cur_btn.copy(), cur_raw.copy()
    t_tgt[a:b, i], t_btn[a:b, i], t_raw[a:b, i] = tgt, btn, raw
    e.restore()
    e.override_many(override_rows(t_tgt, t_btn, t_raw, a, end_frame - first, ports, first))
    total, f = 0.0, t0
    while f < end_frame:
        piece = min(EARLY_STOP_PIECE, end_frame - f)
        total += state_cost(e.steps(piece), f, per, ports, pos_tail)
        f += piece
        if early_stop and not full and f < end_frame and total > best.value:
            return total                              # can't be the best: stopped early
    with best.get_lock():                             # played to the end: a real cost, share it
        if total < best.value:
            best.value = total
    return total


def helper(conn, number, per, ports, pos_tail, best):
    """One helper process: owns one engine and does what the pool asks, until "close".
    Messages: ("open", engine arguments), ("call", (method name, args)), ("window", the chunk's
    inputs, no reply), ("trial", (candidate, full)), ("close", None).
    Replies: ("ok", result) or ("error", text). Only helper 0 sends long results (states) back."""
    e, window = None, None
    while True:
        cmd, arg = conn.recv()
        try:
            if cmd == "close":
                if e is not None:
                    e.close()
                conn.send(("ok", None))
                return
            if cmd == "open":
                if e is not None:
                    e.close()
                slp, kwargs = arg
                e = Engine(slp, **kwargs)
                result = e.state()
            elif cmd == "call":
                name, args = arg
                result = getattr(e, name)(*args)
                if name == "steps" and number != 0:
                    result = None                     # the other copies' states are the same
            elif cmd == "window":
                window = arg                              # kept for the trials that follow
                continue
            elif cmd == "trial":
                cand, full = arg
                result = play_trial(e, window, cand, full, best, per, ports, pos_tail)
            conn.send(("ok", result))
        except Exception as ex:                       # e.g. the engine stopped (replay file ended)
            conn.send(("error", f"{type(ex).__name__}: {ex}"))


def summary(st):
    """What must be equal in every copy after a capture: frame, random seed and each player's
    action, position and damage."""
    return (st["match_frame"], st["rng"],
            tuple((p["action"], p["x"], p["y"], p["percent"]) for p in st["players"]))


class EnginePool:
    """n engine copies that look like ONE Engine to the closed-loop search, plus trials().
    per: perceived arrays (for the cost), ports: the engine ports of the two players."""

    def __init__(self, n, per, ports, pos_tail=0.0):
        ctx = multiprocessing.get_context("spawn")    # a fresh Python per helper (no CUDA, no torch)
        self.best = ctx.Value("d", float("inf"))      # best cost of the chunk so far, shared by all helpers
        self.conns, self.procs = [], []
        for k in range(n):
            ours, theirs = ctx.Pipe()
            p = ctx.Process(target=helper, args=(theirs, k, per, list(ports), pos_tail, self.best), daemon=True)
            p.start()
            self.conns.append(ours)
            self.procs.append(p)
        self._state = None

    def _ask(self, cmd, arg, conns=None):
        """Send one message to the given helpers (default: all) and return helper 0's result."""
        conns = conns or self.conns
        for c in conns:
            c.send((cmd, arg))
        replies = [c.recv() for c in conns]
        errors = [r[1] for r in replies if r[0] == "error"]
        if errors:
            raise RuntimeError(errors[0])
        return [r[1] for r in replies]

    # ---- the Engine interface used by the search (every copy does the same) ----

    def open(self, slp, **kwargs):
        """(Re)start every copy's engine on this starting file (as Engine(slp, **kwargs))."""
        self._state = self._ask("open", (slp, kwargs))[0]
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass                                          # the copies stay up for the next open()

    def state(self):
        return self._state

    def capture(self):
        states = self._ask("call", ("capture", ()))
        if len({summary(s) for s in states}) > 1:
            raise RuntimeError("engine copies are out of step (they should all play the same game)")
        self._state = states[0]
        return self._state

    def restore(self):
        self._state = self._ask("call", ("restore", ()))[0]
        return self._state

    def mark(self):
        states = self._ask("call", ("mark", ()))
        if len({summary(s) for s in states}) > 1:
            raise RuntimeError("engine copies are out of step (they should all play the same game)")
        self._state = states[0]
        return self._state

    def back(self):
        self._state = self._ask("call", ("back", ()))[0]
        return self._state

    def steps(self, n):
        out = self._ask("call", ("steps", (n,)))[0]
        self._state = out[-1]
        return out

    def override_many(self, rows):
        self._ask("call", ("override_many", (rows,)))

    def override_seed(self, frame, seed):
        self._ask("call", ("override_seed", (frame, seed)))

    def frame_seeds(self, on):
        self._ask("call", ("frame_seeds", (on,)))

    def record(self, trim=True):
        return self._ask("call", ("record", (trim,)), self.conns[:1])[0]   # one copy is enough

    # ---- candidates on all copies at once ----

    def trials(self, t0, n, end_frame, cur, cands, early_stop=True, start=None, lead=None):
        """Costs of candidate inputs for frames t0 .. t0+n-1, each played from the saved state up to
        end_frame (as m4_closed_loop's trial()). cur: the current input arrays (whole game);
        cands: [(player, tgt, btn, raw)]. Returns the costs in the order of cands.
        The candidates are handed out one at a time, in order, to whichever copy is free (copies
        finish at different times: a candidate that turns out bad stops early).
        start: a real cost of one of these candidates known already (it is not played again).
        lead: a candidate played alone first, always to the end ("base": usually the best, chosen
        813 of 1,110 times, 2026-11-05; once its cost is known, worse candidates can stop early)."""
        if not cands:
            return []
        first = max(0, t0 - HISTORY)                  # the helpers only need this part of the arrays
        window = (t0, n, end_frame, first, *[cur[k][first:end_frame] for k in ("tgt", "btn", "raw")], early_stop)
        self.best.value = float("inf") if start is None else start   # no copy is busy now: safe to reset
        for c in self.conns:
            c.send(("window", window))
        costs, error = [None] * len(cands), None
        order = list(range(len(cands)))
        if lead is not None and early_stop:
            order.remove(lead)
            self.conns[0].send(("trial", (cands[lead], True)))
            kind, result = self.conns[0].recv()
            if kind == "error":
                raise RuntimeError(result)
            costs[lead] = result
        free, busy, k = list(reversed(self.conns)), {}, 0
        while k < len(order) or busy:
            while free and k < len(order) and error is None:
                c, j = free.pop(), order[k]
                c.send(("trial", (cands[j], j == lead)))
                busy[c] = j
                k += 1
            if not busy:
                break
            for c in multiprocessing.connection.wait(list(busy)):
                kind, result = c.recv()
                if kind == "error":
                    error = result
                else:
                    costs[busy[c]] = result
                del busy[c]
                free.append(c)
        if error:
            raise RuntimeError(error)
        return costs

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
                c.recv()
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=10)
