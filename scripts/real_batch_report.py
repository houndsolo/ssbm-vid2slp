"""
real_batch_report.py - what goes wrong most often when real games are rebuilt?

Why (2026-11-12): scripts/real_rebuild_batch.sh rebuilds one game from each of many real videos.
This collects, over all finished games, where the rebuilt replay disagrees with what perception
read in the video (there is no true replay of a real game, so every number is "vs perception"):
  - per game: action match, position / damage error, stocks, whether the game ended too early;
  - stock losses: does every death in the video happen in the replay, and how close in time;
  - mismatched frames split into kinds (same split as scripts/real_mismatch.py):
      timing      the video's action shows up in the replay within +-2 frames
      drift       part of a mismatch run longer than 30 frames (the replay went elsewhere)
      unsure      perception itself was less than 80 % sure of its action on that frame
      other       the rest
    (a frame counts in the first kind that fits, in that order);
  - the most common confused action pairs over all games;
  - games that failed (no results.csv) and the error.

Run (after the batch):
    .venv/bin/python scripts/real_batch_report.py --list data/real/rebuild_batch1.csv --tag b1
Writes logs/real_batch_report_<tag>.csv (one row per game) and prints a summary.
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_mismatch import load, name, runs_of            # noqa: E402

REAL = "data/real"


def stock_losses(g, col):
    """Frames where the stock count drops (one player, sorted by frame)."""
    s, f = g[col].to_numpy(), g.frame.to_numpy()
    return [int(f[i]) for i in range(1, len(s)) if s[i] < s[i - 1]]


def one_game(vid, k, tag, perception, confusions):
    a = argparse.Namespace(id=vid, tag=tag, perception=perception)
    t = load(a, k)
    t["ok"] = t.action_r == t.action_p
    row = {"frames": len(t) // 2, "action_match": t.ok.mean()}

    # kinds of mismatch (first that fits: timing, drift, unsure, other)
    near = np.zeros(len(t), dtype=bool)
    long = np.zeros(len(t), dtype=int)
    for port, g in t.groupby("port"):
        r, p = g.action_r.to_numpy(), g.action_p.to_numpy()
        hit = np.zeros(len(g), dtype=bool)
        for d in range(-2, 3):
            hit |= np.roll(r, d) == p
        near[g.index] = hit
        long[g.index] = runs_of((~g.ok).to_numpy())
    bad = ~t.ok.to_numpy()
    timing = bad & near
    drift = bad & ~timing & (long > 30)
    unsure = bad & ~timing & ~drift & (t.action_prob.to_numpy() < 0.8)
    other = bad & ~timing & ~drift & ~unsure
    n = len(t)
    row.update({"miss_timing": timing.sum() / n, "miss_drift": drift.sum() / n,
                "miss_unsure": unsure.sum() / n, "miss_other": other.sum() / n})
    for p_act, r_act in zip(t.action_p[other | drift], t.action_r[other | drift]):
        confusions[(name(p_act), name(r_act))] += 1

    # stock losses: pair each of the video's with the nearest one in the replay
    paired, gaps, extra = 0, [], 0
    for port, g in t.groupby("port"):
        r, p = stock_losses(g, "stocks_r"), stock_losses(g, "stocks_p")
        for x in p:
            if r:
                gaps.append(min(abs(x - y) for y in r))
        paired += min(len(r), len(p))
        extra += max(0, len(r) - len(p))
        row.setdefault("video_losses", 0)
        row["video_losses"] += len(p)
    row.update({"replay_losses_missing": row["video_losses"] - paired, "replay_losses_extra": extra,
                "losses_within_20": int(sum(gp <= 20 for gp in gaps)),
                "worst_loss_gap": int(max(gaps)) if gaps else 0})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--perception", default="percep_v6soft")
    args = ap.parse_args()
    games = pd.read_csv(args.list)
    rows, confusions = [], Counter()
    for g in games.itertuples():
        vid, k = g.video, int(g.game)
        out = f"{REAL}/{vid}/rebuild/{args.tag}"
        base = {"video": vid, "game": k, "matchup": g.matchup}
        if not os.path.exists(f"{out}/results.csv"):
            log = f"logs/real_rebuild_batch/{args.tag}/{vid}_game{k}.log"
            err = ""
            if os.path.exists(log):
                lines = [ln.strip() for ln in open(log, errors="replace") if ln.strip()]
                err = next((ln for ln in reversed(lines) if "Error" in ln or "error" in ln), lines[-1] if lines else "")
            rows.append({**base, "status": "failed", "error": err[:160]})
            continue
        res = pd.read_csv(f"{out}/results.csv").iloc[0]
        row = {**base, "status": "done", "seconds": res.seconds, "stage": res.stage,
               "last_frame": res.last_frame, "x_err": res.x_err, "percent_err": res.percent_err,
               "stocks_match": res.stocks_match, "spawns": res.spawns}
        try:
            row.update(one_game(vid, k, args.tag, args.perception, confusions))
        except Exception as err:                     # a report must not stop at one odd game
            row["error"] = f"report: {err}"[:160]
        rows.append(row)
    t = pd.DataFrame(rows)
    os.makedirs("logs", exist_ok=True)
    t.to_csv(f"logs/real_batch_report_{args.tag}.csv", index=False)

    done = t[t.status == "done"]
    print(f"{len(t)} games: {len(done)} rebuilt, {(t.status == 'failed').sum()} failed")
    for e in t[t.status == "failed"].itertuples():
        print(f"  FAILED {e.video} game {e.game}: {e.error}")
    if done.empty:
        return
    print("\nper game (median, worst):")
    for c, worst in (("action_match", "min"), ("x_err", "max"), ("percent_err", "max"), ("stocks_match", "min")):
        print(f"  {c:14s} median {done[c].median():.3f}  worst {getattr(done[c], worst)():.3f}")
    print("\nshare of player-frames that mismatch, by kind (mean over games):")
    for c in ("miss_timing", "miss_drift", "miss_unsure", "miss_other"):
        print(f"  {c[5:]:8s} {done[c].mean():.3f}")
    print(f"\nstock losses in the videos: {int(done.video_losses.sum())}; missing in the replay "
          f"{int(done.replay_losses_missing.sum())}, extra in the replay {int(done.replay_losses_extra.sum())}, "
          f"within 20 frames {int(done.losses_within_20.sum())}; games with a gap > 60 frames: "
          f"{int((done.worst_loss_gap > 60).sum())}")
    print("\nby matchup:")
    print(done.groupby("matchup")[["action_match", "x_err", "stocks_match"]].agg(["median", "size"]).round(3).to_string())
    total = sum(confusions.values())
    print("\nmost common confusions in drift/other mismatches (video -> replay, share):")
    for (p, r), n in confusions.most_common(15):
        print(f"  {p:24s} -> {r:24s} {n / total:.3f}")
    print("\nworst 8 games by action match:")
    print(done.sort_values("action_match").head(8)[["video", "game", "matchup", "stage", "action_match", "x_err",
                                                     "stocks_match", "miss_drift", "replay_losses_missing",
                                                     "replay_losses_extra"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
