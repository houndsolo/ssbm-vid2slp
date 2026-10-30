"""
m4_pick_best_run.py - choose, for every game, the best of several closed-loop runs.

Why: the closed loop (scripts/m4_closed_loop.py) is chaotic. With another random seed the same
game can come out much better or worse (one game: action match 0.90 in one run, 0.59 in another),
because a few unlucky choices make the rebuilt game drift away from the video. Running it several
times and keeping the run that fits the VIDEO best fixes most of that.

How "fits the video best" is measured: the same cost the search uses (scripts/m4_closed_loop.py,
cost() with the flat 30-unit position cap): per frame and player, wrong action 1, position error
up to 30 units / 10, wrong facing 0.5, damage error up to 20 % / 10, wrong stock count 3;
frames where perception did not see the character count 0.2. It compares the output with the
PERCEIVED states only. The original replay is used only for the score printed afterwards.

Run:
    .venv/bin/python scripts/m4_pick_best_run.py --tags cl_restart3_full,cl_base_s1 --out-tag best_of_2
Output: data/m4/closed_loop/<out-tag>/ with the chosen .slp per game and results.csv (the chosen
run's scores, copied from that run's results.csv, plus which run was chosen).
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from vid2slp.slp_parse import parse_replay

OUT = "data/m4/closed_loop"


def perceived_cost(out, per):
    """Total cost of output states `out` against perceived states `per` (frames already cut)."""
    m = per.merge(out, on=["frame", "port"], suffixes=("_p", ""), how="left")
    # a frame missing from the output (the rebuilt game ended early) counts as fully wrong
    c = (m.action != m.action_p).astype(float)
    c += np.minimum((m.x - m.x_p).abs() + (m.y - m.y_p).abs(), 30).fillna(30) / 10
    c += 0.5 * ((m.facing > 0) != (m.facing_p > 0))
    c += np.minimum((m.percent - m.percent_p).abs(), 20).fillna(20) / 10
    c += 3 * (m.stocks != m.stocks_p)
    c *= np.where(m.seen > 0, 1.0, 0.2)
    return float(c.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma-separated closed-loop run tags")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--perception", default="percep_v3")
    ap.add_argument("--end", type=int, default=3600)
    args = ap.parse_args()
    tags = args.tags.split(",")
    results = {t: pd.read_csv(f"{OUT}/{t}/results.csv").set_index("game") for t in tags}
    games = sorted(set.intersection(*[set(r.index) for r in results.values()]))
    shas = pd.read_csv("data/index/splits.csv").sha256
    os.makedirs(f"{OUT}/{args.out_tag}", exist_ok=True)
    rows = []
    for g in games:
        sha = [s for s in shas if s.startswith(g)][0]
        per = pd.read_parquet(f"data/m3/pred/{args.perception}/{sha}.parquet")
        per = per[(per.frame >= 1) & (per.frame <= args.end)]
        costs = {}
        for t in tags:
            out, _ = parse_replay(f"{OUT}/{t}/{g}.slp")
            costs[t] = perceived_cost(out, per)
        pick = min(costs, key=costs.get)
        shutil.copy(f"{OUT}/{pick}/{g}.slp", f"{OUT}/{args.out_tag}/{g}.slp")
        row = results[pick].loc[g].to_dict()
        row.update({"game": g, "picked": pick, "perceived_cost": round(costs[pick], 1),
                    "best_possible_action": max(results[t].loc[g, "action_match"] for t in tags)})
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/{args.out_tag}/results.csv", index=False)
    cols = ["action_match", "action_anim_match", "airborne_match", "x_err", "stocks_match"]
    pd.set_option("display.width", 200)
    print(res[["game", "picked", "perceived_cost"] + cols + ["best_possible_action"]].round(3).to_string(index=False))
    print("mean:", res[cols + ["best_possible_action"]].mean().round(3).to_dict())
    for t in tags:
        print(f"single run {t}:", results[t].loc[games, cols].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
