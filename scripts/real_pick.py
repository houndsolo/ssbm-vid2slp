"""
real_pick.py - from several rebuilds of the same real games, keep per game the one that fits the
video best.

Why (2026-11-10): the search is chaotic; one unlucky choice can make the rebuilt game drift away
from the video for a while. Rebuilding the same game several times with different random seeds
and keeping the run with the lowest search cost (how far the rebuilt game was from what
perception read, summed over the game: `track_cost` in results.csv) uses only the video, never a
true replay. Tested on the 12 validation renders (where the truth exists), 4 runs each: picked
by this cost 0.919 action match, picked by the truth 0.920, single runs 0.879-0.908.

A run whose game ended earlier than the others (a player lost all stocks too soon) has fewer
frames in its cost, so it is only considered if no run reached the full length.

Run (after scripts/real_rebuild_max.sh):
    .venv/bin/python scripts/real_pick.py --id 5njRxIAgS-o --tags v3max_s0,v3max_s1,v3max_s2,v3max_s3 --out v3
Output: data/real/<id>/rebuild/<out>/game<k>.slp (copies of the picked ones) and results.csv
(the picked rows plus a "picked_from" column and every run's cost per frame).
"""

import argparse
import glob
import json
import os
import shutil

import pandas as pd

REAL = "data/real"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--tags", required=True, help="comma-separated rebuild tags of the same games")
    ap.add_argument("--out", required=True, help="tag of the folder for the picked games")
    args = ap.parse_args()
    base = f"{REAL}/{args.id}/rebuild"
    runs = []
    for tag in args.tags.split(","):
        path = f"{base}/{tag}/results.csv"
        if not os.path.exists(path):
            print(f"{tag}: no results.csv (not finished?), left out")
            continue
        runs.append(pd.read_csv(path).assign(tag=tag))
    t = pd.concat(runs, ignore_index=True)
    t["cost_per_frame"] = t.track_cost / t.last_frame.clip(lower=1)

    os.makedirs(f"{base}/{args.out}", exist_ok=True)
    # each picked game's settings (stage, ports, spawn rule ...) from its run's config.json, and the
    # engine code lists, so later steps (e.g. scripts/real_background.py) can use the picked folder
    config = {"picked_from": {}, "games": {}}
    rows = []
    for game, g in t.groupby("game"):
        full = g[g.last_frame == g.last_frame.max()]          # runs that reached the full length
        best = full.loc[full.track_cost.idxmin()]
        print(f"{game}: " + ", ".join(f"{r.tag} {r.cost_per_frame:.3f}/frame (action {r.action_match:.3f}, "
                                      f"last frame {r.last_frame})" for r in g.itertuples())
              + f" -> {best.tag}")
        shutil.copy(f"{base}/{best.tag}/{game}.slp", f"{base}/{args.out}/{game}.slp")
        run_config = json.load(open(f"{base}/{best.tag}/config.json"))
        config["picked_from"][game] = best.tag
        config["games"][game] = run_config["games"][game]
        config.setdefault("runs", {})[best.tag] = {k: run_config[k] for k in ("args", "settings")}
        for codes in glob.glob(f"{base}/{best.tag}/codes_*.bin"):    # the same in every run
            shutil.copy(codes, f"{base}/{args.out}/")
        rows.append({**best.drop(["tag"]).to_dict(), "picked_from": best.tag,
                     "all_costs_per_frame": {r.tag: round(r.cost_per_frame, 3) for r in g.itertuples()}})
    json.dump(config, open(f"{base}/{args.out}/config.json", "w"), indent=1)
    res = pd.DataFrame(rows)
    res.to_csv(f"{base}/{args.out}/results.csv", index=False)
    print(res[["game", "picked_from", "action_match", "x_err", "y_err", "percent_err", "stocks_match"]].to_string())


if __name__ == "__main__":
    main()
