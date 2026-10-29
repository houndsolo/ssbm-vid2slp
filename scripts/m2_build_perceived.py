"""
m2_build_perceived.py - build inverse-dynamics (M2) arrays whose INPUT FEATURES come from the
perception model's guesses (data/m3/pred/<run>/) instead of the true replay states, while the
TARGETS are still the true controller inputs.

Why: the Stage C input model (visible_v1) was trained on exact game states. In the pipeline it
sees perception's states, which are a few units off and sometimes have the wrong action. Training
on perception's real mistakes should teach it to cope with them.

Writes data/m2_perc/ in the same layout as scripts/m2_build_dataset.py (v2 features; the fields
perception can't see are 0, and the visible-features models ignore them anyway):
    train_*            perception train games (all rendered Fox-Fox / Fox-Falco train games)
    validation_*       perception validation games
    validation_truth_* the SAME validation games and frames with the TRUE states, so a model can
                       be scored on true vs perceived states on exactly the same data
Splits are the frozen replay split (data/index/splits.csv), so no game is in two splits.
Games are chosen like scripts/m3_train.py does: Fox-Fox / Fox-Falco, playback did not end early
(a desynced render would pair wrong states with the inputs), on-screen timer started at 8:00.

Run (CPU, a few minutes):
    .venv/bin/python scripts/m2_build_perceived.py --perception percep_v3
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from vid2slp import id_data
from vid2slp.matchups import wanted_games

OUT = "data/m2_perc"
FEATURES = "v2"
PRED = ""


def work(job):
    """(num, cat) from perception + (tgt, btn) from the true replay, for one game.
    Also the TRUE (num, cat), zeroed the same way (for validation_truth)."""
    sha, stage_id = job
    true_num, true_cat, tgt, btn = id_data.game_arrays(f"data/parsed/{sha}.parquet", stage_id, FEATURES)
    perceived = pd.read_parquet(f"{PRED}/{sha}.parquet")
    num, cat = id_data.arrays_from_states(perceived, stage_id, FEATURES)
    assert len(num) == len(tgt), f"{sha}: {len(num)} perceived frames vs {len(tgt)} true frames"
    return num, cat, tgt, btn, true_num, true_cat


def build(split, games, with_truth=False):
    jobs = list(zip(games.sha256, games.stage_id))
    with Pool(12) as pool:
        results = pool.map(work, jobs, chunksize=4)     # keeps the order of games
    lengths = [len(r[0]) for r in results]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    names = {"num": 0, "cat": 1, "tgt": 2, "btn": 3}
    for key, i in names.items():
        np.save(f"{OUT}/{split}_{key}.npy", np.concatenate([r[i] for r in results]))
    np.save(f"{OUT}/{split}_offsets.npy", offsets)
    open(f"{OUT}/{split}_games.txt", "w").write("\n".join(games.sha256) + "\n")
    print(f"{split:18s} games {len(games):5d}  frames {offsets[-1]:,}", flush=True)
    if with_truth:
        truth = {"num": 4, "cat": 5, "tgt": 2, "btn": 3}
        for key, i in truth.items():
            np.save(f"{OUT}/{split}_truth_{key}.npy", np.concatenate([r[i] for r in results]))
        np.save(f"{OUT}/{split}_truth_offsets.npy", offsets)
        open(f"{OUT}/{split}_truth_games.txt", "w").write("\n".join(games.sha256) + "\n")
        print(f"{split + '_truth':18s} games {len(games):5d}  frames {offsets[-1]:,}", flush=True)


def main():
    global PRED
    ap = argparse.ArgumentParser()
    ap.add_argument("--perception", default="percep_v3")
    args = ap.parse_args()
    PRED = f"data/m3/pred/{args.perception}"
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/features.txt", "w").write(FEATURES + "\n")
    open(f"{OUT}/perception.txt", "w").write(args.perception + "\n")

    splits = pd.read_csv("data/index/splits.csv")[["sha256", "split"]]
    stage = pd.read_csv("data/index/slp_index.csv").drop_duplicates("sha256")[["sha256", "stage_id"]]
    have = set()
    fox = wanted_games()
    for f in os.listdir(PRED):
        sha = f[:-8]
        info = json.load(open(f"data/m3/renders/{sha}/info.json"))
        if (f.endswith(".parquet") and sha in fox and not info.get("playback_ended_early", False)
                and info["timer_start_seconds"] == 480):
            have.add(sha)
    games = splits[splits.sha256.isin(have)].merge(stage, on="sha256").sort_values("sha256")
    build("train", games[games.split == "train"].reset_index(drop=True))
    build("validation", games[games.split == "validation"].reset_index(drop=True), with_truth=True)


if __name__ == "__main__":
    main()
