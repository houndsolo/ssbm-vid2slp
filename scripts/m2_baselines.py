"""
m2_baselines.py - simple, no-learning baselines for inverse dynamics (milestone 2).

A trained model is only interesting if it clearly beats these:

  neutral - always predict "nothing pressed, sticks centred, trigger released".
  lookup  - a table built from the TRAINING games: for each combination of
            (character, action state on the previous / current / next frame), predict the
            most common input seen in training. If that combination never occurred in
            training, fall back to (character, current, next), then (character, current),
            then neutral. "Next frame" is allowed because we reconstruct offline from a
            whole video, so the future is available.

Scores both on validation and test (baselines have no settings to tune, so looking at
test here leaks nothing). Uses the metrics in src/vid2slp/id_metrics.py.

Run (CPU, ~2-3 min, ~10 GB RAM):
    .venv/bin/python scripts/m2_baselines.py
Writes data/m2/baselines.json
"""

import json

import numpy as np

from vid2slp.id_data import CATEGORICAL, load_split
from vid2slp.id_metrics import format_scores, score

DATA = "data/m2"
ACTION = [c for c, _ in CATEGORICAL].index("action")
CHARACTER = [c for c, _ in CATEGORICAL].index("character")
NEUTRAL_TGT = np.array([80, 80, 80, 80, 0], np.int16)


def keys_for(split):
    """Lookup keys for every (frame, port): three levels, from most to least specific."""
    cat = np.asarray(split["cat"])
    char = cat[:, :, CHARACTER].astype(np.int64)
    act = cat[:, :, ACTION].astype(np.int64)
    # previous / next action within the same game (at game edges, repeat the current one)
    prev = np.vstack([act[:1], act[:-1]])
    nxt = np.vstack([act[1:], act[-1:]])
    starts, ends = split["offsets"][:-1], split["offsets"][1:] - 1
    prev[starts] = act[starts]
    nxt[ends] = act[ends]
    k3 = ((char * 512 + prev) * 512 + act) * 512 + nxt
    k2 = (char * 512 + act) * 512 + nxt
    k1 = char * 512 + act
    return [k3.ravel(), k2.ravel(), k1.ravel()]


def fit_mode(keys, values):
    """For each distinct key, the most common value. Returns (sorted keys, their modes)."""
    pair = keys * 1024 + values           # values are < 1024
    uniq, counts = np.unique(pair, return_counts=True)
    k, v = uniq // 1024, uniq % 1024
    # within each key, keep the value with the highest count
    order = np.lexsort((-counts, k))
    k, v = k[order], v[order]
    first = np.concatenate([[True], k[1:] != k[:-1]])
    return k[first], v[first]


def predict(table_list, key_list, default):
    """Look up each key in the most specific table that knows it."""
    out = np.full(len(key_list[0]), default, np.int64)
    found = np.zeros(len(out), bool)
    for (tk, tv), keys in zip(table_list, key_list):
        pos = np.clip(np.searchsorted(tk, keys), 0, len(tk) - 1)
        hit = (tk[pos] == keys) & ~found
        out[hit] = tv[pos[hit]]
        found |= hit
    return out, found


def main():
    train = load_split(DATA, "train")
    print("building lookup tables from training games ...", flush=True)
    train_keys = keys_for(train)
    tgt = np.asarray(train["tgt"]).reshape(-1, 5).astype(np.int64)
    btn = np.asarray(train["btn"]).reshape(-1).astype(np.int64)
    # one set of tables per predicted quantity: 4 stick axes, trigger, button bitmask
    tables = [[fit_mode(k, col) for k in train_keys] for col in [tgt[:, i] for i in range(5)] + [btn]]
    defaults = list(NEUTRAL_TGT) + [0]
    del tgt, btn

    results = {}
    for split_name in ("validation", "test"):
        split = load_split(DATA, split_name)
        true_tgt, true_btn = np.asarray(split["tgt"]), np.asarray(split["btn"])
        n = len(true_tgt)

        neutral = score(true_tgt, np.broadcast_to(NEUTRAL_TGT, true_tgt.shape), true_btn,
                        np.zeros_like(true_btn), split["offsets"])
        keys = keys_for(split)
        preds, found = zip(*(predict(t, keys, d) for t, d in zip(tables, defaults)))
        pred_tgt = np.stack(preds[:5], axis=-1).reshape(n, 2, 5).astype(np.int16)
        pred_btn = preds[5].reshape(n, 2).astype(np.uint8)
        lookup = score(true_tgt, pred_tgt, true_btn, pred_btn, split["offsets"])
        lookup["key_coverage_most_specific"] = float(np.mean(predict(tables[0][:1], keys[:1], 0)[1]))

        results[split_name] = {"neutral": neutral, "lookup": lookup}
        print(f"\n[{split_name}] {n * 2:,} player-frames")
        print("  neutral:", format_scores(neutral))
        print("  lookup: ", format_scores(lookup))
        print(f"  lookup: share of frames whose exact (prev, cur, next) key was seen in training: "
              f"{lookup['key_coverage_most_specific']:.3f}")

    json.dump(results, open(f"{DATA}/baselines.json", "w"), indent=1)
    print("\nwrote", f"{DATA}/baselines.json")


if __name__ == "__main__":
    main()
