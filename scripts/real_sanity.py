"""
real_sanity.py - plausibility checks of perception on every cut real video.

Why: real footage has no true labels, so we cannot score it like renders. But some things are
impossible in a real game, and how often perception produces them is a useful health number:
  - a player's character changing during the game          ("character steady": >= 99 % of frames
                                                              show the player's most common character)
  - a player's stock count going UP                         ("stocks never up")
  - more than 3 stocks lost (a game starts with 4)          ("at most 3 losses")
  - damage dropping a lot without a stock lost nearby       ("no odd damage drops")
Stocks are smoothed first (median over 31 frames), so one-frame flickers do not count.

"Full games" are games where the countdown was seen, at least 30 s were cut, and the timer was
read on at least 80 % of the frames; everything else is a fragment (highlights, replays, a
game cut in the middle) and is reported separately.

Run (after m3_predict.py --real for each video):
    .venv/bin/python scripts/real_sanity.py --run percep_v6soft
Writes one row per game and player to logs/real_sanity_<run>.csv.
"""

import argparse
import glob
import json

import numpy as np
import pandas as pd


def check_player(g):
    """g: the prediction rows of one player in one game, sorted by frame. Returns a dict."""
    stocks = g.stocks.rolling(31, center=True, min_periods=1).median().round().to_numpy()
    ups = int(np.sum(np.diff(stocks) > 0))
    downs = int(np.sum(np.diff(stocks) < 0))
    # damage drops of more than 15 that do not happen within 3 s of a stock loss
    damage = g.percent.rolling(15, center=True, min_periods=1).median().to_numpy()
    drop = np.diff(damage) < -15
    lost = (np.diff(stocks) < 0).astype(int)
    near_loss = pd.Series(lost).rolling(361, center=True, min_periods=1).max().to_numpy() > 0
    chars = g.character.value_counts(normalize=True)
    return dict(char=int(chars.index[0]), char_share=round(float(chars.iloc[0]), 3),
                stock_ups=ups, stock_downs=downs, odd_damage_drops=int(np.sum(drop & ~near_loss)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="percep_v6soft")
    args = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(f"data/real/*/pred_{args.run}_countdown")):
        video = d.split("/")[2]
        if video.startswith("_"):
            continue                      # test folders (_ws_, _lvc_, ...), not downloaded videos
        games = {g["game"]: g for g in json.load(open(f"data/real/{video}/games.json"))}
        for f in sorted(glob.glob(f"{d}/game*.parquet")):
            k = int(f.split("/")[-1][4:-8])
            if k not in games:
                continue                  # leftover from an older cut
            info = games[k]
            full = (info["has_countdown"] and info["video_frames_labeled"] >= 1800
                    and info["video_frames_labeled"] / info["video_frames_total"] >= 0.8)
            t = pd.read_parquet(f)
            t = t[t.frame >= 0]
            for port, g in t.groupby("port"):
                rows.append(dict(video=video, game=k, slot=port, full_game=full, **check_player(g.sort_values("frame"))))
    r = pd.DataFrame(rows)
    r.to_csv(f"logs/real_sanity_{args.run}.csv", index=False)
    full = r[r.full_game]
    print(f"{r.video.nunique()} videos, {len(r) // 2} games; full games: {len(full) // 2} in {full.video.nunique()} videos")
    for name, x in (("full games", full), ("fragments", r[~r.full_game])):
        print(f"{name} ({len(x)} player tracks): character steady {np.mean(x.char_share >= 0.99):.0%}, "
              f"stocks never up {np.mean(x.stock_ups == 0):.0%}, at most 3 losses {np.mean(x.stock_downs <= 3):.0%}, "
              f"no odd damage drops {np.mean(x.odd_damage_drops == 0):.0%}")
    ok = (full.char_share >= 0.99) & (full.stock_ups == 0) & (full.stock_downs <= 3) & (full.odd_damage_drops == 0)
    print(f"full-game player tracks passing all four checks: {ok.mean():.0%}")


if __name__ == "__main__":
    main()
