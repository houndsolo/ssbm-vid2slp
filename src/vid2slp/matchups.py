"""
matchups.py - which characters played in each replay, and the project's matchup filter.

Scope (owner's choice 2026-10-27): only Fox vs Fox and Fox vs Falco games.

The characters of every parsed replay are worked out once (the most common internal character
id of each player, from data/parsed/<sha256>.parquet) and cached in
data/index/game_characters.csv, because reading ~6,000 parquet files takes about a minute.
"""

import os

import pandas as pd

CACHE = "data/index/game_characters.csv"
FOX, FALCO = 1, 22                      # internal character ids (as stored per frame)
WANTED = {(FOX, FOX), (FOX, FALCO)}     # sorted pairs of the two players' characters


def game_characters():
    """DataFrame: sha256, split, char_a, char_b (char_a <= char_b). Built once, then cached."""
    if os.path.exists(CACHE):
        return pd.read_csv(CACHE)
    manifest = pd.read_csv("data/parsed/manifest.csv")
    rows = []
    for r in manifest.itertuples():
        df = pd.read_parquet(r.parquet, columns=["port", "character"])
        chars = sorted(int(g.character.mode()[0]) for _, g in df.groupby("port"))
        if len(chars) != 2:
            continue                    # not a 2-player game
        rows.append({"sha256": r.sha256, "split": r.split, "char_a": chars[0], "char_b": chars[1]})
    out = pd.DataFrame(rows)
    out.to_csv(CACHE, index=False)
    return out


def wanted_games():
    """Set of sha256 of the Fox-Fox and Fox-Falco games (all splits)."""
    g = game_characters()
    keep = [(a, b) in WANTED for a, b in zip(g.char_a, g.char_b)]
    return set(g.sha256[keep])
