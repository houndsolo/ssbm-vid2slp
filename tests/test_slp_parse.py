"""
Tests for the SLP data pipeline (milestone 1).

Run:  .venv/bin/python -m pytest -q

Tests that need data files (the .slp library, data/old_pilot, data/index) are skipped
automatically if those files are not present.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import pytest

from vid2slp.slp_parse import FIRST_FRAME, last_copy_rows, parse_replay

OLD_ROOT = "/home/prompter/yt2slp"
ROLLBACK_REPLAY = OLD_ROOT + "/recordings/examples/Game_20260425T023703.slp"  # has 3 duplicate frame ids


def test_last_copy_rows_keeps_last_copy_in_frame_order():
    # frame 2 and 3 were written twice (rollback); the later copies (rows 3, 4) must win
    assert list(last_copy_rows([1, 2, 3, 2, 3, 4])) == [0, 3, 4, 5]
    assert list(last_copy_rows([5, 6, 7])) == [0, 1, 2]


@pytest.mark.skipif(not os.path.exists(ROLLBACK_REPLAY), reason="replay library not present")
def test_rollback_replay_has_unique_contiguous_frames():
    table, info = parse_replay(ROLLBACK_REPLAY)
    assert info["duplicate_frame_ids"] == 3
    assert info["missing_frames"] == 0
    for _, per_port in table.groupby("port"):
        frames = per_port.frame.to_numpy()
        assert frames[0] == FIRST_FRAME
        assert np.array_equal(frames, np.arange(FIRST_FRAME, frames[-1] + 1))


# The old corpus's labels.json values were checked against the raw .slp with a different
# parser (slippi-js) in the audit: 681,600 rows, 0 mismatches. So they are a good
# independent reference for OUR parser. We compare a few clips exactly (as float32).
PILOT_LABELS = sorted(glob.glob("data/old_pilot/*/clips/*/attempt-*/labels.json"))
FIELDS = {  # our column -> (labels.json section, key)
    "x": ("state", "x"), "y": ("state", "y"), "percent": ("state", "percent"),
    "action": ("state", "action"), "stocks": ("state", "stocks"), "facing": ("state", "facing"),
    "jumps": ("state", "jumps"), "airborne": ("state", "airborne"),
    "vel_air_x": ("state", "velocity_air_x"), "vel_y": ("state", "velocity_y"),
    "stick_x": ("effective", "stick_x"), "stick_y": ("effective", "stick_y"),
    "cstick_x": ("effective", "cstick_x"), "cstick_y": ("effective", "cstick_y"),
    "trigger": ("effective", "trigger"), "buttons": ("effective", "buttons"),
    "buttons_physical": ("physical", "buttons"), "raw_stick_x": ("physical", "raw_stick_x"),
}


@pytest.mark.skipif(not PILOT_LABELS, reason="data/old_pilot not present")
@pytest.mark.parametrize("labels_path", PILOT_LABELS[::60])  # 5 clips spread over both corpora
def test_parser_matches_verified_old_labels(labels_path):
    labels = json.load(open(labels_path))
    shard = json.load(open(labels_path.split("/attempt-")[0] + "/shard.json"))
    table, _ = parse_replay(os.path.join(OLD_ROOT, shard["file"]))
    ours = table.set_index(["frame", "port"])

    mismatches = {}
    for row in labels["rows"]:
        mine = ours.loc[(row["frame"], row["port"])]
        for col, (section, key) in FIELDS.items():
            if not row[section]["mask"][key]:
                continue  # value not recorded in the old labels
            want = np.float32(row[section]["values"][key])
            if np.float32(mine[col]) != want:
                mismatches[col] = mismatches.get(col, 0) + 1
    assert mismatches == {}, f"{os.path.basename(os.path.dirname(os.path.dirname(labels_path)))}: {mismatches}"


@pytest.mark.skipif(not os.path.exists("data/index/splits.csv"), reason="splits not built")
def test_splits_have_no_leaks():
    s = pd.read_csv("data/index/splits.csv")
    assert set(s.split) == {"train", "validation", "test"}
    assert s.groupby("sha256").split.nunique().max() == 1, "a replay file is in two splits"
    assert s.groupby("group").split.nunique().max() == 1, "an opponent is in two splits"


@pytest.mark.skipif(not os.path.exists("data/parsed/manifest.csv"), reason="parsed tables not built")
def test_every_split_game_was_parsed_and_tables_are_well_formed():
    s = pd.read_csv("data/index/splits.csv")
    m = pd.read_csv("data/parsed/manifest.csv")
    ok = m[m["error"].isna()] if "error" in m else m
    assert set(s.sha256) == set(ok.sha256)
    assert (ok.missing_frames == 0).all()
    # spot-check one table per split
    for _, r in ok.groupby("split").head(1).iterrows():
        t = pd.read_parquet(r.parquet)
        assert not t.duplicated(["frame", "port"]).any()
        assert t.port.nunique() == 2
        assert t.frame.min() == FIRST_FRAME
