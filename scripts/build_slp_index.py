"""
build_slp_index.py - make a catalog (index) of every .slp replay in one or more folders.

READ-ONLY: it only reads the .slp files.

For each file it records a SHA-256 hash of the file contents (to spot exact duplicate
files) and the game-start header: Slippi version, stage, players, characters, CPU or
human, netplay connect codes, start time, and length. Then it marks which games are
"eligible" for our first experiments:

    eligible = 2 players, both human, not teams, NTSC, a legal stage,
               at least 30 seconds long, and not a duplicate of an earlier file

Output: data/index/slp_index.csv  (one row per file; open it in any spreadsheet)

Run:
    .venv/bin/python scripts/build_slp_index.py data/recordings
    (more folders can be listed after the first)
    --out FILE writes somewhere else than data/index/slp_index.csv (e.g. to compare first)
"""

import csv
import hashlib
import os
import sys
from collections import Counter
from multiprocessing import Pool

import peppi_py

from vid2slp.melee_ids import CHARACTERS, LEGAL_STAGES

OUT = "data/index/slp_index.csv"
MIN_FRAMES = 30 * 60  # 30 seconds of gameplay


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def describe(path):
    """Header facts for one file. Never raises: a broken file gets an 'error' column."""
    row = {"path": path, "bytes": os.path.getsize(path)}
    try:
        row["sha256"] = sha256_of(path)
        g = peppi_py.read_slippi(path, skip_frames=True)
        s = g.start
        players = s.players
        row.update({
            "version": ".".join(map(str, s.slippi.version)),
            "stage_id": s.stage,
            "stage": LEGAL_STAGES.get(s.stage, f"other ({s.stage})"),
            "n_players": len(players),
            "ports": " ".join(str(int(p.port) + 1) for p in players),          # 1-based like the game
            "character_ids": " ".join(str(p.character) for p in players),
            "characters": " vs ".join(CHARACTERS.get(p.character, str(p.character)) for p in players),
            "all_human": all(int(p.type) == 0 for p in players),
            "is_teams": bool(s.is_teams),
            "is_pal": bool(s.is_pal),
            # Netplay connect codes (empty for offline/console games). Used to group players.
            "codes": " ".join((p.netplay.code if p.netplay else "") for p in players),
            "start_at": (g.metadata or {}).get("startAt", ""),
            "last_frame": (g.metadata or {}).get("lastFrame"),
            "end_method": g.end.method.name if g.end and g.end.method is not None else "",
        })
    except Exception as e:  # corrupt / truncated / unsupported file
        row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    return row


def main():
    global OUT
    folders = sys.argv[1:]
    if "--out" in folders:                   # optional: --out FILE
        i = folders.index("--out")
        OUT = folders[i + 1]
        folders = folders[:i] + folders[i + 2:]
    if not folders:
        sys.exit("usage: build_slp_index.py FOLDER [FOLDER ...]")
    paths = sorted(os.path.join(d, f) for folder in folders
                   for d, _, files in os.walk(folder) for f in files if f.endswith(".slp"))
    print(f"found {len(paths)} .slp files; hashing and reading headers ...", flush=True)
    with Pool(12) as pool:
        rows = pool.map(describe, paths, chunksize=16)

    # Mark eligibility. The first file with a given hash wins; later copies are duplicates.
    seen = set()
    for r in rows:
        if "error" in r:
            r["eligible"], r["why_not"] = False, "unreadable"
            continue
        reasons = []
        if r["sha256"] in seen:
            reasons.append("duplicate file")
        seen.add(r["sha256"])
        if r["n_players"] != 2:
            reasons.append(f"{r['n_players']} players")
        if not r["all_human"]:
            reasons.append("CPU player")
        if r["is_teams"]:
            reasons.append("teams")
        if r["is_pal"]:
            reasons.append("PAL")
        if r["stage_id"] not in LEGAL_STAGES:
            reasons.append("non-legal stage")
        if (r["last_frame"] or 0) < MIN_FRAMES:
            reasons.append("shorter than 30 s")
        r["eligible"] = not reasons
        r["why_not"] = "; ".join(reasons)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fields = ["sha256", "eligible", "why_not", "path", "bytes", "version", "stage_id", "stage",
              "n_players", "ports", "character_ids", "characters", "all_human", "is_teams", "is_pal",
              "codes", "start_at", "last_frame", "end_method", "error"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["eligible"]]
    print(f"files: {len(rows)}   unreadable: {sum('error' in r for r in rows)}   "
          f"duplicates: {sum('duplicate file' in r['why_not'] for r in rows)}   ELIGIBLE: {len(ok)}")
    print("reasons not eligible:", Counter(x for r in rows for x in r["why_not"].split("; ") if x).most_common())
    hours = sum(r["last_frame"] + 123 for r in ok) / 60 / 3600
    print(f"eligible gameplay: {hours:.1f} hours")
    print("eligible stages:", Counter(r["stage"] for r in ok).most_common())
    print("eligible characters (per player):",
          Counter(c for r in ok for c in r["characters"].split(" vs ")).most_common())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
