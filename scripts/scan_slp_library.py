"""
scan_slp_library.py - quick inventory of a folder of .slp replays (headers only, fast).

READ-ONLY: it only opens .slp files for reading.

For every .slp under the folder it reads the game-start header (not the frames) and
records: Slippi version, stage, characters, number of players, CPU or human, game
length in frames, and how the game ended. Then it prints totals.

Run:
    .venv/bin/python scripts/scan_slp_library.py /path/to/slp/folder
    .venv/bin/python scripts/scan_slp_library.py /home/prompter/yt2slp/recordings
Writes the per-file table to data/audit/slp_library_<foldername>.csv
"""

import csv
import os
import sys
from collections import Counter

import peppi_py

# Same ID tables as audit_corpus_stats.py (Slippi external character IDs, Melee stage IDs).
CHARACTERS = {
    0: "Captain Falcon", 1: "Donkey Kong", 2: "Fox", 3: "Game & Watch", 4: "Kirby",
    5: "Bowser", 6: "Link", 7: "Luigi", 8: "Mario", 9: "Marth", 10: "Mewtwo",
    11: "Ness", 12: "Peach", 13: "Pikachu", 14: "Ice Climbers", 15: "Jigglypuff",
    16: "Samus", 17: "Yoshi", 18: "Zelda", 19: "Sheik", 20: "Falco",
    21: "Young Link", 22: "Dr. Mario", 23: "Roy", 24: "Pichu", 25: "Ganondorf",
}
STAGES = {
    2: "Fountain of Dreams", 3: "Pokemon Stadium", 8: "Yoshi's Story",
    28: "Dream Land", 31: "Battlefield", 32: "Final Destination",
}


def scan_one(path):
    g = peppi_py.read_slippi(path, skip_frames=True)
    players = g.start.players
    last = (g.metadata or {}).get("lastFrame")
    return {
        "file": path,
        "version": ".".join(map(str, g.start.slippi.version)),
        "stage": STAGES.get(g.start.stage, f"stage {g.start.stage}"),
        "n_players": len(players),
        "characters": "+".join(sorted(CHARACTERS.get(p.character, str(p.character)) for p in players)),
        # player type 0 = human, 1 = CPU
        "any_cpu": any(int(p.type) == 1 for p in players),
        "last_frame": last,
        "end_method": g.end.method.name if g.end and g.end.method is not None else "none",
    }


def main():
    folder = sys.argv[1]
    paths = sorted(os.path.join(d, f) for d, _, fs in os.walk(folder) for f in fs if f.endswith(".slp"))
    rows, errors = [], 0
    for p in paths:
        try:
            rows.append(scan_one(p))
        except Exception as e:  # a corrupt or truncated file should not stop the scan
            errors += 1
            rows.append({"file": p, "error": str(e)[:200]})
    ok = [r for r in rows if "error" not in r]

    minutes = sum((r["last_frame"] or 0) + 123 for r in ok) / 60 / 60
    print(f"files: {len(paths)}   parsed: {len(ok)}   failed: {errors}   total game time: {minutes:.0f} min")
    print("versions:", Counter(r["version"].rsplit(".", 1)[0] for r in ok).most_common())
    print("players per game:", Counter(r["n_players"] for r in ok))
    print("games with a CPU:", sum(r["any_cpu"] for r in ok))
    print("end method:", Counter(r["end_method"] for r in ok))
    print("stages:", Counter(r["stage"] for r in ok).most_common())
    print("characters (per player slot):",
          Counter(c for r in ok for c in r["characters"].split("+")).most_common())
    print("top matchups:", Counter(r["characters"] for r in ok).most_common(15))
    short = sum(1 for r in ok if (r["last_frame"] or 0) < 60 * 30)
    print(f"games shorter than 30 s (likely handwarmers/quits): {short}")

    os.makedirs("data/audit", exist_ok=True)
    out = f"data/audit/slp_library_{os.path.basename(os.path.normpath(folder))}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
