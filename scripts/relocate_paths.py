"""
relocate_paths.py - after scripts/relocate_old_project.sh, rewrite the old file locations in
this project's data tables (CSV / JSON) so they point to the new places.

    /home/prompter/yt2slp/recordings                  -> /home/prompter/ssbm-vid2slp/data/recordings
    /home/prompter/yt2slp/.docker-state/local/playback -> /home/prompter/ssbm-vid2slp/data/tools/playback

Only exact text replacement of these two prefixes; every changed file is listed.
Run:
    .venv/bin/python scripts/relocate_paths.py           # dry run: only counts
    .venv/bin/python scripts/relocate_paths.py --apply
"""

import glob
import sys

MOVES = [
    ("/home/prompter/yt2slp/recordings", "/home/prompter/ssbm-vid2slp/data/recordings"),
    ("/home/prompter/yt2slp/.docker-state/local/playback", "/home/prompter/ssbm-vid2slp/data/tools/playback"),
]
FILES = (glob.glob("data/**/*.csv", recursive=True) + glob.glob("data/**/*.json", recursive=True)
         + glob.glob("data/**/*.txt", recursive=True))


def main():
    apply = "--apply" in sys.argv
    changed = 0
    for path in FILES:
        if path.startswith("data/recordings/") or path.startswith("data/tools/"):
            continue                         # never touch the moved files themselves
        try:
            text = open(path).read()
        except UnicodeDecodeError:
            continue
        new = text
        for old, repl in MOVES:
            new = new.replace(old, repl)
        if new != text:
            changed += 1
            if apply:
                open(path, "w").write(new)
    left = sum("/home/prompter/yt2slp" in open(p, errors="ignore").read() for p in FILES
               if not p.startswith(("data/recordings/", "data/tools/")))
    print(f"{'changed' if apply else 'would change'} {changed} files; files still mentioning the old folder: {left}")


if __name__ == "__main__":
    main()
