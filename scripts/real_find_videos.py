"""
real_find_videos.py - find YouTube videos of tournament spacie dittos (Fox / Falco vs Fox / Falco)
to test (and later train) perception on real footage.

How:
  1. Many YouTube searches (tournaments x matchups, players x matchups) with yt-dlp, titles only.
  2. Keep a video when its title names Melee characters and ALL of them are Fox or Falco (at least
     two mentions, e.g. "... | Fox vs Falco" or "Leffen (Fox) vs Mango (Falco)"), and it is
     5-60 minutes long (shorter: clips/highlights; longer: whole streams).
  3. For each kept video, check the formats: it must have a 50-60 fps version (30 fps loses
     frames; see CLAUDE.md). Records the best 60 fps height and its size.
Output: data/real/candidates.csv (id, title, channel, seconds, height, fps, MB). Nothing is
downloaded here; scripts/real_prepare.py downloads and cuts.

Run:
    .venv/bin/python scripts/real_find_videos.py
"""

import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

YTDLP = ".venv/bin/yt-dlp"
OUT = "data/real/candidates.csv"

CHARACTERS = ["Fox", "Falco", "Marth", "Sheik", "Peach", "Puff", "Jigglypuff", "Falcon", "Captain Falcon",
              "Ice Climbers", "ICs", "Samus", "Pikachu", "Luigi", "Mario", "Doc", "Dr. Mario", "Yoshi",
              "Ganon", "Ganondorf", "Link", "Young Link", "Zelda", "Roy", "Mewtwo", "Game & Watch",
              "Mr. Game & Watch", "Ness", "Bowser", "Kirby", "DK", "Donkey Kong", "Pichu"]
SPACIES = {"Fox", "Falco"}

TOURNAMENTS = ["Genesis", "Evo", "Smash Summit", "The Big House", "Pound", "Shine", "Super Smash Con",
               "Full Bloom", "Battle of BC", "Riptide", "Mainstage", "LACS", "Collision", "Apex",
               "CEO", "DreamHack", "Smash N Splash", "Get On My Level", "Major Upset", "Coinbox",
               "Ludwig Smash Invitational", "Tipped Off", "Fight Pitt", "Port Priority", "Mango Cup"]
PLAYERS = ["Leffen", "Hax$", "Mango", "Westballz", "SFAT", "Moky", "Soonsay", "Magi", "Zamu",
           "Ginger", "Medz", "Salt", "PPMD", "Lucky", "KoreanDJ", "Polish", "Kevin Maples", "Krudo",
           "Spark", "Mekk", "Zain", "Aklo", "Trif", "Ice", "Wizzy", "Joshman", "Kalamazhu", "Bbatts"]
MATCHUPS = ["Fox vs Fox", "Fox vs Falco", "Falco vs Falco"]


def search(query, n=30):
    """yt-dlp search: list of (id, seconds, channel, title)."""
    r = subprocess.run([YTDLP, f"ytsearch{n}:{query}", "--flat-playlist", "--print",
                        "%(id)s\t%(duration)s\t%(channel)s\t%(title)s"], capture_output=True, text=True)
    rows = []
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 4:
            rows.append(parts)
    return rows


def characters_in(title):
    """Melee character names mentioned in a title, in order (longest names first so
    'Captain Falcon' is not also counted as 'Falco')."""
    found, rest = [], title
    for name in sorted(CHARACTERS, key=len, reverse=True):
        pattern = r"\b" + re.escape(name) + r"\b"
        for _ in re.findall(pattern, rest):
            found.append(name)
        rest = re.sub(pattern, " ", rest)
    return found


def is_spacie_ditto(title):
    chars = characters_in(title)
    return len(chars) >= 2 and set(chars) <= SPACIES


def best_60fps(video_id):
    """(height, fps, MB) of the best 50-60 fps video-only format up to 1080p, or None."""
    r = subprocess.run([YTDLP, "-J", f"https://youtu.be/{video_id}"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    formats = json.loads(r.stdout).get("formats", [])
    best = None
    for f in formats:
        fps, h = f.get("fps") or 0, f.get("height") or 0
        if fps >= 50 and 0 < h <= 1080 and f.get("vcodec", "none") != "none":
            size = f.get("filesize") or f.get("filesize_approx") or 0
            if best is None or h > best[0]:
                best = (h, fps, round(size / 1e6))
    return best


def main():
    queries = [f"{t} Melee {m}" for t in TOURNAMENTS for m in MATCHUPS[:2]]
    queries += [f"{p} Melee {m}" for p in PLAYERS for m in MATCHUPS[:2]]
    queries += [f"Melee Singles {m}" for m in MATCHUPS]
    print(f"{len(queries)} searches ...", flush=True)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(search, queries))
    seen, keep = set(), []
    for rows in results:
        for vid, seconds, channel, title in rows:
            if vid in seen:
                continue
            seen.add(vid)
            try:
                seconds = float(seconds)
            except ValueError:
                continue
            if 300 <= seconds <= 3600 and is_spacie_ditto(title):
                keep.append({"id": vid, "title": title, "channel": channel, "seconds": int(seconds)})
    print(f"{len(seen)} videos seen, {len(keep)} spacie dittos 5-60 min; checking formats ...", flush=True)
    with ThreadPoolExecutor(8) as pool:
        fmts = list(pool.map(best_60fps, [k["id"] for k in keep]))
    for k, f in zip(keep, fmts):
        k["height"], k["fps"], k["MB"] = f if f else (0, 0, 0)
    df = pd.DataFrame(keep).sort_values(["fps", "height"], ascending=False)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    ok = df[df.fps >= 50]
    print(f"{len(ok)} with 60 fps, {ok.seconds.sum() / 3600:.1f} h, {ok.MB.sum() / 1000:.1f} GB -> {OUT}")


if __name__ == "__main__":
    main()
