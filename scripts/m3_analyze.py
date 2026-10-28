"""
m3_analyze.py - look at WHERE a perception run makes mistakes on the validation games.

Uses the same fixed validation clips as m3_train.py (evenly spaced through every validation
game) and the run's best checkpoint. Prints:
  1. character: accuracy per character, and whether errors are whole games (e.g. a costume the
     model never saw) or scattered frames;
  2. action: how many wrong frames are only "timing" errors (the predicted action is the right
     one for a frame up to 2 frames earlier or later: the model saw the move but is off by a
     frame or two at a change) vs. truly different actions; and the most common confusions;
  3. action accuracy and position error per character and when airborne vs. on the ground;
  4. damage %: share of frames within 1 and 5 % of the truth.

Run:
    .venv/bin/python scripts/m3_analyze.py --run percep_v1 [--fox-only]
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from m3_train import FixedClips  # noqa: E402  (same validation clips as training)

from vid2slp.matchups import wanted_games  # noqa: E402
from vid2slp.melee_ids import ACTION_NAMES  # noqa: E402
from vid2slp.perception_data import POS_SCALE  # noqa: E402
from vid2slp.perception_model import Perception  # noqa: E402

# internal character ids (as stored per frame in the replay; NOT the character-select ids)
INTERNAL = {0: "Mario", 1: "Fox", 2: "Captain Falcon", 3: "Donkey Kong", 4: "Kirby", 5: "Bowser",
            6: "Link", 7: "Sheik", 8: "Ness", 9: "Peach", 10: "Popo (Ice Climbers)", 11: "Nana",
            12: "Pikachu", 13: "Samus", 14: "Yoshi", 15: "Jigglypuff", 16: "Mewtwo", 17: "Luigi",
            18: "Marth", 19: "Zelda", 20: "Young Link", 21: "Dr. Mario", 22: "Falco", 23: "Pichu",
            24: "Game & Watch", 25: "Ganondorf", 26: "Roy"}
# action names (libmelee's; ids 341+ are character-specific, so names there may not fit Fox/Falco)


def name(a):
    return f"{ACTION_NAMES.get(a, 'action')} ({a})" if a < 341 else f"character-specific {a}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--fox-only", action="store_true",
                    help="only the run's validation games that are Fox-Fox or Fox-Falco")
    ap.add_argument("--games-from", default=None,
                    help="use ANOTHER run's validation games (to compare two runs on the same games)")
    args = ap.parse_args()
    run_dir = f"data/m3/runs/{args.run}"
    cfg = json.load(open(f"{run_dir}/config.json"))
    device = torch.device("cuda:0")
    size = (cfg["width"], cfg["height"])
    model = Perception(size=size, max_len=cfg["clip"], pretrained=False).to(device)
    ck = torch.load(f"{run_dir}/{args.checkpoint}", map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"run {args.run}, checkpoint {args.checkpoint} (step {ck['step']})")

    # the validation games the run itself used (recorded in its config), or another run's
    games = cfg["validation_games"]
    if args.games_from:
        games = json.load(open(f"data/m3/runs/{args.games_from}/config.json"))["validation_games"]
        print(f"scoring on the validation games of run {args.games_from}")
    if args.fox_only:
        games = [g for g in games if g in wanted_games()]
    val = FixedClips(games, cfg["clip"], size, cfg["val_per_game"])
    loader = torch.utils.data.DataLoader(val, batch_size=8, num_workers=10)
    rows = {k: [] for k in ["game", "slot", "char", "char_p", "act", "act_p", "act_near", "air",
                            "dx", "dy", "pct", "pct_p"]}
    item = 0
    with torch.no_grad():
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(batch["video"].to(device))
            out = {k: v.float().cpu() for k, v in out.items()}
            for b in range(len(batch["valid"])):
                game = val.items[item][0]
                item += 1
                v = batch["valid"][b]
                act = batch["action"][b]
                act_p = out["action"][b].argmax(-1)
                for t in torch.where(v)[0].tolist():
                    for s in (0, 1):
                        # "near": predicted action equals the true action within +-2 frames
                        lo, hi = max(t - 2, 0), min(t + 3, len(v))
                        near = bool((act[lo:hi, s] == act_p[t, s]).any())
                        rows["game"].append(game)
                        rows["slot"].append(s)
                        rows["char"].append(int(batch["character"][b, t, s]))
                        rows["char_p"].append(int(out["character"][b, t, s].argmax()))
                        rows["act"].append(int(act[t, s]))
                        rows["act_p"].append(int(act_p[t, s]))
                        rows["act_near"].append(near)
                        rows["air"].append(bool(batch["airborne"][b, t, s] > 0.5))
                        rows["dx"].append(abs(float(out["reg"][b, t, s, 0] - batch["x"][b, t, s])) * POS_SCALE)
                        rows["dy"].append(abs(float(out["reg"][b, t, s, 1] - batch["y"][b, t, s])) * POS_SCALE)
                        rows["pct"].append(float(batch["percent"][b, t, s]) * 100)
                        rows["pct_p"].append(float(out["reg"][b, t, s, 2]) * 100)
    r = {k: np.array(v) for k, v in rows.items()}
    n = len(r["char"])
    print(f"{n:,} player-frames from {len(set(r['game']))} validation games\n")

    # 1. character
    ok = r["char"] == r["char_p"]
    print(f"== character: {ok.mean():.4f} right")
    for c, cnt in Counter(r["char"]).most_common():
        m = r["char"] == c
        wrong = Counter(r["char_p"][m & ~ok]).most_common(2)
        print(f"  {INTERNAL.get(c, c):22s} {m.sum():6d} frames  {ok[m].mean():.3f} right"
              + (f"  mistaken for: {', '.join(f'{INTERNAL.get(p, p)} ({k})' for p, k in wrong)}" if wrong else ""))
    # per game and slot: a player who is wrong on most frames = a whole-game problem
    bad = []
    for g in sorted(set(r["game"])):
        for s in (0, 1):
            m = (r["game"] == g) & (r["slot"] == s)
            if ok[m].mean() < 0.5:
                bad.append(f"{g[:12]} slot {s}: {INTERNAL.get(int(np.median(r['char'][m])))} "
                           f"right on {ok[m].mean():.2f} (mostly called {INTERNAL.get(Counter(r['char_p'][m]).most_common(1)[0][0])})")
    print(f"  players wrong on more than half their frames: {len(bad)}")
    for line in bad:
        print("   ", line)

    # 2. action
    ok = r["act"] == r["act_p"]
    timing = ~ok & r["act_near"]
    print(f"\n== action: {ok.mean():.4f} right; {timing.mean():.4f} more are timing-only "
          f"(right action within 2 frames) -> {(ok | timing).mean():.4f} right allowing +-2 frames")
    print("  most common real mistakes (true -> predicted):")
    for (a, p), cnt in Counter(zip(r["act"][~ok & ~timing], r["act_p"][~ok & ~timing])).most_common(15):
        print(f"    {cnt:5d}  {name(a)} -> {name(p)}")

    # 3. by character and air/ground
    print("\n== action accuracy and position error by group")
    groups = [(INTERNAL.get(c, c), r["char"] == c) for c, _ in Counter(r["char"]).most_common(8)]
    groups += [("on the ground", ~r["air"]), ("in the air", r["air"])]
    for label, m in groups:
        print(f"  {label:22s} {m.sum():6d}  action {ok[m].mean():.3f}  x err {r['dx'][m].mean():5.2f}  "
              f"y err {r['dy'][m].mean():5.2f}")

    # 4. damage %
    e = np.abs(r["pct"] - r["pct_p"])
    print(f"\n== damage %: mean error {e.mean():.2f}; within 1%: {(e <= 1).mean():.3f}; within 5%: "
          f"{(e <= 5).mean():.3f}; over 20% off: {(e > 20).mean():.4f}")


if __name__ == "__main__":
    main()
