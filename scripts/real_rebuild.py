"""
real_rebuild.py - turn a game from a REAL video into a .slp replay (Stage C on real footage).

Until now the closed loop (scripts/m4_closed_loop.py) only rebuilt our own rendered test games,
where the original replay exists and can score the result. Here the input is a game cut from a
downloaded video (scripts/real_prepare.py) and read by perception (scripts/m3_predict.py --real
<id> --countdown). There is no original replay, so nothing can be borrowed from one:
  - stage:       recognised from the video (scripts/m3_stage_id.py, nearest thumbnails);
  - characters:  perception's most common guess per player over the whole game;
  - stocks:      perception's stock count, smoothed and never allowed to go back up (a real game
                 cannot gain a stock; perception sometimes flickers, scripts/real_sanity.py);
  - starting file: a training replay with the same stage and characters (only its settings are
                 used: every input, random number and stored state is replaced);
  - inputs:      chosen by the closed loop: the input model (M2) proposes, the native engine plays,
                 and the candidates that keep the engine's game closest to what perception saw
                 in the video are kept (the recommended settings of docs/stage_c_closed_loop.md).
  - ports:       the ports seen in the video (where the damage "%" signs sit). Every replay in
                 our library uses ports 1 and 2, so the starting file is first moved to the
                 video's ports (src/vid2slp/slp_ports.py) and the engine plays those ports, so
                 the HUD sits where the video shows it.
  - start positions (--spawns, added 2026-11-03): Slippi netplay (all our starting files) runs
                 the "NeutralSpawn" code: the two players always start on the stage's two neutral
                 spawn points. A tournament console without it (vanilla Melee, e.g. Genesis 7)
                 starts each player on the stage's spawn point of their PORT (Battlefield, ports
                 1 and 3: P1 on the ground in the middle, P3 on the left platform). "vanilla"
                 leaves NeutralSpawn out of the engine's code list (the recording carries the
                 list, so Slippi Playback does the same) and sets the players' spawn point bytes
                 to -1 (= their port's). "auto" (default) plays the first frames with both rules
                 and keeps the one whose start positions are closer to what perception saw at
                 the first frame players can move; a game whose video misses the countdown takes
                 the other games' choice (or vanilla).
  - costumes (added 2026-11-04): read from the stock icons in the video (scripts/real_costume.py)
                 and written into the starting file (looks only; the gameplay is the same).

Because there is no original replay, the only score is how well the rebuilt game FOLLOWS what
perception read from the video (action, position, damage, stocks). That says nothing about the
mistakes perception itself made; the comparison video (scripts/real_compare_video.py) is the
check for that.

Run (in tmux; about 45-50 minutes of computer time per minute of game, games run in parallel):
    .venv/bin/python scripts/real_rebuild.py --id 5njRxIAgS-o --games 1,2,3 --tag v1
    .venv/bin/python scripts/real_rebuild.py --id 5njRxIAgS-o --games 1 --end 300 --tag smoke
Output: data/real/<id>/rebuild/<tag>/game<k>.slp, results.csv, config.json.
"""

import argparse
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import av
import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3_stage_id                                       # noqa: E402  (scripts/m3_stage_id.py)
import real_costume                                      # noqa: E402  (scripts/real_costume.py)
import real_prepare                                      # noqa: E402  (scripts/real_prepare.py)
from m4_closed_loop import COUNTDOWN, INTERNAL_TO_EXTERNAL, choose_template, run_game   # noqa: E402
from m4_closed_loop import CODES                                                          # noqa: E402
from vid2slp.engine import Engine                                                        # noqa: E402
from vid2slp.id_data import FEATURE_SETS, arrays_from_states                             # noqa: E402
from vid2slp.id_model import build_from_config, predict_probs                            # noqa: E402
from vid2slp.slp_parse import parse_replay                                               # noqa: E402
from vid2slp.slp_codes import NEUTRAL_SPAWN, without_code                                # noqa: E402
from vid2slp.slp_ports import move_ports, set_spawn_points                               # noqa: E402

REAL = "data/real"
PERCENT_X = real_costume.PERCENT_X    # centre column of each port's damage "%" sign (584 x 480)

# The closed-loop settings recommended in docs/stage_c_closed_loop.md ("How to run"), with
# --countdown --from-video (nothing borrowed from an original game).
RECOMMENDED = dict(chunk=4, lookahead=60, samples=8, long_lookahead=0, finalists=3, early_stop=True,
                   restarts=3, segment=300, open_loop=False, replan=True, replan_samples=4,
                   extra_restarts=5, restart_threshold=3.0, backtrack=4, pos_tail=0.0,
                   countdown=True, from_video=True, start_seed=0, seeds=0,
                   oracle=False, true_inputs=False, true_raw=False, target="perceived")


def video_ports(video, n=80):
    """The two controller ports of the players, from where their damage "%" signs sit in the cut
    game video (Melee draws each port's damage at a fixed place). Returns e.g. (1, 3)."""
    tpl = real_prepare.percent_template()
    container = av.open(video)
    stream = container.streams.video[0]
    total = stream.frames or 10000
    wanted = set(np.linspace(total * 0.05, total * 0.95, n).astype(int))
    votes = {p: 0 for p in PERCENT_X}
    for i, frame in enumerate(container.decode(stream)):
        if i > max(wanted):
            break
        if i not in wanted:
            continue
        grey = frame.to_ndarray(format="gray")[380:475].astype(np.float32)
        best = cv2.matchTemplate(grey, tpl, cv2.TM_CCOEFF_NORMED).max(0)   # best match per column
        for port, x in PERCENT_X.items():
            left = x - tpl.shape[1] // 2
            if best[max(0, left - 6):left + 7].max() >= 0.6:
                votes[port] += 1
    container.close()
    top = sorted(sorted(votes, key=votes.get, reverse=True)[:2])
    return tuple(top), votes


def stage_of(video):
    """Stage name and margin (small = unsure) from the game video, as scripts/m3_stage_id.py does."""
    ref = np.load(m3_stage_id.REF)
    return m3_stage_id.predict(ref["x"], ref["y"], m3_stage_id.thumbnails_of_video(video))


def consistent(perceived):
    """Rules every real game obeys, applied to perception's per-frame guesses:
    one character per player for the whole game (the most common guess while the player is seen),
    and stocks that never go back up (smoothed over 31 frames, then a running minimum)."""
    out = []
    for port, g in perceived.groupby("port"):
        g = g.sort_values("frame").copy()
        seen = g[g.seen > 0]
        g["character"] = int((seen if len(seen) else g).character.mode().iloc[0])
        smooth = g.stocks.rolling(31, center=True, min_periods=1).median().round()
        g["stocks"] = np.minimum.accumulate(smooth.to_numpy()).astype(g.stocks.dtype)
        out.append(g)
    return pd.concat(out).sort_values(["frame", "port"]).reset_index(drop=True)


def write_codes(out_dir):
    """The engine's Gecko list for each spawn rule, written into out_dir. Returns {rule: path}."""
    full = open(CODES, "rb").read()
    paths = {"neutral": f"{out_dir}/codes_neutral.bin", "vanilla": f"{out_dir}/codes_vanilla.bin"}
    open(paths["neutral"], "wb").write(full)
    open(paths["vanilla"], "wb").write(without_code(full, NEUTRAL_SPAWN))
    return paths


def starting_file(template, ports, rule, path, costumes=None):
    """Write the starting file: the template with its players moved to the video's ports and,
    for the vanilla rule, spawn point bytes -1 (each player starts on their own port's point).
    costumes (optional): the costume of each player (slot order), as seen in the video."""
    data = open(template, "rb").read()
    if ports != (1, 2):
        data = move_ports(data, ports)
    if rule == "vanilla":
        data = set_spawn_points(data, {p - 1: -1 for p in ports})
    if costumes is not None:
        data = real_costume.set_costumes(data, {p - 1: c for p, c in zip(ports, costumes)})
    open(path, "wb").write(data)


def engine_start(path, codes, ports):
    """Where the engine puts the two players (x, y) just before they can move (Slippi frame -40)."""
    with Engine(path, codes=codes, preroll=True) as e:
        s = e.state()                                # Slippi frame -123, the countdown starts
        while s["match_frame"] < -40:
            s = e.step()
        return np.array([[s["players"][p - 1]["x"], s["players"][p - 1]["y"]] for p in ports])


def video_start(perceived):
    """Where perception saw the two players (x, y) in the first 4 frames they can move (Slippi
    frames -39..-36, median), or None if the video does not show both there (no countdown)."""
    first = perceived[(perceived.frame >= -39) & (perceived.frame <= -36) & (perceived.seen > 0)]
    if first.groupby("port").size().reindex([0, 1]).fillna(0).min() < 2:
        return None
    return first.groupby("port")[["x", "y"]].median().loc[[0, 1]].to_numpy()


def follow_scores(rebuilt, perceived, end):
    """How closely the rebuilt game follows what perception read (frames 0..end)."""
    # the replay names players by their port (e.g. 0 and 2), perception by slot (0 = lower port)
    slot = {p: i for i, p in enumerate(sorted(rebuilt.port.unique()))}
    rebuilt = rebuilt.assign(port=rebuilt.port.map(slot))

    def frames(t):
        return t[(t.frame >= 0) & (t.frame <= end)].set_index(["frame", "port"]).sort_index()
    r, p = frames(rebuilt), frames(perceived)
    common = r.index.intersection(p.index)
    r, p = r.loc[common], p.loc[common]
    return {"frames": int(len(common) // 2),
            "action_match": round(float((r.action.to_numpy() == p.action.to_numpy()).mean()), 3),
            "x_err": round(float(np.abs(r.x - p.x).mean()), 2), "y_err": round(float(np.abs(r.y - p.y).mean()), 2),
            "percent_err": round(float(np.abs(r.percent - p.percent).mean()), 2),
            "stocks_match": round(float((r.stocks.to_numpy() == p.stocks.to_numpy()).mean()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="folder under data/real/ (the YouTube id)")
    ap.add_argument("--games", default="1", help="comma-separated game numbers, e.g. 1,2,3")
    ap.add_argument("--end", type=int, default=100000, help="last game frame rebuilt (default: whole game)")
    ap.add_argument("--perception", default="percep_v6soft")
    ap.add_argument("--m2", default="perc_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spawns", default="auto", choices=["auto", "vanilla", "neutral"],
                    help="start positions: vanilla (port-based, tournament console), neutral (Slippi netplay) "
                         "or auto (whichever is closer to the video's first frames)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    out_dir = f"{REAL}/{args.id}/rebuild/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    settings = {**RECOMMENDED, "m2": args.m2, "seed": args.seed, "end": args.end}

    index = pd.read_csv("data/index/slp_index.csv").drop_duplicates("sha256")
    splits = pd.read_csv("data/index/splits.csv").set_index("sha256")
    name_to_id = dict(zip(index.stage, index.stage_id))

    # the input model, for the M2 probabilities of every game (GPU, in this process)
    device = "cuda:0"
    ck = torch.load(f"data/m2/runs/{args.m2}/best.pt", map_location=device)
    cfg = ck["config"]
    version = open(f"{cfg['data']}/features.txt").read().strip()
    model = build_from_config(cfg, len(FEATURE_SETS[version])).to(device)
    model.load_state_dict(ck["model"])

    codes = write_codes(out_dir)
    jobs, info = [], {}
    for k in [int(g) for g in args.games.split(",")]:
        video = f"{REAL}/{args.id}/game{k}/video.mp4"
        stage, margin = stage_of(video)
        ports, votes = video_ports(video)
        seen = pd.read_parquet(f"{REAL}/{args.id}/pred_{args.perception}_countdown/game{k}.parquet")
        seen = consistent(seen[seen.frame >= -COUNTDOWN])
        # inside the search, frame numbers start at 0 = Slippi frame -39 (players can move)
        perceived = seen.copy()
        perceived["frame"] += COUNTDOWN
        num, cat = arrays_from_states(perceived, int(name_to_id[stage]), version)
        probs = predict_probs(model, num, cat, device, cfg["window"])
        T = len(num)
        per = {c: np.nan_to_num(perceived[c].to_numpy().astype(np.float32)).reshape(T, 2)
               for c in ("action", "x", "y", "facing", "percent", "stocks", "seen")}
        end = min(args.end + COUNTDOWN, T - settings["lookahead"] - 5)
        chars = [INTERNAL_TO_EXTERNAL[int(perceived[perceived.port == p].character.iloc[0])] for p in (0, 1)]
        template = choose_template(int(name_to_id[stage]), chars, end - COUNTDOWN + settings["lookahead"] + 300,
                                   index, splits)
        keep = min(T, end + settings["lookahead"] + 10)
        probs = {key: v[:keep] for key, v in probs.items()}
        name = f"game{k}"                       # run_game names its output after the first 12 characters
        # the starting file: players moved to the video's ports (engine port numbers 0-3) and,
        # per spawn rule, its spawn point bytes; the rule's start positions vs the video's
        moved = f"{out_dir}/{name}_template.slp"
        in_video = video_start(seen)
        spawn_err = {}
        for rule in ("vanilla", "neutral"):
            starting_file(template, ports, rule, moved)
            if in_video is not None:
                start = engine_start(moved, codes[rule], ports)
                spawn_err[rule] = round(float(np.linalg.norm(start - in_video, axis=1).mean()), 1)
        # costumes from the stock icons in the video (scripts/real_costume.py; looks only, the
        # gameplay is the same in every costume)
        dressed = real_costume.costumes(video, np.load(f"{REAL}/{args.id}/game{k}/frame_map.npy"), seen, chars, ports)
        rule = args.spawns
        if rule == "auto":
            rule = min(spawn_err, key=spawn_err.get) if spawn_err else None   # None: decided below
        jobs.append([name, moved, probs, per, [p - 1 for p in ports], end, settings, out_dir, None, perceived,
                     int(name_to_id[stage])])
        info[name] = {"stage": stage, "stage_margin": round(margin, 3), "characters": chars,
                      "video_ports": list(ports), "port_votes": votes, "template": template,
                      "spawns": rule, "spawn_error": spawn_err, "costumes": [c for c, _, _ in dressed],
                      "costume_margin": [round(m, 1) for _, m, _ in dressed], "seen": seen}
        print(f"game {k}: {stage} (margin {margin:.2f}), characters {chars}, ports {ports} {votes}, "
              f"spawns {rule or '(no countdown seen)'} (start position error per rule {spawn_err}), "
              f"costumes {[c for c, _, _ in dressed]} (margins {[round(m, 1) for _, m, _ in dressed]}), "
              f"{end - COUNTDOWN} frames to rebuild", flush=True)

    # games whose video misses the countdown: the spawn rule the other games of this video
    # showed (the same console all set long), or vanilla if none showed it
    chosen = [v["spawns"] for v in info.values() if v["spawns"]]
    fallback = max(set(chosen), key=chosen.count) if chosen else "vanilla"
    for job in jobs:
        name, moved = job[0], job[1]
        rule = info[name]["spawns"] = info[name]["spawns"] or fallback
        starting_file(info[name]["template"], tuple(info[name]["video_ports"]), rule, moved, info[name]["costumes"])
        job[6] = {**settings, "spawns": rule, "codes": codes[rule]}
        print(f"{name}: spawn rule {rule}", flush=True)
    del model
    torch.cuda.empty_cache()

    with ProcessPoolExecutor(len(jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        runs = list(pool.map(run_game, [tuple(j) for j in jobs]))

    rows = []
    for (name, *_), run in zip(jobs, runs):
        os.remove(f"{out_dir}/{name}_template.slp")
        rebuilt, _ = parse_replay(f"{out_dir}/{name}.slp")
        seen = info[name].pop("seen")
        rows.append({**run, **{k: v for k, v in info[name].items() if k not in ("port_votes", "spawn_error")},
                     **follow_scores(rebuilt, seen, min(args.end, run["last_frame"]))})
    res = pd.DataFrame(rows)
    res.to_csv(f"{out_dir}/results.csv", index=False)
    json.dump({"args": vars(args), "settings": settings,
               "games": {k: {**v, "port_votes": {str(p): n for p, n in v["port_votes"].items()}} for k, v in info.items()}},
              open(f"{out_dir}/config.json", "w"), indent=1)
    pd.set_option("display.width", 220)
    print(res[["game", "seconds", "last_frame", "stage", "frames", "action_match", "x_err", "y_err",
               "percent_err", "stocks_match"]].to_string(index=False))


if __name__ == "__main__":
    main()
