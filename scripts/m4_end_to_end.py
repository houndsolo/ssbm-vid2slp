"""
m4_end_to_end.py - first END-TO-END test: rendered video -> .slp -> played back -> compared
with the original game (Stage C, version 0).

For each chosen validation game (rendered video + perception output from scripts/m3_predict.py):
  1. States: the perception model's per-frame guesses (data/m3/pred/<perception run>/).
  2. Inputs: the "visible features" inverse-dynamics model (m2 run, trained with
     --zero-features/--zero-cats so it only uses what perception can see) applied to those
     perceived states.
  3. Write a .slp: the ORIGINAL replay is used as a template for the game settings (stage,
     characters, costumes, random seed, the countdown frames). This is a shortcut for now: the
     video would have to tell us the stage and costumes too (not built yet). For every frame
     >= 0 we overwrite the inputs and the stored states (pre-frame: action, x, y, facing, %;
     post-frame: also shield, stocks, airborne). scripts/m4_resync_probe.py showed that
     playback only reads the PRE-frame states (resync) and ignores the post-frame ones.
     The start-of-frame states follow src/vid2slp/assemble.py (learned start-of-frame rules;
     optionally (--snap) standing characters snapped onto real floor heights; with --deaths,
     dying characters are put just past the blast zone and living ones kept inside it, because
     resync does not force the stock count).
  4. Play it in Slippi Playback (resync ON = Slippi Launcher's default) and compare the
     regenerated game with the original, frame by frame.

Variants (to separate the effect of states from inputs):
  perc_model          perceived states + predicted inputs       (the real pipeline)
  perc_true_inputs    perceived states + the TRUE inputs         (are the states the limit?)
  perc_neutral        perceived states + no inputs at all        (do inputs matter with resync?)
  perc_model_noresync perceived states + predicted inputs, resync OFF (inputs alone)
Diagnostics (true inputs): truth_true_inputs (stored = true states), truth_xy_from_perc (true
states but perceived positions), truth_action_from_perc (true states but perceived actions).

Run (in tmux):
    .venv/bin/python scripts/m4_end_to_end.py --perception percep_v3 --m2 visible_v1 --games 12
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from vid2slp.assemble import load_rules, place_deaths, replay_scores as compare, snap_to_ground, start_of_frame
from vid2slp.id_data import FEATURE_SETS, arrays_from_states, game_arrays
from vid2slp.id_model import build_from_config, predict_split
from vid2slp.id_to_slp import processed_sequence
from vid2slp.matchups import wanted_games
from vid2slp.playback import regenerate
from vid2slp.slp_parse import parse_replay
from vid2slp.slp_write import write_frames

OUT = "data/m4/end_to_end"
# variant -> (inputs, stored states)
VARIANTS = {
    "perc_model": ("model", "perceived"),
    "perc_true_inputs": ("true", "perceived"),
    "perc_neutral": ("neutral", "perceived"),
    "perc_model_noresync": ("model", "perceived"),
    # diagnostics (true inputs; stored states = truth, or truth with one part from perception)
    "truth_true_inputs": ("true", "truth"),
    "truth_xy_from_perc": ("true", "truth_but_perceived_xy"),
    "truth_action_from_perc": ("true", "truth_but_perceived_action"),
    "truth_deaths_placed": ("true", "truth_deaths_placed"),     # needs --deaths
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perception", default="percep_v3")
    ap.add_argument("--m2", default="visible_v1")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--end", type=int, default=3600, help="last frame compared (3600 = 1 minute)")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--naive-pre", action="store_true",
                    help="old writer: start-of-frame state = end of the previous frame (no learned rules)")
    ap.add_argument("--snap", action="store_true",
                    help="snap grounded y onto floor heights (tested 2026-10-28 on 4 games: slightly worse)")
    ap.add_argument("--deaths", action="store_true",
                    help="place deaths on the blast zones and keep living characters inside them")
    ap.add_argument("--same-games-as", default="", help="reuse the games of an earlier run (its --tag)")
    ap.add_argument("--tag", default="", help="output folder name (default: <perception>_<m2>)")
    ap.add_argument("--variants", default="perc_model,perc_true_inputs,perc_neutral,perc_model_noresync")
    args = ap.parse_args()
    wanted = args.variants.split(",")
    pred_dir = f"data/m3/pred/{args.perception}"
    out_root = f"{OUT}/{args.tag or args.perception + '_' + args.m2}"
    shutil.rmtree(out_root, ignore_errors=True)
    os.makedirs(out_root)

    # Games: validation, Fox-Fox / Fox-Falco, with a perception output.
    splits = pd.read_csv("data/index/splits.csv").set_index("sha256")
    stage_ids = pd.read_csv("data/index/slp_index.csv").drop_duplicates("sha256").set_index("sha256").stage_id
    fox = wanted_games()
    have = sorted(f[:-8] for f in os.listdir(pred_dir) if f.endswith(".parquet"))
    candidates = [s for s in have if s in fox and splits.loc[s, "split"] == "validation"]
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(candidates, min(args.games, len(candidates)), replace=False))
    if args.same_games_as:
        chosen = json.load(open(f"{OUT}/{args.same_games_as}/config.json"))["games"]
    print(f"{len(candidates)} candidate games, using {len(chosen)}", flush=True)

    early_actions, facing_early = load_rules()
    device = "cuda:0"
    ck = torch.load(f"data/m2/runs/{args.m2}/best.pt", map_location=device)
    cfg = ck["config"]
    version = open(f"{cfg['data']}/features.txt").read().strip()
    model = build_from_config(cfg, len(FEATURE_SETS[version])).to(device)
    model.load_state_dict(ck["model"])

    jobs = []
    for sha in chosen:
        path = splits.loc[sha, "path"]
        stage = int(stage_ids[sha])
        perceived = pd.read_parquet(f"{pred_dir}/{sha}.parquet")
        if args.snap:
            perceived, snapped = snap_to_ground(perceived, stage)
        orig, _ = parse_replay(path)
        ports = sorted(orig.port.unique())

        # inputs predicted from the PERCEIVED states
        num, cat = arrays_from_states(perceived, stage, version)
        one = {"num": num, "cat": cat, "offsets": np.array([0, len(num)])}
        pred_tgt, pred_btn = predict_split(model, one, device, cfg["window"])
        # true inputs, in the same encoding (shown lossless by the M2 oracle check)
        _, _, true_tgt, true_btn = game_arrays(f"data/parsed/{sha}.parquet", stage, version)
        neutral_tgt = np.zeros_like(true_tgt)
        neutral_tgt[..., :4] = 80
        inputs = {"model": (pred_tgt, pred_btn), "true": (true_tgt, true_btn),
                  "neutral": (neutral_tgt, np.zeros_like(true_btn))}

        # stored states: perception's, optionally with deaths placed on the blast zones
        # (after the input model has seen the plain perceived states)
        stored = perceived
        if args.deaths:
            stored, n_deaths = place_deaths(perceived, stage)
            print(f"{sha[:12]}: {n_deaths} deaths placed", flush=True)
        state = stored.set_index(["frame", "port"]).sort_index()
        names = ["action", "x", "y", "facing", "percent", "shield", "stocks", "airborne", "character"]
        truth = orig[orig.frame >= 0].set_index(["frame", "port"]).sort_index().loc[state.index]
        sources = {"perceived": {c: state[c].to_numpy() for c in names},
                   "truth": {c: truth[c].to_numpy() for c in names}}
        # diagnostics: the truth, with only the positions (or only the actions) from perception
        sources["truth_but_perceived_xy"] = {**sources["truth"], "x": state.x.to_numpy(), "y": state.y.to_numpy()}
        # a check of the death fix alone: true states, but deaths placed the same way
        if args.deaths:
            fixed, _ = place_deaths(orig[orig.frame >= 0], stage)
            fixed = fixed.set_index(["frame", "port"]).sort_index().loc[state.index]
            sources["truth_deaths_placed"] = {c: fixed[c].to_numpy() for c in names}
        sources["truth_but_perceived_action"] = {**sources["truth"], "action": state.action.to_numpy()}
        row_of = {key: i for i, key in enumerate(state.index)}

        def make_writer(kind, source="perceived"):
            cols = sources[source]
            tgt, btn = inputs[kind]
            seqs = [processed_sequence(tgt[:, i], btn[:, i]) for i in range(2)]
            slot = {int(p): i for i, p in enumerate(ports)}

            def pre(frame, port):
                if frame < 0 or frame >= len(seqs[0]):
                    return None
                values = dict(seqs[slot[int(port)]][frame])
                i = row_of.get((frame - 1, port))         # end of the previous frame
                j = row_of.get((frame, port))             # end of this frame
                if i is not None:
                    prev = {k: cols[k][i] for k in ("action", "x", "y", "facing", "percent")}
                    if args.naive_pre or j is None:
                        now = prev                        # start of frame = end of previous frame
                    else:
                        now = {k: cols[k][j] for k in ("action", "x", "y", "facing", "percent")}
                    values.update(start_of_frame(prev, now, early_actions, facing_early))
                return values

            def post(frame, port):
                i = row_of.get((frame, port))
                if frame < 0 or i is None:
                    return None
                return {"action": int(cols["action"][i]), "x": float(cols["x"][i]), "y": float(cols["y"][i]),
                        "facing": float(cols["facing"][i]), "percent": float(cols["percent"][i]),
                        "shield": float(cols["shield"][i]), "stocks": int(cols["stocks"][i]),
                        "airborne": int(cols["airborne"][i]), "character": int(cols["character"][i])}
            return pre, post

        for name in wanted:
            kind, source = VARIANTS[name]
            slp = f"{out_root}/{sha[:12]}_{name}.slp"
            pre, post = make_writer(kind, source)
            write_frames(path, slp, pre=pre, post=post)
            jobs.append((sha, name, slp, name != "perc_model_noresync", orig, perceived))

    def run(job):
        sha, name, slp, resync, orig, perceived = job
        try:
            regen, seconds = regenerate(slp, slp[:-4] + "_run", -123, args.end, resync=resync)
            table, _ = parse_replay(regen)
            return {"game": sha[:12], "variant": name, **compare(orig, table, perceived, args.end)}
        except Exception as e:
            return {"game": sha[:12], "variant": name, "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(args.parallel) as pool:
        rows = list(pool.map(run, jobs))
    res = pd.DataFrame(rows)
    res.to_csv(f"{out_root}/results.csv", index=False)

    # perception alone on the same frames, for reference
    ref = []
    for sha in chosen:
        o = parse_replay(splits.loc[sha, "path"])[0]
        p = pd.read_parquet(f"{pred_dir}/{sha}.parquet")
        if args.snap:
            p, _ = snap_to_ground(p, int(stage_ids[sha]))
        ref.append(compare(o, p, p, args.end))
    ref = pd.DataFrame(ref)
    print(f"\nperception alone (stored states vs truth): action {ref.action_match.mean():.3f}  "
          f"x err {ref.x_err.mean():.2f}  y err {ref.y_err.mean():.2f}  % err {ref.percent_err.mean():.2f}")
    if "error" in res.columns:
        print(res[res.error.notna()][["game", "variant", "error"]].to_string(index=False))
        res = res[res.error.isna()]
    print("\nregenerated game vs original (mean over games):")
    print(res.groupby("variant")[["action_match", "action_anim_match", "airborne_match", "x_err", "y_err", "percent_err", "stocks_match",
                                  "follows_stored_action", "follows_stored_x"]].mean().round(3).to_string())
    json.dump({"perception": args.perception, "m2": args.m2, "m2_step": ck["step"], "games": chosen,
               "end": args.end, "seed": args.seed, "deaths": args.deaths, "snap": args.snap,
               "naive_pre": args.naive_pre, "variants": wanted}, open(f"{out_root}/config.json", "w"), indent=1)
    print(f"\nwrote {out_root}/results.csv")


if __name__ == "__main__":
    main()
