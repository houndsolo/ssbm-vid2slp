"""
m3_stage_id.py - recognise the STAGE of a game from its video.

Why: Stage C (scripts/m4_closed_loop.py) still borrows the stage from the original replay. A real
video has no replay, so the stage must come from the picture. Perception (percep_v3) already
gives the characters, but not the stage.

How (deliberately simple): the stage background fills most of the picture and looks very
different from stage to stage. So:
  1. From every video take only its keyframes (the frames the mp4 stores whole; about one
     every 2 seconds, very fast to decode) and shrink each one to a 32 x 24 colour thumbnail.
  2. Reference set: thumbnails of TRAINING-split games, labelled with the stage from their replay.
  3. A new game: for each of its thumbnails and each stage, the distance to the most similar
     reference thumbnail of that stage (smallest pixel difference, "nearest neighbour"). The
     stage with the smallest total over all thumbnails wins. (Tried first: each thumbnail votes
     for the stage of its single nearest reference; 162/164 right, both misses near 50/50
     votes. The total-distance rule got 164/164.)
The score below is measured on validation and test games (split by replay file, never seen in
the reference set).

Limits: tested on our own renders (Slippi Playback Dolphin, full screen, no overlay). Real
footage (stream overlays, crops, other colours) is untested and may need more than this.

Run:
    .venv/bin/python scripts/m3_stage_id.py --build                 # reference set from train renders
    .venv/bin/python scripts/m3_stage_id.py --eval                  # accuracy on validation + test renders
    .venv/bin/python scripts/m3_stage_id.py --games list.txt        # predict stages -> data/m3/stage_id/pred.csv (added to it)
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import av
import numpy as np
import pandas as pd

RENDERS = "data/m3/renders"
OUT = "data/m3/stage_id"
REF = f"{OUT}/reference.npz"
SIZE = (32, 24)          # thumbnail width, height


def thumbnails(sha):
    """All keyframes of a game's video as small colour thumbnails, float32 [n, 24*32*3] in 0..1."""
    container = av.open(os.path.join(RENDERS, sha, "video.mp4"))
    stream = container.streams.video[0]
    stream.codec_context.skip_frame = "NONKEY"     # decode keyframes only (fast)
    out = [f.to_ndarray(format="rgb24", width=SIZE[0], height=SIZE[1]) for f in container.decode(stream)]
    container.close()
    return np.stack(out).reshape(len(out), -1).astype(np.float32) / 255.0


def rendered(split):
    """(sha256, stage name) of the games of one split that have a rendered video."""
    s = pd.read_csv("data/index/splits.csv")
    s = s[s.split == split]
    s = s[[os.path.exists(os.path.join(RENDERS, h, "video.mp4")) for h in s.sha256]]
    return list(zip(s.sha256, s.stage))


def predict(ref_x, ref_y, x):
    """Stage of a game from its thumbnails x. Returns (stage, margin): margin = how much larger
    the runner-up stage's total distance is (0.10 = 10 % larger; small = unsure)."""
    # squared pixel distance between every thumbnail and every reference thumbnail
    d = (x ** 2).sum(1)[:, None] - 2 * x @ ref_x.T + (ref_x ** 2).sum(1)[None, :]
    stages = np.unique(ref_y)
    # for each stage: distance to its nearest reference thumbnail, added up over the thumbnails
    total = np.array([d[:, ref_y == s].min(1).clip(min=0).sum() for s in stages])
    order = total.argsort()
    return stages[order[0]], float(total[order[1]] / max(total[order[0]], 1e-9) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--games", default="", help="file with sha256s to predict")
    ap.add_argument("--per-stage", type=int, default=150, help="training games per stage in the reference set")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if args.build:
        games = rendered("train")
        rng = np.random.default_rng(args.seed)
        by_stage = {}
        for sha, stage in games:
            by_stage.setdefault(stage, []).append(sha)
        chosen = []
        for stage, shas in sorted(by_stage.items()):
            pick = rng.permutation(shas)[:args.per_stage]
            chosen += [(sha, stage) for sha in pick]
            print(f"{stage:20s} {len(shas):5d} train renders, {len(pick)} used")
        with ProcessPoolExecutor(12) as pool:
            thumbs = list(pool.map(thumbnails, [sha for sha, _ in chosen], chunksize=4))
        x = np.concatenate(thumbs)
        y = np.concatenate([[stage] * len(t) for (_, stage), t in zip(chosen, thumbs)])
        np.savez_compressed(REF, x=x, y=y, games=np.array([sha for sha, _ in chosen]))
        print(f"reference set: {len(x)} thumbnails from {len(chosen)} games -> {REF}")

    ref = np.load(REF)
    ref_x, ref_y = ref["x"], ref["y"]

    if args.eval:
        rows = []
        for split in ["validation", "test"]:
            games = rendered(split)
            with ProcessPoolExecutor(12) as pool:
                thumbs = list(pool.map(thumbnails, [sha for sha, _ in games], chunksize=4))
            for (sha, stage), t in zip(games, thumbs):
                guess, share = predict(ref_x, ref_y, t)
                rows.append({"split": split, "sha256": sha, "stage": stage, "guess": guess, "margin": share})
        res = pd.DataFrame(rows)
        res["correct"] = res.stage == res.guess
        res.to_csv(f"{OUT}/eval.csv", index=False)
        print(res.groupby("split").correct.agg(["mean", "count"]))
        print(res.groupby("stage").correct.agg(["mean", "count"]))
        print("wrong:", res[~res.correct][["split", "sha256", "stage", "guess", "margin"]].to_string())

    if args.games:
        shas = [l.strip() for l in open(args.games) if l.strip()]
        rows = []
        for sha in shas:
            guess, share = predict(ref_x, ref_y, thumbnails(sha))
            rows.append({"sha256": sha, "stage": guess, "margin": share})
        new = pd.DataFrame(rows)
        # keep earlier predictions of other games in the same file (one file for all games)
        if os.path.exists(f"{OUT}/pred.csv"):
            old = pd.read_csv(f"{OUT}/pred.csv")
            new = pd.concat([old[~old.sha256.isin(new.sha256)], new])
        new.to_csv(f"{OUT}/pred.csv", index=False)
        print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
