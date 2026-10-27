"""
m3_train.py - train the first perception model (milestone 3) on rendered replays.

Data: rendered games in data/m3/renders (scripts/m3_render.py). Train on games of the TRAIN
split, check on VALIDATION games; the test games are never touched here.
Each training example is a random clip of --clip consecutive video frames from a random
training game (decoded on the fly by DataLoader workers; src/vid2slp/perception_data.py).

Every --eval-every steps the model predicts a fixed set of validation clips (evenly spaced
through every validation game) and we print:
    character / action / stocks / facing / airborne accuracy,
    x and y error (world units; the stage is roughly -250..250 wide),
    percent error (and how often the rounded % is exactly right), shield error.
The checkpoint with the best validation ACTION accuracy is kept as best.pt.

Outputs in data/m3/runs/<name>/: config.json, metrics.jsonl, best.pt, last.pt, train.log.

Run (inside tmux):
    .venv/bin/python scripts/m3_train.py --name percep_v1
Quick smoke test:
    .venv/bin/python scripts/m3_train.py --name smoke --steps 50 --eval-every 50 --val-per-game 2
"""

import argparse
import glob
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from vid2slp.id_model import lr_at
from vid2slp.perception_data import BIN_FIELDS, CAT_FIELDS, REG_FIELDS, Game, RandomClips, to_tensors
from vid2slp.perception_model import Perception, loss_fn

OUT = "data/m3/runs"


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=8, help="clips per step")
    ap.add_argument("--clip", type=int, default=32, help="frames per clip")
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--val-per-game", type=int, default=20, help="validation clips per validation game")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def rendered_games(split):
    """sha256 of every finished render that belongs to `split`, leaving out games whose playback
    ended early (desynced, see scripts/m3_frame_maps.py) and games whose on-screen timer did
    not start at 8 minutes (owner's choice 2026-10-27: only standard 8-minute games)."""
    manifest = pd.read_csv("data/parsed/manifest.csv").set_index("sha256")
    done = []
    for p in glob.glob("data/m3/renders/*/info.json"):
        info = json.load(open(p))
        if not info.get("playback_ended_early", False) and info["timer_start_seconds"] == 480:
            done.append(os.path.basename(os.path.dirname(p)))
    return sorted(s for s in done if s in manifest.index and manifest.loc[s, "split"] == split)


class FixedClips(torch.utils.data.Dataset):
    """Validation clips: `per_game` clips evenly spaced through each game (same every time)."""

    def __init__(self, shas, length, size, per_game):
        self.length, self.size = length, size
        self.games = {s: Game(s) for s in shas}
        self.items = []
        for s, g in self.games.items():
            for start in np.linspace(g.first_valid, len(g) - length, per_game).astype(int):
                self.items.append((s, int(start)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s, start = self.items[i]
        return to_tensors(*self.games[s].clip(start, self.length, self.size))


def evaluate(model, loader, device):
    """Predict all validation clips and return a dict of metrics."""
    model.eval()
    sums, n = {}, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(batch["video"])
            v = batch["valid"]
            m = {}
            for k in CAT_FIELDS:
                m[f"{k}_acc"] = (out[k][v].argmax(-1) == batch[k][v]).float()
            for j, k in enumerate(BIN_FIELDS):
                m[f"{k}_acc"] = ((out["bin"][v][..., j] > 0).float() == batch[k][v]).float()
            for j, (k, scale) in enumerate(REG_FIELDS.items()):
                err = (out["reg"][v][..., j].float() - batch[k][v]).abs() * scale
                m[f"{k}_mae"] = err
                if k == "percent":
                    pred = (out["reg"][v][..., j].float() * scale).round()
                    m["percent_exact"] = (pred == (batch[k][v] * scale).round()).float()
            for k, t in m.items():
                sums[k] = sums.get(k, 0.0) + t.sum().item()
            n += int(v.sum().item()) * 2   # labelled frames x 2 players
    model.train()
    return {k: s / max(n, 1) for k, s in sums.items()} | {"player_frames": n}


def fmt(m):
    keys = ["action_acc", "character_acc", "stocks_acc", "facing_acc", "airborne_acc",
            "x_mae", "y_mae", "percent_mae", "percent_exact", "shield_mae"]
    return "  ".join(f"{k} {m[k]:.4f}" if "acc" in k or "exact" in k else f"{k} {m[k]:.2f}" for k in keys)


def main():
    args = get_args()
    run_dir = f"{OUT}/{args.name}"
    os.makedirs(run_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    log = open(f"{run_dir}/train.log", "a")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    size = (args.width, args.height)
    train_shas, val_shas = rendered_games("train"), rendered_games("validation")
    say(f"games: {len(train_shas)} train, {len(val_shas)} validation")
    train_loader = torch.utils.data.DataLoader(
        RandomClips(train_shas, args.clip, size, args.seed), batch_size=args.batch,
        num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_set = FixedClips(val_shas, args.clip, size, args.val_per_game)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch, num_workers=args.workers)
    say(f"validation clips: {len(val_set)}")

    model = Perception(size=size, max_len=args.clip).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    step, best = 0, -1.0
    if args.resume and os.path.exists(f"{run_dir}/last.pt"):
        ck = torch.load(f"{run_dir}/last.pt", map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, best = ck["step"], ck["best"]
        say(f"resumed from step {step}, best so far {best:.4f}")
    else:
        json.dump(vars(args) | {"train_games": train_shas, "validation_games": val_shas, "torch": torch.__version__},
                  open(f"{run_dir}/config.json", "w"), indent=1)
    say(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def save(path, metrics=None):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "best": best, "config": vars(args), "val_metrics": metrics}, path)

    model.train()
    t0, seen, running = time.time(), 0, {}
    for batch in train_loader:
        if step >= args.steps:
            break
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(batch["video"])
        loss, parts = loss_fn(out, batch)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        seen += 1
        for k, v in parts.items():
            running[k] = running.get(k, 0.0) + v
        if step % 100 == 0:
            rate = seen / (time.time() - t0)
            say(f"step {step}/{args.steps}  loss {sum(running.values()) / seen:.4f}  "
                + " ".join(f"{k} {v / seen:.3f}" for k, v in running.items())
                + f"  lr {opt.param_groups[0]['lr']:.2e}  {rate:.2f} steps/s")
            t0, seen, running = time.time(), 0, {}
        if step % args.eval_every == 0 or step == args.steps:
            te = time.time()
            m = evaluate(model, val_loader, device)
            improved = m["action_acc"] > best
            if improved:
                best = m["action_acc"]
                save(f"{run_dir}/best.pt", m)
            save(f"{run_dir}/last.pt", m)
            with open(f"{run_dir}/metrics.jsonl", "a") as f:
                f.write(json.dumps({"step": step, **m}) + "\n")
            say(f"VALIDATION step {step}: {fmt(m)}{'  (best)' if improved else ''}  [{time.time() - te:.0f}s]")
            t0, seen, running = time.time(), 0, {}
    say(f"done. best validation action accuracy {best:.4f}")


if __name__ == "__main__":
    main()
