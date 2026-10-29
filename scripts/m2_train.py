"""
m2_train.py - train the inverse-dynamics model (milestone 2) on the V100.

What it does:
  1. Loads the training arrays (data/m2/train_*.npy) entirely onto the GPU (~4 GB),
     so batches are assembled on the GPU with no data-loading bottleneck.
  2. Each training example = a random 256-frame window from a random training game,
     seen from a random player's point of view ("self" = the player whose inputs we predict).
  3. fp16 autocast + GradScaler (V100 has no bf16), AdamW, warm-up then cosine decay.
  4. Every --eval-every steps: predict ALL validation games and score them with
     src/vid2slp/id_metrics.py. The checkpoint with the best validation
     "frame_all_correct" is kept as best.pt. The TEST split is never touched here.

Outputs in data/m2/runs/<name>/: config.json, metrics.jsonl, best.pt, last.pt, train.log.

Run (inside tmux):
    .venv/bin/python scripts/m2_train.py --name base_v1
    .venv/bin/python scripts/m2_train.py --name base_v1 --resume     # continue after a crash
Fine-tune an earlier run on another dataset (e.g. perceived states, scripts/m2_build_perceived.py):
    .venv/bin/python scripts/m2_train.py --name perc_v1 --data data/m2_perc --init visible_v1 ...
Quick smoke test:
    .venv/bin/python scripts/m2_train.py --name smoke --steps 300 --eval-every 300
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from vid2slp.id_data import CATEGORICAL, FEATURE_SETS, load_split
from vid2slp.id_metrics import format_scores, score
from vid2slp.id_model import InverseDynamics, count_parameters, loss_fn, lr_at, predict_split

DATA = "data/m2"


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=64, help="windows per step")
    ap.add_argument("--window", type=int, default=256, help="frames per window")
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default="", help="start from the weights of this run's best.pt (fine-tune; "
                    "model options must match; optimizer starts fresh)")
    ap.add_argument("--data", default="data/m2", help="dataset folder (data/m2 = v1 features, data/m2_v2 = v2)")
    ap.add_argument("--fp32-encoder", action="store_true", help="input layers in full precision")
    ap.add_argument("--zero-features", default="",
                    help="comma-separated numeric features to hide from the model, e.g. percent (ablation)")
    ap.add_argument("--zero-cats", default="",
                    help="comma-separated categorical features to hide, e.g. ground,hurtbox_state")
    return ap.parse_args()


def main():
    args = get_args()
    run_dir = f"{DATA}/runs/{args.name}"
    os.makedirs(run_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")  # the V100 (see scripts/check_env.py)
    log = open(f"{run_dir}/train.log", "a")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    # ---- data -------------------------------------------------------------------------
    train = load_split(args.data, "train", mmap=False)
    val = load_split(args.data, "validation", mmap=False)
    say(f"train frames {train['offsets'][-1]:,}  validation frames {val['offsets'][-1]:,}")
    g_num = torch.from_numpy(train["num"]).to(device)
    g_cat = torch.from_numpy(train["cat"]).to(device)
    g_tgt = torch.from_numpy(train["tgt"]).to(device)
    g_btn = torch.from_numpy(train["btn"]).to(device)
    # every row where a full window fits inside one game is a valid window start
    off = train["offsets"]
    starts = np.concatenate([np.arange(a, b - args.window + 1) for a, b in zip(off[:-1], off[1:])])
    g_starts = torch.from_numpy(starts).to(device)
    frame_offsets = torch.arange(args.window, device=device)
    del train
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    def batch():
        pick = g_starts[torch.randint(len(g_starts), (args.batch,), device=device, generator=gen)]
        rows = pick[:, None] + frame_offsets[None]                       # [B, L]
        self_port = torch.randint(2, (args.batch,), device=device, generator=gen)
        order = torch.stack([self_port, 1 - self_port], dim=1)            # [B, 2]
        o = order[:, None, :, None]                                      # broadcast over L and features
        num = torch.gather(g_num[rows], 2, o.expand(-1, args.window, -1, g_num.shape[-1]))
        cat = torch.gather(g_cat[rows], 2, o.expand(-1, args.window, -1, g_cat.shape[-1]))
        tgt = torch.gather(g_tgt[rows], 2, o[..., :1, :].expand(-1, args.window, 1, 5))[:, :, 0]
        btn = torch.gather(g_btn[rows], 2, order[:, None, :1].expand(-1, args.window, 1))[:, :, 0]
        return num, cat, tgt, btn

    # ---- model ------------------------------------------------------------------------
    # Features to hide (ablation): names -> column numbers in this dataset's feature set.
    # (data/m2 predates features.txt and uses the v1 features)
    version = open(f"{args.data}/features.txt").read().strip() if os.path.exists(f"{args.data}/features.txt") else "v1"
    feature_names = [c for c, _ in FEATURE_SETS[version]]
    args.zero_feature_columns = [feature_names.index(f) for f in args.zero_features.split(",") if f]
    cat_names = [c for c, _ in CATEGORICAL]
    args.zero_cat_columns = [cat_names.index(f) for f in args.zero_cats.split(",") if f]
    model = InverseDynamics(args.d_model, args.layers, args.heads, args.dropout, args.window,
                            num_features=g_num.shape[-1], fp32_encoder=args.fp32_encoder,
                            zero_features=args.zero_feature_columns,
                            zero_cats=args.zero_cat_columns).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    scaler = torch.amp.GradScaler("cuda")
    step, best = 0, -1.0
    if args.resume and os.path.exists(f"{run_dir}/last.pt"):
        ck = torch.load(f"{run_dir}/last.pt", map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, best = ck["step"], ck["best"]
        gen.manual_seed(args.seed + step)  # fresh but reproducible sampling after resume
        say(f"resumed from step {step}, best so far {best:.4f}")
    else:
        if args.init:
            ck = torch.load(f"{DATA}/runs/{args.init}/best.pt", map_location=device)
            model.load_state_dict(ck["model"])
            say(f"initialized from {args.init} best.pt (step {ck['step']})")
        json.dump(vars(args) | {"parameters": count_parameters(model), "torch": torch.__version__},
                  open(f"{run_dir}/config.json", "w"), indent=1)
    say(f"model parameters: {count_parameters(model):,}")

    def save(path, metrics=None):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "best": best, "config": vars(args), "val_metrics": metrics}, path)

    # ---- train loop -------------------------------------------------------------------
    model.train()
    t0, seen = time.time(), 0
    running = {}
    while step < args.steps:
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        num, cat, tgt, btn = batch()
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(num, cat)
        loss, parts = loss_fn(out, tgt, btn)
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

        if step % 200 == 0:
            rate = seen / (time.time() - t0)
            say(f"step {step}/{args.steps}  loss {sum(running.values()) / seen:.4f}  "
                + " ".join(f"{k} {v / seen:.3f}" for k, v in running.items())
                + f"  lr {opt.param_groups[0]['lr']:.2e}  {rate:.1f} steps/s")
            t0, seen, running = time.time(), 0, {}

        if step % args.eval_every == 0 or step == args.steps:
            te = time.time()
            pred_tgt, pred_btn = predict_split(model, val, device, args.window)
            m = score(val["tgt"], pred_tgt, val["btn"], pred_btn, val["offsets"])
            model.train()
            improved = m["frame_all_correct"] > best
            if improved:
                best = m["frame_all_correct"]
                save(f"{run_dir}/best.pt", m)
            save(f"{run_dir}/last.pt", m)
            with open(f"{run_dir}/metrics.jsonl", "a") as f:
                f.write(json.dumps({"step": step, **m}) + "\n")
            say(f"VALIDATION step {step}: {format_scores(m)}{'  (best)' if improved else ''}  "
                f"[{time.time() - te:.0f}s]")
            t0, seen, running = time.time(), 0, {}

    say(f"done. best validation frame_all_correct {best:.4f}")


if __name__ == "__main__":
    main()
