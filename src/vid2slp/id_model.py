"""
id_model.py - the inverse-dynamics network (milestone 2): game states -> controller inputs.

Input:  a window of L consecutive frames. For every frame, the state of "self" (the player
        whose inputs we predict) and "opponent" (numeric + categorical features from id_data).
Output: for every frame in the window, self's inputs: 4 stick axes (161 classes each),
        trigger (141 classes) and 5 buttons (each pressed / not pressed).

Architecture: each frame becomes one vector ("token"); a transformer encoder lets every
frame look at every other frame in the window (past AND future: inputs at frame t show up
in the state at t, t+1, ...). Attention gets a penalty that grows with distance between
frames (ALiBi), so the model prefers nearby frames and does not need position embeddings.
Uses F.scaled_dot_product_attention (V100: no flash-attn 2).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vid2slp.id_data import AXIS_CLASSES, BUTTONS, CATEGORICAL, NUM_FEATURES, TRIGGER_CLASSES

# embedding width for each categorical feature (same order as id_data.CATEGORICAL)
CAT_DIMS = {"action": 64, "character": 16, "ground": 16, "hurtbox_state": 4,
            "l_cancel": 4, "last_attack_landed": 16, "stage": 8}


class FrameEncoder(nn.Module):
    """Turns [B, L, 2 players, features] into one d_model vector per frame."""

    def __init__(self, d_model, num_features=NUM_FEATURES):
        super().__init__()
        self.embeds = nn.ModuleList(nn.Embedding(size, CAT_DIMS[name]) for name, size in CATEGORICAL)
        per_player = num_features + sum(CAT_DIMS[name] for name, _ in CATEGORICAL)
        self.proj = nn.Sequential(nn.Linear(2 * per_player, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, num, cat):
        # num [B, L, 2, NUM], cat [B, L, 2, CAT]; index 0 = self, 1 = opponent
        parts = [num] + [emb(cat[..., i].long()) for i, emb in enumerate(self.embeds)]
        x = torch.cat(parts, dim=-1)              # [B, L, 2, per_player]
        return self.proj(x.flatten(2))            # [B, L, d_model]


class Block(nn.Module):
    """Pre-norm transformer block with a given additive attention bias."""

    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.drop = nn.Dropout(dropout)
        self.p = dropout

    def forward(self, x, bias):
        b, length, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, length, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype),
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(b, length, d)))
        return x + self.drop(self.ff(self.norm2(x)))


class InverseDynamics(nn.Module):
    def __init__(self, d_model=384, layers=8, heads=8, dropout=0.1, window=256,
                 num_features=NUM_FEATURES, fp32_encoder=False, zero_features=(), zero_cats=()):
        super().__init__()
        # zero_features: column numbers of numeric features the model must NOT see (they are
        # multiplied by 0, for both players). Used for ablations, e.g. "does damage % help?".
        mask = torch.ones(num_features)
        mask[list(zero_features)] = 0.0
        self.register_buffer("feature_mask", mask, persistent=False)
        # zero_cats: column numbers of CATEGORICAL features the model must not see (set to 0).
        # Used for the "perceivable features only" model (Stage C): the video model cannot see
        # e.g. the ground id or the hurtbox state.
        self.zero_cats = list(zero_cats)
        # fp32_encoder: run the first (input) layers in full precision. In fp16, small values
        # like a 0.01 velocity change next to a 2.5 position lose precision.
        self.fp32_encoder = fp32_encoder
        self.encoder = FrameEncoder(d_model, num_features)
        self.blocks = nn.ModuleList(Block(d_model, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.axis_heads = nn.ModuleList(nn.Linear(d_model, AXIS_CLASSES) for _ in range(4))
        self.trigger_head = nn.Linear(d_model, TRIGGER_CLASSES)
        self.button_head = nn.Linear(d_model, len(BUTTONS))
        # ALiBi: head h penalises attention to a frame at distance d by slope_h * d.
        slopes = torch.tensor([2 ** (-8 * (h + 1) / heads) for h in range(heads)])
        dist = torch.arange(window)[None, :] - torch.arange(window)[:, None]
        self.register_buffer("bias", (-slopes[:, None, None] * dist.abs()[None]).unsqueeze(0), persistent=False)

    def forward(self, num, cat):
        num = num * self.feature_mask.to(num.dtype)
        if self.zero_cats:
            cat = cat.clone()
            cat[..., self.zero_cats] = 0
        if self.fp32_encoder:
            with torch.autocast("cuda", enabled=False):
                x = self.encoder(num.float(), cat)
        else:
            x = self.encoder(num, cat)
        length = x.shape[1]
        bias = self.bias[..., :length, :length]
        for blk in self.blocks:
            x = blk(x, bias)
        x = self.norm(x)
        return {
            "axes": [h(x) for h in self.axis_heads],   # 4 x [B, L, 161]
            "trigger": self.trigger_head(x),            # [B, L, 141]
            "buttons": self.button_head(x),             # [B, L, 5] logits
        }


def loss_fn(out, tgt, btn):
    """tgt [B, L, 5] int (4 axes + trigger), btn [B, L] uint8 bitmask -> scalar loss and parts."""
    parts = {}
    for i, logits in enumerate(out["axes"]):
        parts[f"axis{i}"] = F.cross_entropy(logits.float().flatten(0, 1), tgt[..., i].long().flatten())
    parts["trigger"] = F.cross_entropy(out["trigger"].float().flatten(0, 1), tgt[..., 4].long().flatten())
    bits = torch.stack([(btn.long() >> i) & 1 for i in range(len(BUTTONS))], dim=-1).float()
    parts["buttons"] = F.binary_cross_entropy_with_logits(out["buttons"].float(), bits) * len(BUTTONS)
    return sum(parts.values()), {k: v.item() for k, v in parts.items()}


def decode(out, threshold=0.5):
    """Model outputs -> (tgt [B, L, 5] int16, btn [B, L] uint8) using the most likely values."""
    tgt = torch.stack([o.argmax(-1) for o in out["axes"]] + [out["trigger"].argmax(-1)], dim=-1)
    pressed = torch.sigmoid(out["buttons"].float()) > threshold
    btn = sum(pressed[..., i].to(torch.uint8) << i for i in range(len(BUTTONS)))
    return tgt.to(torch.int16), btn.to(torch.uint8)


def window_jobs(offsets, window):
    """Cut every game into half-overlapping windows.

    Returns a list of (window start row, first row to keep, end row to keep). Every row of
    every game is kept from exactly one window (tests/test_id_model.py checks this), and
    apart from game edges it sits in the middle half of that window.
    """
    half, quarter = window // 2, window // 4
    jobs = []
    for g in range(len(offsets) - 1):
        a, b = int(offsets[g]), int(offsets[g + 1])
        starts = list(range(a, b - window + 1, half))
        if starts[-1] != b - window:
            starts.append(b - window)
        for j, s in enumerate(starts):
            keep_from = s if j == 0 else s + quarter
            keep_to = b if j == len(starts) - 1 else min(s + quarter + half, starts[j + 1] + quarter)
            jobs.append((s, keep_from, keep_to))
    return jobs


@torch.no_grad()
def predict_split(model, split, device, window=256, batch=128):
    """Predict inputs for every frame of every game in a split, for both players.

    Each game is cut into windows that overlap by half. Every frame takes its prediction
    from a window where it sits in the middle half, so it always has >= window/4 frames of
    context on both sides (except at the very start/end of a game).
    Returns (pred_tgt [N, 2, 5] int16, pred_btn [N, 2] uint8).
    """
    model.eval()
    num_all, cat_all, offsets = split["num"], split["cat"], split["offsets"]
    n = offsets[-1]
    pred_tgt = np.zeros((n, 2, 5), np.int16)
    pred_btn = np.zeros((n, 2), np.uint8)

    jobs = window_jobs(offsets, window)

    for self_port in (0, 1):
        order = [self_port, 1 - self_port]
        for i in range(0, len(jobs), batch):
            chunk = jobs[i:i + batch]
            idx = np.stack([np.arange(s, s + window) for s, _, _ in chunk])
            num = torch.from_numpy(np.asarray(num_all[idx.ravel()])[:, order]).to(device)
            cat = torch.from_numpy(np.asarray(cat_all[idx.ravel()])[:, order]).to(device)
            num = num.view(len(chunk), window, 2, -1)
            cat = cat.view(len(chunk), window, 2, -1)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(num, cat)
            tgt, btn = decode(out)
            tgt, btn = tgt.cpu().numpy(), btn.cpu().numpy()
            for row, (s, keep_from, keep_to) in enumerate(chunk):
                pred_tgt[keep_from:keep_to, self_port] = tgt[row, keep_from - s:keep_to - s]
                pred_btn[keep_from:keep_to, self_port] = btn[row, keep_from - s:keep_to - s]
    return pred_tgt, pred_btn


@torch.no_grad()
def predict_probs(model, num_all, cat_all, device, window=256, batch=64):
    """Like predict_split for ONE game, but returns the model's probabilities instead of only
    the most likely values (Stage C's closed loop samples alternative inputs from them).

    num_all [T, 2, F], cat_all [T, 2, C] (arrays_from_states). Returns float16 arrays:
        axes [T, 2, 4, 161]   probability of each stick / c-stick value (-80..80)
        trigger [T, 2, 141]   probability of each trigger value (0..140)
        buttons [T, 2, len(BUTTONS)]   probability that each logical button bit is set
    """
    model.eval()
    n = len(num_all)
    axes = np.zeros((n, 2, 4, 161), np.float16)
    trigger = np.zeros((n, 2, 141), np.float16)
    buttons = np.zeros((n, 2, len(BUTTONS)), np.float16)
    jobs = window_jobs(np.array([0, n]), window)
    for self_port in (0, 1):
        order = [self_port, 1 - self_port]
        for i in range(0, len(jobs), batch):
            chunk = jobs[i:i + batch]
            idx = np.stack([np.arange(s, s + window) for s, _, _ in chunk])
            num = torch.from_numpy(np.asarray(num_all[idx.ravel()])[:, order]).to(device)
            cat = torch.from_numpy(np.asarray(cat_all[idx.ravel()])[:, order]).to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(num.view(len(chunk), window, 2, -1), cat.view(len(chunk), window, 2, -1))
            a = torch.stack([o.float().softmax(-1) for o in out["axes"]], dim=2).cpu().numpy()
            t = out["trigger"].float().softmax(-1).cpu().numpy()
            b = torch.sigmoid(out["buttons"].float()).cpu().numpy()
            for row, (s, keep_from, keep_to) in enumerate(chunk):
                axes[keep_from:keep_to, self_port] = a[row, keep_from - s:keep_to - s]
                trigger[keep_from:keep_to, self_port] = t[row, keep_from - s:keep_to - s]
                buttons[keep_from:keep_to, self_port] = b[row, keep_from - s:keep_to - s]
    return {"axes": axes, "trigger": trigger, "buttons": buttons}


def build_from_config(cfg, num_features):
    """Recreate a model from a checkpoint's saved config (older configs lack the new keys)."""
    return InverseDynamics(cfg["d_model"], cfg["layers"], cfg["heads"], 0.0, cfg["window"],
                           num_features=num_features, fp32_encoder=cfg.get("fp32_encoder", False),
                           zero_features=cfg.get("zero_feature_columns", ()),
                           zero_cats=cfg.get("zero_cat_columns", ()))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def lr_at(step, total, peak, warmup):
    """Linear warm-up, then cosine decay to 5% of peak."""
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress)))
