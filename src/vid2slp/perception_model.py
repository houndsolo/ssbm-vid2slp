"""
perception_model.py - the first perception network (milestone 3): video frames -> game state.

Input:  a clip of T consecutive video frames, uint8 [B, T, 3, H, W].
Output: for every frame and both players (slot 0 = lower port, slot 1 = higher port), the
        fields in perception_data (character, action, stocks as classes; facing, airborne as
        yes/no; x, y, percent, shield as numbers).

Architecture, in two steps:
  1. Per frame: ResNet-18 (pretrained on ImageNet, then trained further on our frames). We
     keep its spatial feature map (where things are on screen matters for position) and
     shrink it with a 1x1 convolution, flatten, and project to one d_model vector per frame.
  2. Across frames: a small transformer over the T frame vectors (with the same distance
     penalty "ALiBi" as milestone 2), so each frame's prediction can use motion and the
     frames around it (an action state is often only clear from movement).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from vid2slp.id_model import Block
from vid2slp.perception_data import BIN_FIELDS, CAT_FIELDS, REG_FIELDS

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Perception(nn.Module):
    def __init__(self, size=(384, 320), d_model=512, layers=4, heads=8, dropout=0.1, max_len=64, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = torchvision.models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])   # drop avgpool + classifier
        fh, fw = (size[1] + 31) // 32, (size[0] + 31) // 32                # feature map size
        self.reduce = nn.Conv2d(512, 64, 1)
        self.frame_proj = nn.Sequential(nn.Linear(64 * fh * fw, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList(Block(d_model, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        # one head per field; each predicts both players at once
        self.cat_heads = nn.ModuleDict({k: nn.Linear(d_model, 2 * n) for k, n in CAT_FIELDS.items()})
        self.bin_head = nn.Linear(d_model, 2 * len(BIN_FIELDS))
        self.reg_head = nn.Linear(d_model, 2 * len(REG_FIELDS))
        slopes = torch.tensor([2 ** (-8 * (h + 1) / heads) for h in range(heads)])
        dist = torch.arange(max_len)[None, :] - torch.arange(max_len)[:, None]
        self.register_buffer("bias", (-slopes[:, None, None] * dist.abs()[None]).unsqueeze(0), persistent=False)
        self.register_buffer("mean", IMAGENET_MEAN, persistent=False)
        self.register_buffer("std", IMAGENET_STD, persistent=False)

    def forward(self, video):
        b, t = video.shape[:2]
        x = video.flatten(0, 1).float() / 255.0                 # [B*T, 3, H, W]
        x = (x - self.mean) / self.std
        f = self.reduce(self.backbone(x))                        # [B*T, 64, fh, fw]
        f = self.frame_proj(f.flatten(1)).view(b, t, -1)         # [B, T, d]
        bias = self.bias[..., :t, :t]
        for blk in self.blocks:
            f = blk(f, bias)
        f = self.norm(f)
        out = {k: h(f).view(b, t, 2, -1) for k, h in self.cat_heads.items()}   # [B, T, 2, classes]
        out["bin"] = self.bin_head(f).view(b, t, 2, len(BIN_FIELDS))
        out["reg"] = self.reg_head(f).view(b, t, 2, len(REG_FIELDS))
        return out


def loss_fn(out, batch):
    """Sum of per-field losses over labelled frames (batch["valid"]). Returns (loss, parts)."""
    valid = batch["valid"]                                     # [B, T]
    parts = {}
    for k in CAT_FIELDS:
        logits = out[k][valid].float()                         # [N, 2, classes]
        parts[k] = F.cross_entropy(logits.flatten(0, 1), batch[k][valid].flatten())
    tgt_bin = torch.stack([batch[k] for k in BIN_FIELDS], -1)[valid]     # [N, 2, nb]
    parts["bin"] = F.binary_cross_entropy_with_logits(out["bin"][valid].float(), tgt_bin)
    tgt_reg = torch.stack([batch[k] for k in REG_FIELDS], -1)[valid]
    parts["reg"] = 10 * F.smooth_l1_loss(out["reg"][valid].float(), tgt_reg, beta=0.05)
    return sum(parts.values()), {k: v.item() for k, v in parts.items()}
