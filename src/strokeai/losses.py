from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Batch-level soft Dice (sums over the whole batch, not per sample) so that slices with an empty
    target do not produce a degenerate 0/0 term. smooth=1 keeps the loss finite and bounded in [0, 1]."""
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    denom = p.sum() + target.sum()
    return 1.0 - (2.0 * inter + smooth) / (denom + smooth)


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 1.0, pos_weight: float | None = None):
    pw = torch.tensor(pos_weight, device=logits.device) if pos_weight else None
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
    dice = soft_dice_loss(logits, target)
    return bce + dice_weight * dice, {"bce": float(bce.detach()), "dice": float(dice.detach())}
