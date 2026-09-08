from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0,
                   reduction: str = "batch") -> torch.Tensor:
    """Soft Dice loss.  smooth=1 keeps it finite and bounded in [0, 1].

    reduction
      "batch"  (default) -- one Dice over the *whole batch*: intersection and denominator are summed
                across every slice.  Empty-target slices then produce no degenerate 0/0 term, and their
                false positives still enter the denominator, so they are penalised.  The price is that
                the value depends on the batch composition and that the term is a *micro* Dice: one
                large lesion in the batch outweighs many small ones.  A4 measured this on real batches
                (150 draws, fixed predictor): positive pixels per batch span p5 3,331 - p95 8,754 at
                batch_size 64 (2.6x; 4.1x at batch_size 32), the loss correlates -0.9996 with that
                count, and its batch-to-batch SD is 0.0032 at bs 64 vs 0.0038 at bs 32.
      "sample" -- mean of the per-slice Dice.  Every slice counts equally (small lesions get the same
                weight as large ones) but an empty-target slice contributes 1 - smooth/(p.sum()+smooth),
                i.e. it degenerates into a pure false-positive penalty.
    """
    p = torch.sigmoid(logits)
    if reduction == "batch":
        inter = (p * target).sum()
        denom = p.sum() + target.sum()
        return 1.0 - (2.0 * inter + smooth) / (denom + smooth)
    if reduction == "sample":
        dims = tuple(range(1, p.ndim))
        inter = (p * target).sum(dim=dims)
        denom = p.sum(dim=dims) + target.sum(dim=dims)
        return (1.0 - (2.0 * inter + smooth) / (denom + smooth)).mean()
    raise ValueError(f"unknown reduction {reduction!r}")


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 1.0, pos_weight: float | None = None,
                  dice_reduction: str = "batch"):
    pw = torch.tensor(pos_weight, device=logits.device) if pos_weight else None
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
    dice = soft_dice_loss(logits, target, reduction=dice_reduction)
    return bce + dice_weight * dice, {"bce": float(bce.detach()), "dice": float(dice.detach())}
