from __future__ import annotations

import torch
import torch.nn.functional as F


def _invsize_weights(target: torch.Tensor, power: float, dims: tuple[int, ...]) -> torch.Tensor:
    """Per-slice weights ∝ (lesion pixel count)^-power, normalised to sum 1 (A4b H5).

    Empty-target slices have no lesion size, so they cannot be weighted by one.  They receive the MEAN
    of the positive weights, which keeps their aggregate influence exactly what it is under the plain
    "sample" reduction (each empty slice = one average slice) while the positive slices are re-weighted
    among themselves.  With no positive slice in the batch every weight is equal (= plain mean).
    """
    n = target.sum(dim=dims)
    pos = n > 0
    w = torch.where(pos, n.clamp(min=1.0) ** (-power), torch.zeros_like(n))
    fill = w[pos].mean() if bool(pos.any()) else torch.ones_like(n[0])
    w = torch.where(pos, w, fill)
    return w / w.sum()


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0,
                   reduction: str = "batch", invsize_power: float = 0.5) -> torch.Tensor:
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
      "sample_invsize" (A4b H5) -- like "sample" but the per-slice Dice terms are weighted by
                (lesion pixel count)^-`invsize_power`, so a 3-voxel lacune outweighs a 3,000-voxel MCA
                infarct instead of being invisible next to it.
    """
    p = torch.sigmoid(logits)
    if reduction == "batch":
        inter = (p * target).sum()
        denom = p.sum() + target.sum()
        return 1.0 - (2.0 * inter + smooth) / (denom + smooth)
    if reduction in ("sample", "sample_invsize"):
        dims = tuple(range(1, p.ndim))
        inter = (p * target).sum(dim=dims)
        denom = p.sum(dim=dims) + target.sum(dim=dims)
        per = 1.0 - (2.0 * inter + smooth) / (denom + smooth)
        if reduction == "sample":
            return per.mean()
        return (per * _invsize_weights(target, invsize_power, dims)).sum()
    raise ValueError(f"unknown reduction {reduction!r}")


def tversky_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.7, beta: float = 0.3,
                 gamma: float = 1.0, smooth: float = 1.0, reduction: str = "batch",
                 invsize_power: float = 0.5) -> torch.Tensor:
    """(Focal) Tversky loss: 1 - TP/(TP + alpha*FN + beta*FP), raised to the power `gamma`.

    The index is written as (2TP + smooth) / (2TP + 2*alpha*FN + 2*beta*FP + smooth), i.e. numerator and
    denominator of the usual Tversky index are both doubled, so that alpha = beta = 0.5 reproduces
    `soft_dice_loss` EXACTLY (same smoothing constant) and the two are directly comparable.  alpha > beta penalises FALSE NEGATIVES harder, which is
    the theoretically right direction for the < 2 mL band, where the failure mode measured on the val
    split is under-segmentation (A4b H5).  gamma > 1 (Abraham & Khan's focal Tversky) additionally
    concentrates the gradient on the badly-segmented cases; gamma = 1 is the plain Tversky.
    `reduction` follows `soft_dice_loss`.
    """
    p = torch.sigmoid(logits)
    if reduction == "batch":
        tp = (p * target).sum()
        fn = ((1.0 - p) * target).sum()
        fp = (p * (1.0 - target)).sum()
        ti = (2.0 * tp + smooth) / (2.0 * tp + 2.0 * alpha * fn + 2.0 * beta * fp + smooth)
        return (1.0 - ti) ** gamma
    if reduction in ("sample", "sample_invsize"):
        dims = tuple(range(1, p.ndim))
        tp = (p * target).sum(dim=dims)
        fn = ((1.0 - p) * target).sum(dim=dims)
        fp = (p * (1.0 - target)).sum(dim=dims)
        per = (1.0 - (2.0 * tp + smooth) / (2.0 * tp + 2.0 * alpha * fn + 2.0 * beta * fp + smooth)) ** gamma
        if reduction == "sample":
            return per.mean()
        return (per * _invsize_weights(target, invsize_power, dims)).sum()
    raise ValueError(f"unknown reduction {reduction!r}")


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 1.0, pos_weight: float | None = None,
                  dice_reduction: str = "batch", region_loss: str = "dice", tversky_alpha: float = 0.7,
                  tversky_beta: float = 0.3, tversky_gamma: float = 1.0, invsize_power: float = 0.5):
    """BCE + `dice_weight` * (region term).

    `region_loss` selects the region term: "dice" (default, the shipped behaviour) or "tversky"
    (A4b H5).  The returned parts dict keeps the key "dice" for both so the JSONL log schema and every
    downstream reader stay unchanged.
    """
    pw = torch.tensor(pos_weight, device=logits.device) if pos_weight else None
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
    if region_loss == "dice":
        dice = soft_dice_loss(logits, target, reduction=dice_reduction, invsize_power=invsize_power)
    elif region_loss == "tversky":
        dice = tversky_loss(logits, target, alpha=tversky_alpha, beta=tversky_beta, gamma=tversky_gamma,
                            reduction=dice_reduction, invsize_power=invsize_power)
    else:
        raise ValueError(f"unknown region_loss {region_loss!r}")
    return bce + dice_weight * dice, {"bce": float(bce.detach()), "dice": float(dice.detach())}
