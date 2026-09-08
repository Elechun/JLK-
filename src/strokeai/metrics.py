"""Evaluation metrics. This is the ONLY place metrics are defined (see CLAUDE.md).

Conventions
-----------
* Segmentation metrics are computed per *subject* on the full 3-D stack (never per slice averaged),
  because that is what a clinician / the product reports.
* Dice on a subject whose ground-truth mask is empty is undefined. We therefore report
  - `dice_pos`: mean Dice over subjects with a non-empty GT mask (primary), and
  - subject-level detection (any predicted voxel vs any GT voxel) separately.
  A both-empty subject is *not* silently counted as Dice = 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def dice_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice = 2|P∩G| / (|P|+|G|). Returns NaN when both are empty (undefined)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def volume_ml(mask: np.ndarray, voxel_volume_mm3: float) -> float:
    return float(mask.astype(bool).sum() * voxel_volume_mm3 / 1000.0)


@dataclass
class VolumeAgreement:
    n: int
    pearson_r: float
    mean_diff_ml: float  # pred - gt
    loa_low_ml: float  # Bland-Altman 95% limits of agreement
    loa_high_ml: float
    mae_ml: float
    icc21: float  # ICC(2,1) absolute agreement, two-way random, single measure


def icc21(x: np.ndarray, y: np.ndarray) -> float:
    """ICC(2,1) for two raters (Shrout & Fleiss). x, y: shape (n,)."""
    data = np.stack([x, y], axis=1).astype(float)
    n, k = data.shape
    grand = data.mean()
    ms_r = k * ((data.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((data.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = data - data.mean(axis=1, keepdims=True) - data.mean(axis=0, keepdims=True) + grand
    ms_e = (resid**2).sum() / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / denom) if denom != 0 else float("nan")


def volume_agreement(pred_ml: np.ndarray, gt_ml: np.ndarray) -> VolumeAgreement:
    pred_ml = np.asarray(pred_ml, float)
    gt_ml = np.asarray(gt_ml, float)
    diff = pred_ml - gt_ml
    r = float(np.corrcoef(pred_ml, gt_ml)[0, 1]) if len(pred_ml) > 1 and gt_ml.std() > 0 and pred_ml.std() > 0 else float("nan")
    return VolumeAgreement(
        n=len(pred_ml),
        pearson_r=r,
        mean_diff_ml=float(diff.mean()),
        loa_low_ml=float(diff.mean() - 1.96 * diff.std(ddof=1)) if len(diff) > 1 else float("nan"),
        loa_high_ml=float(diff.mean() + 1.96 * diff.std(ddof=1)) if len(diff) > 1 else float("nan"),
        mae_ml=float(np.abs(diff).mean()),
        icc21=icc21(pred_ml, gt_ml) if len(pred_ml) > 1 else float("nan"),
    )


def bootstrap_ci(values: np.ndarray, stat=np.nanmean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of `stat` over subjects (resampling subjects, not voxels)."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.array([stat(values[rng.integers(0, n, n)]) for _ in range(n_boot)])
    return float(stat(values)), float(np.nanpercentile(boots, 100 * alpha / 2)), float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))


def segmentation_summary(per_subject: list[dict], seed: int = 0) -> dict:
    """per_subject: dicts with keys dice (nan if GT empty), gt_ml, pred_ml, gt_pos (bool), pred_pos (bool)."""
    dices = np.array([d["dice"] for d in per_subject], float)
    gt_pos = np.array([d["gt_pos"] for d in per_subject], bool)
    pred_pos = np.array([d["pred_pos"] for d in per_subject], bool)
    gt_ml = np.array([d["gt_ml"] for d in per_subject], float)
    pred_ml = np.array([d["pred_ml"] for d in per_subject], float)

    pos = gt_pos
    mean_dice, lo, hi = bootstrap_ci(dices[pos], np.nanmean, seed=seed)
    med = float(np.nanmedian(dices[pos])) if pos.any() else float("nan")
    tp = int((gt_pos & pred_pos).sum())
    fn = int((gt_pos & ~pred_pos).sum())
    fp = int((~gt_pos & pred_pos).sum())
    tn = int((~gt_pos & ~pred_pos).sum())
    va = volume_agreement(pred_ml[pos], gt_ml[pos]) if pos.sum() > 1 else None
    out = {
        "n_subjects": len(per_subject),
        "n_gt_positive": int(pos.sum()),
        "dice_pos_mean": mean_dice,
        "dice_pos_ci95": [lo, hi],
        "dice_pos_median": med,
        "detection_sensitivity": tp / max(tp + fn, 1),
        "detection_specificity": tn / max(tn + fp, 1) if (tn + fp) > 0 else float("nan"),
        "detection_confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }
    if va:
        out["volume"] = va.__dict__
    # Dice stratified by lesion size (small lesions are the clinically hard ones: lacunes < 1.5 cm ~ < 2 mL)
    bins = [(0, 2), (2, 10), (10, 50), (50, np.inf)]
    out["dice_by_gt_volume_ml"] = {
        f"[{a},{b})": {"n": int(m.sum()), "dice_mean": float(np.nanmean(dices[m])) if m.any() else float("nan")}
        for a, b in bins
        for m in [pos & (gt_ml >= a) & (gt_ml < b)]
    }
    return out


def multiclass_auc(y_true: np.ndarray, proba: np.ndarray, classes: list) -> dict:
    """Macro one-vs-rest ROC-AUC plus per-class AUC. Classes absent in y_true get NaN."""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    per = {}
    for j, c in enumerate(classes):
        yb = (y_true == c).astype(int)
        per[str(c)] = float(roc_auc_score(yb, proba[:, j])) if 0 < yb.sum() < len(yb) else float("nan")
    return {"macro_auc_ovr": float(np.nanmean(list(per.values()))), "per_class_auc": per}
