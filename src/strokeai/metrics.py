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
    """Dice = 2|P∩G| / (|P|+|G|). Returns NaN when both are empty (undefined).

    Shapes must match exactly: NumPy broadcasting would otherwise let a mis-aligned pair such as
    (1, 3) vs (3, 1) return Dice = 3.0 without complaint (A5a item 16).
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")
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
    # A3 2026-09-09: lesion volumes span 0.07-557 mL and the native voxel volume varies 5.9x across
    # subjects, so an mL-only error is dominated by the few largest lesions. Report the relative error
    # next to it (A2 hand-off item 5).
    mape_pct: float  # mean |pred-gt| / gt x 100
    median_abs_pct_err: float  # median |pred-gt| / gt x 100 (robust to the long tail)
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
        mape_pct=float(np.mean(np.abs(diff[gt_ml > 0]) / gt_ml[gt_ml > 0]) * 100) if (gt_ml > 0).any() else float("nan"),
        median_abs_pct_err=float(np.median(np.abs(diff[gt_ml > 0]) / gt_ml[gt_ml > 0]) * 100) if (gt_ml > 0).any() else float("nan"),
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


def bootstrap_icc_ci(pred_ml: np.ndarray, gt_ml: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of ICC(2,1), resampling *subjects* (the pair moves together).

    The charter's ICC threshold (0.85) sits inside this interval for the val split, so the point
    estimate alone must never be reported (A6 2026-09-09).
    """
    pred_ml = np.asarray(pred_ml, float)
    gt_ml = np.asarray(gt_ml, float)
    n = len(pred_ml)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array([icc21(pred_ml[i], gt_ml[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    return float(icc21(pred_ml, gt_ml)), float(np.nanpercentile(boots, 100 * alpha / 2)), float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))


def segmentation_summary(per_subject: list[dict], seed: int = 0) -> dict:
    """per_subject: dicts with keys dice (nan if GT empty), gt_ml, pred_ml, gt_pos (bool), pred_pos (bool),
    and optionally overlap_pos (bool: pred ∩ GT non-empty).

    Detection semantics (A5b 2026-09-09, A5a item 13):
      * `detection_sensitivity`         = P(pred mask non-empty | GT non-empty).  This is the charter's S2
                                          definition ("환자 단위 검출 = 예측 마스크 비어 있지 않음") and is kept
                                          unchanged, but it is a *non-empty-output rate*: a single stray voxel
                                          anywhere counts as TP (val: 9/218 subjects have Dice = 0 yet count).
      * `detection_sensitivity_overlap` = P(pred ∩ GT non-empty | GT non-empty): the prediction touches the
                                          lesion.  Reported next to it whenever `overlap_pos` is supplied.
      Both are NaN (not 0) when there is no GT-positive subject.
    """
    dices = np.array([d["dice"] for d in per_subject], float)
    gt_pos = np.array([d["gt_pos"] for d in per_subject], bool)
    pred_pos = np.array([d["pred_pos"] for d in per_subject], bool)
    has_overlap = all("overlap_pos" in d for d in per_subject) and len(per_subject) > 0
    overlap_pos = np.array([d.get("overlap_pos", False) for d in per_subject], bool)
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
        "detection_sensitivity": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "detection_specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        "detection_confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }
    if has_overlap:
        tp_o = int((gt_pos & overlap_pos).sum())
        out["detection_sensitivity_overlap"] = tp_o / (tp + fn) if (tp + fn) > 0 else float("nan")
        out["n_pred_nonempty_but_no_overlap"] = int((gt_pos & pred_pos & ~overlap_pos).sum())
    if va:
        out["volume"] = va.__dict__
    # Dice stratified by lesion size (small lesions are the clinically hard ones: lacunes < 1.5 cm ~ < 2 mL).
    # A3/A6 2026-09-09: after the 128^2 resample the smallest lesion is 3 voxels, so a single voxel moves
    # Dice by > 0.1 in the [0,2) mL band. Dice alone is therefore not interpretable there -> every band
    # also reports whether the lesion was DETECTED at all and the relative volume error.
    bins = [(0, 2), (2, 10), (10, 50), (50, np.inf)]

    def _band(m):
        pct = np.abs(pred_ml[m] - gt_ml[m]) / gt_ml[m] * 100 if m.any() else np.array([])
        return {"n": int(m.sum()),
                "dice_mean": float(np.nanmean(dices[m])) if m.any() else float("nan"),
                "dice_median": float(np.nanmedian(dices[m])) if m.any() else float("nan"),
                "detect_frac": float(np.mean(pred_pos[m])) if m.any() else float("nan"),
                "median_abs_pct_err": float(np.median(pct)) if m.any() else float("nan"),
                "median_gt_ml": float(np.median(gt_ml[m])) if m.any() else float("nan")}

    out["dice_by_gt_volume_ml"] = {f"[{a},{b})": _band(pos & (gt_ml >= a) & (gt_ml < b)) for a, b in bins}
    if va:
        # the ICC threshold in the charter sits inside the bootstrap CI, so the CI is not optional
        _, lo_i, hi_i = bootstrap_icc_ci(pred_ml[pos], gt_ml[pos], seed=seed)
        out["volume"]["icc21_ci95"] = [lo_i, hi_i]
    return out


def laa_vs_ce_auc(y_true: np.ndarray, proba: np.ndarray, classes: list) -> float:
    """AUC for LAA vs CE among the subjects whose true label is LAA or CE, scored by p(LAA) - p(CE).

    A6 2026-09-09 (charter, "etiology"): the 4-class macro AUC is inflated by the TOAST definition of
    SVO ("lacune < 1.5 cm"), which makes lesion volume alone a strong classifier. LAA and CE are both
    large-vessel/large-lesion classes, so this axis is where the label-definition shortcut is weakest
    and is therefore the primary etiology criterion. Returns NaN when either class is absent.
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    if "LAA" not in classes or "CE" not in classes:
        return float("nan")
    m = np.isin(y_true, ["LAA", "CE"])
    if m.sum() < 2 or len(set(y_true[m])) < 2:
        return float("nan")
    score = proba[m, classes.index("LAA")] - proba[m, classes.index("CE")]
    return float(roc_auc_score((y_true[m] == "LAA").astype(int), score))


def multiclass_auc(y_true: np.ndarray, proba: np.ndarray, classes: list) -> dict:
    """Macro one-vs-rest ROC-AUC plus per-class AUC. Classes absent in y_true get NaN."""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    per = {}
    for j, c in enumerate(classes):
        yb = (y_true == c).astype(int)
        per[str(c)] = float(roc_auc_score(yb, proba[:, j])) if 0 < yb.sum() < len(yb) else float("nan")
    return {"macro_auc_ovr": float(np.nanmean(list(per.values()))), "per_class_auc": per}


# ---------------------------------------------------------------------------------------------
# Prognosis (discharge mRS) metrics - A7 2026-09-10.
# The outcome is binary "poor" = mRS 3-6 vs "good" = mRS 0-2 (charter, extension task).  Everything
# below scores a *probability of poor outcome*, so calibration is reported next to discrimination:
# an AUC says only how the subjects are ranked, and a prognosis model that is used at the bedside is
# read as an absolute risk.
# ---------------------------------------------------------------------------------------------


def binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC of a single score against a 0/1 outcome. NaN when one class is absent."""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, float)
    if len(y_true) < 2 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(roc_auc_score(y_true, score))


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean squared error of the predicted probability (Brier 1950). Lower is better; 0.25 = coin flip."""
    y_true = np.asarray(y_true, float)
    proba = np.asarray(proba, float)
    return float(np.mean((proba - y_true) ** 2))


def brier_skill_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """1 - Brier / Brier(no-skill), where the no-skill reference predicts the observed prevalence.

    A raw Brier score is not comparable across cohorts with different event rates, so the skill score
    against the prevalence-only forecast is reported next to it.
    """
    y_true = np.asarray(y_true, float)
    ref = brier_score(y_true, np.full_like(y_true, y_true.mean()))
    return float(1 - brier_score(y_true, proba) / ref) if ref > 0 else float("nan")


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Cox (1958) calibration: refit `logit(y) = a + b * logit(p_hat)` on the evaluation set.

    * slope b = 1 and intercept a = 0 is perfect calibration.
    * b < 1 means the predictions are too extreme (the usual sign of overfitting);
      b > 1 means they are too timid.
    * `calibration_in_the_large` is the mean predicted risk minus the observed event rate.
    Returns NaN slope/intercept when one outcome class is absent (the refit is undefined).
    """
    from sklearn.linear_model import LogisticRegression

    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, float)
    out = {"mean_predicted": float(proba.mean()), "observed_rate": float(y_true.mean()),
           "calibration_in_the_large": float(proba.mean() - y_true.mean())}
    if len(set(y_true.tolist())) < 2:
        return out | {"slope": float("nan"), "intercept": float("nan")}
    lr = LogisticRegression(penalty=None, max_iter=1000)
    lr.fit(_logit(proba).reshape(-1, 1), y_true)
    return out | {"slope": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0])}


def calibration_bins(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 5) -> list:
    """Equal-count (quantile) calibration table: predicted vs observed risk per bin."""
    y_true = np.asarray(y_true, float)
    proba = np.asarray(proba, float)
    order = np.argsort(proba)
    chunks = np.array_split(order, n_bins)
    return [{"n": int(len(c)), "mean_predicted": float(proba[c].mean()),
             "observed_rate": float(y_true[c].mean()),
             "p_lo": float(proba[c].min()), "p_hi": float(proba[c].max())} for c in chunks if len(c)]


def binary_threshold_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Confusion matrix and the derived rates at a fixed operating point."""
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba, float) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {"threshold": float(threshold), "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "sensitivity": sens, "specificity": spec,
            "ppv": tp / (tp + fp) if tp + fp else float("nan"),
            "npv": tn / (tn + fn) if tn + fn else float("nan"),
            "accuracy": (tp + tn) / len(y_true) if len(y_true) else float("nan"),
            "balanced_accuracy": float(np.nanmean([sens, spec]))}


def prognosis_summary(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5,
                      n_bins: int = 5) -> dict:
    """Discrimination + calibration + one operating point, for a poor-outcome probability."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, float)
    return {"n": int(len(y_true)), "n_poor": int(y_true.sum()), "n_good": int((1 - y_true).sum()),
            "auc": binary_auc(y_true, proba),
            "brier": brier_score(y_true, proba),
            "brier_skill": brier_skill_score(y_true, proba),
            "calibration": calibration_slope_intercept(y_true, proba),
            "calibration_bins": calibration_bins(y_true, proba, n_bins),
            "operating_point": binary_threshold_metrics(y_true, proba, threshold)}


def bootstrap_binary_auc_ci(y_true: np.ndarray, score: np.ndarray, n_boot: int = 2000,
                            alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile bootstrap CI of a binary AUC, resampling *subjects*.

    Resamples that end up with a single outcome class are dropped (the AUC is undefined there) and
    the number dropped is reported, mirroring the etiology bootstrap rule fixed by A5a/A5b.
    """
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals, dropped = [], 0
    while len(vals) < n_boot:
        i = rng.integers(0, n, n)
        if 0 < y_true[i].sum() < n:
            vals.append(binary_auc(y_true[i], score[i]))
        else:
            dropped += 1
            if dropped > 100 * n_boot:
                break
    v = np.asarray(vals, float)
    return {"auc": binary_auc(y_true, score),
            "ci95": [float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2)))],
            "boot_sd": float(np.nanstd(v, ddof=1)), "n_boot_effective": int(len(v)),
            "n_resamples_dropped_single_class": int(dropped), "seed": int(seed)}


def paired_auc_delta_ci(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
                        n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Paired subject bootstrap of AUC(a) - AUC(b): both scores are evaluated on the SAME resample.

    `p_gt_0` is the fraction of resamples with a positive difference. It is neither a p-value nor a
    posterior probability (A5a item 5); it is reported only as a description of the resample cloud.
    """
    y_true = np.asarray(y_true).astype(int)
    a = np.asarray(score_a, float)
    b = np.asarray(score_b, float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    da, dropped = [], 0
    while len(da) < n_boot:
        i = rng.integers(0, n, n)
        if 0 < y_true[i].sum() < n:
            da.append(binary_auc(y_true[i], a[i]) - binary_auc(y_true[i], b[i]))
        else:
            dropped += 1
            if dropped > 100 * n_boot:
                break
    d = np.asarray(da, float)
    return {"auc_a": binary_auc(y_true, a), "auc_b": binary_auc(y_true, b),
            "delta": binary_auc(y_true, a) - binary_auc(y_true, b),
            "delta_ci95": [float(np.nanpercentile(d, 100 * alpha / 2)), float(np.nanpercentile(d, 100 * (1 - alpha / 2)))],
            "delta_boot_sd": float(np.nanstd(d, ddof=1)), "p_gt_0": float(np.mean(d > 0)),
            "n_boot_effective": int(len(d)), "n_resamples_dropped_single_class": int(dropped), "seed": int(seed)}


def hanley_mcneil_se(auc: float, n1: int, n2: int) -> float:
    """SE of an AUC with n1 positives and n2 negatives, Hanley & McNeil (1982) exponential approximation.

    Used to size the pre-registered test-set thresholds before the labels are read (A6 for etiology,
    A7 for prognosis). Defined here so that no analysis script re-implements it (CLAUDE.md).
    """
    from math import sqrt

    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    return float(sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n2 - 1) * (q2 - auc**2)) / (n1 * n2)))


def ordinal_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Auxiliary ordinal-mRS (0-6) agreement: Spearman rho, MAE, and |error| <= 1 rate."""
    from scipy import stats as _st

    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    rho = _st.spearmanr(y_true, y_pred)
    err = np.abs(y_pred - y_true)
    return {"n": int(len(y_true)), "spearman_rho": float(rho.statistic), "spearman_p": float(rho.pvalue),
            "mae": float(err.mean()), "within_1_rate": float((err <= 1).mean()),
            "rmse": float(np.sqrt((err**2).mean()))}
