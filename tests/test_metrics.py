import numpy as np
import pytest

from strokeai.metrics import bootstrap_ci, dice_binary, icc21, segmentation_summary, volume_agreement, volume_ml


def test_dice_basic():
    a = np.zeros((4, 4), bool); a[:2] = True
    b = np.zeros((4, 4), bool); b[1:3] = True
    assert dice_binary(a, b) == pytest.approx(2 * 4 / (8 + 8))
    assert dice_binary(a, a) == 1.0
    assert dice_binary(a, ~a) == 0.0


def test_dice_both_empty_is_nan():
    z = np.zeros((3, 3), bool)
    assert np.isnan(dice_binary(z, z))


def test_volume():
    m = np.ones((2, 3, 4), bool)
    assert volume_ml(m, 100.0) == pytest.approx(24 * 100 / 1000)


def test_icc_perfect_and_noise():
    x = np.arange(20, dtype=float)
    assert icc21(x, x) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    assert icc21(x, x + rng.normal(0, 5, 20)) < 0.9


def test_volume_agreement_bias():
    gt = np.linspace(1, 50, 30)
    va = volume_agreement(gt + 2.0, gt)
    assert va.mean_diff_ml == pytest.approx(2.0)
    assert va.pearson_r == pytest.approx(1.0)
    assert va.mae_ml == pytest.approx(2.0)


def test_bootstrap_ci_contains_mean():
    v = np.random.default_rng(1).normal(0.6, 0.1, 100)
    m, lo, hi = bootstrap_ci(v, np.nanmean, n_boot=500)
    assert lo <= m <= hi


def test_summary_excludes_empty_gt_from_dice():
    per = [
        {"dice": 0.8, "gt_ml": 5, "pred_ml": 5, "gt_pos": True, "pred_pos": True},
        {"dice": 0.4, "gt_ml": 3, "pred_ml": 6, "gt_pos": True, "pred_pos": True},
        {"dice": float("nan"), "gt_ml": 0, "pred_ml": 0, "gt_pos": False, "pred_pos": False},
        {"dice": 0.0, "gt_ml": 2, "pred_ml": 0, "gt_pos": True, "pred_pos": False},
    ]
    s = segmentation_summary(per, seed=0)
    assert s["n_gt_positive"] == 3
    assert s["dice_pos_mean"] == pytest.approx((0.8 + 0.4 + 0.0) / 3)
    assert s["detection_sensitivity"] == pytest.approx(2 / 3)
    assert s["detection_confusion"] == {"tp": 2, "fn": 1, "fp": 0, "tn": 1}


def test_volume_agreement_reports_relative_error():
    """A3/A2 hand-off: volume error must be reported in mL *and* in %, because lesion volumes span
    four orders of magnitude and mL-only errors are dominated by the largest lesions."""
    import numpy as np

    from strokeai.metrics import volume_agreement

    gt = np.array([1.0, 10.0, 100.0])
    pred = np.array([2.0, 11.0, 110.0])
    va = volume_agreement(pred, gt)
    assert va.mae_ml == pytest.approx((1 + 1 + 10) / 3)
    assert va.mape_pct == pytest.approx((100 + 10 + 10) / 3)
    assert va.median_abs_pct_err == pytest.approx(10.0)


def test_small_lesion_band_reports_detection_and_volume_error():
    """A3 hand-off 3 / A6: in the [0,2) mL band a single voxel moves Dice by >0.1 (the smallest cached
    lesion is 3 voxels), so Dice alone is not interpretable there - the band must also carry whether the
    lesion was detected at all and the relative volume error."""
    per = [
        {"dice": 0.2, "gt_ml": 0.5, "pred_ml": 0.25, "gt_pos": True, "pred_pos": True},
        {"dice": 0.0, "gt_ml": 1.0, "pred_ml": 0.0, "gt_pos": True, "pred_pos": False},
        {"dice": 0.9, "gt_ml": 20.0, "pred_ml": 22.0, "gt_pos": True, "pred_pos": True},
    ]
    s = segmentation_summary(per, seed=0)
    small = s["dice_by_gt_volume_ml"]["[0,2)"]
    assert small["n"] == 2
    assert small["detect_frac"] == pytest.approx(0.5)
    assert small["median_abs_pct_err"] == pytest.approx((50.0 + 100.0) / 2)
    assert small["median_gt_ml"] == pytest.approx(0.75)
    assert s["dice_by_gt_volume_ml"]["[10,50)"]["detect_frac"] == pytest.approx(1.0)


def test_icc_ci_is_reported_and_brackets_the_point_estimate():
    """A6: the charter's ICC >= 0.85 threshold falls inside the val bootstrap interval, so the point
    estimate must never be reported without the interval."""
    rng = np.random.default_rng(0)
    gt = rng.uniform(1, 100, 60)
    pred = gt * rng.uniform(0.8, 1.2, 60)
    per = [{"dice": 0.7, "gt_ml": float(g), "pred_ml": float(p), "gt_pos": True, "pred_pos": True}
           for g, p in zip(gt, pred)]
    s = segmentation_summary(per, seed=0)
    lo, hi = s["volume"]["icc21_ci95"]
    assert lo <= s["volume"]["icc21"] <= hi
    assert hi <= 1.0


def test_laa_vs_ce_auc_is_conditional_on_those_two_classes():
    """A6 charter criterion E1. The metric must ignore SVO / Others rows entirely and score only
    p(LAA) - p(CE) on the LAA/CE subjects, so that the SVO size shortcut cannot inflate it."""
    from strokeai.metrics import laa_vs_ce_auc

    classes = ["LAA", "CE", "SVO", "Others"]
    y = np.array(["LAA", "LAA", "CE", "CE", "SVO", "Others"])
    proba = np.array([
        [0.7, 0.1, 0.1, 0.1],   # LAA, confident   -> ranked above
        [0.5, 0.3, 0.1, 0.1],   # LAA
        [0.2, 0.6, 0.1, 0.1],   # CE
        [0.1, 0.7, 0.1, 0.1],   # CE
        [0.9, 0.0, 0.1, 0.0],   # SVO row: must not be used at all
        [0.0, 0.9, 0.1, 0.0],   # Others row: must not be used at all
    ])
    assert laa_vs_ce_auc(y, proba, classes) == pytest.approx(1.0)
    # perfectly reversed scores -> 0.0, and the ignored rows still do not matter
    assert laa_vs_ce_auc(y, proba[:, [1, 0, 2, 3]], classes) == pytest.approx(0.0)
    # missing class -> NaN, never a silent 0.5
    assert np.isnan(laa_vs_ce_auc(np.array(["SVO", "Others"]), proba[:2], classes))


def test_dice_rejects_shape_mismatch_and_sensitivity_is_nan_without_positives():
    """A5a item 16 / A5b: broadcasting made dice_binary((1,3), (3,1)) return 3.0; and a cohort without a
    single GT-positive subject must report sensitivity NaN, not 0."""
    with pytest.raises(ValueError):
        dice_binary(np.ones((1, 3), bool), np.ones((3, 1), bool))
    per = [{"dice": float("nan"), "gt_ml": 0, "pred_ml": 0, "gt_pos": False, "pred_pos": False}]
    s = segmentation_summary(per, seed=0)
    assert np.isnan(s["detection_sensitivity"])


def test_overlap_sensitivity_is_stricter_than_nonempty_output():
    """A5a item 13 / A5b: `detection_sensitivity` counts any non-empty prediction; the overlap variant
    requires the prediction to touch the lesion. Dice = 0 with a non-empty prediction separates them."""
    per = [
        {"dice": 0.8, "gt_ml": 5, "pred_ml": 5, "gt_pos": True, "pred_pos": True, "overlap_pos": True},
        {"dice": 0.0, "gt_ml": 3, "pred_ml": 1, "gt_pos": True, "pred_pos": True, "overlap_pos": False},
        {"dice": 0.0, "gt_ml": 2, "pred_ml": 0, "gt_pos": True, "pred_pos": False, "overlap_pos": False},
    ]
    s = segmentation_summary(per, seed=0)
    assert s["detection_sensitivity"] == pytest.approx(2 / 3)
    assert s["detection_sensitivity_overlap"] == pytest.approx(1 / 3)
    assert s["n_pred_nonempty_but_no_overlap"] == 1
    # without the optional key the summary stays backward compatible
    s2 = segmentation_summary([{k: v for k, v in d.items() if k != "overlap_pos"} for d in per], seed=0)
    assert "detection_sensitivity_overlap" not in s2
