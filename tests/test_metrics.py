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
