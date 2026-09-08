"""A3 regression tests: robust normalisation, mask binarisation, TRACE/ADC geometry agreement."""
import nibabel as nib
import numpy as np
import pytest

from strokeai.data.preprocess import preprocess_subject, robust_stats, robust_zscore


def _legacy_zscore(x, fg, clip=6.0):
    """The pre-fix formula. Must stay bit-identical on non-degenerate volumes."""
    v = x[fg]
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 + 1e-6
    return np.clip((x - med) / mad, -clip, clip).astype(np.float32)


def test_robust_stats_uses_mad_when_well_defined():
    v = np.random.default_rng(0).normal(100.0, 7.0, 5000)
    med, scale, deg = robust_stats(v)
    assert not deg
    assert med == float(np.median(v))
    assert scale == float(np.median(np.abs(v - np.median(v))) * 1.4826)


def test_robust_zscore_bit_identical_to_legacy_on_normal_volume():
    rng = np.random.default_rng(1)
    x = rng.normal(200.0, 40.0, (24, 24, 6)).astype(np.float32)
    fg = np.ones_like(x, dtype=bool)
    assert np.array_equal(robust_zscore(x, fg), _legacy_zscore(x, fg))


def test_robust_zscore_survives_zero_mad():
    """>50 % of the sample is a constant fill value (44/1233 SOOP subjects' ADC maps).

    The old code divided by 1e-6 and turned the whole channel into a saturated +-6 image.
    """
    rng = np.random.default_rng(2)
    x = np.full((20, 20, 8), -4.9, dtype=np.float32)
    tissue = rng.uniform(500.0, 15000.0, x[:, :9].shape)  # SOOP ADC spread; 45 % tissue, 55 % fill value
    x[:, :9] = tissue
    fg = np.ones_like(x, dtype=bool)
    assert np.median(np.abs(x - np.median(x))) == 0.0  # precondition: MAD is exactly 0
    med, scale, deg = robust_stats(x.ravel())
    assert deg and scale > 1.0
    z = robust_zscore(x, fg)
    assert np.isfinite(z).all()
    saturated = float((np.abs(z) >= 5.999).mean())
    assert saturated < 0.05, f"normalisation still saturates ({saturated:.2%} of voxels at the clip)"
    assert float((np.abs(z[:, :9]) >= 5.999).mean()) == 0.0  # tissue contrast is preserved
    assert len(np.unique(z)) > 1000  # not collapsed onto a handful of levels
    # the old code divided by 1e-6: every tissue voxel hit the clip and the channel collapsed to 2 levels
    legacy = _legacy_zscore(x, fg)
    assert float((np.abs(legacy[:, :9]) >= 5.999).mean()) == 1.0
    assert len(np.unique(legacy)) == 2


def _write(tmp, name, data, affine):
    p = tmp / name
    nib.save(nib.Nifti1Image(np.asarray(data), np.asarray(affine)), str(p))
    return p


def _subject(tmp, adc_affine=None, mask_affine=None, mask_value=1, size=(48, 40, 6)):
    rng = np.random.default_rng(3)
    H, W, S = size
    aff = np.diag([2.0, 2.0, 5.0, 1.0])
    tr = rng.normal(200.0, 30.0, (H, W, S)).astype(np.float32)
    tr[:4] = 0.0  # background, so foreground() has something to reject
    ad = rng.normal(900.0, 120.0, (H, W, S)).astype(np.float32)
    mk = np.zeros((H, W, S), np.float32)
    mk[10:20, 10:20, 2:4] = mask_value
    t = _write(tmp, "tr.nii.gz", tr[..., None], aff)          # 4-D like SOOP
    a = _write(tmp, "ad.nii.gz", ad[..., None], adc_affine if adc_affine is not None else aff)
    m = _write(tmp, "mk.nii.gz", mk, mask_affine if mask_affine is not None else aff)
    return t, a, m


def test_mask_is_binary_and_grid_matches_image(tmp_path):
    t, a, m = _subject(tmp_path, mask_value=3)  # A2: 10 SOOP subjects encode the label as 2 or 3
    r = preprocess_subject(t, a, m, size=32)
    assert r["img"].shape == (6, 2, 32, 32)
    assert r["mask"].shape == (6, 32, 32)
    assert set(np.unique(r["mask"]).tolist()) <= {0, 1}
    assert r["mask"].sum() > 0
    assert r["flags"]["mask_max_value"] == 3.0
    assert r["img"].dtype == np.float16 and r["mask"].dtype == np.uint8
    # in-plane voxel volume grows exactly by the resampling factor
    assert r["voxel_volume_mm3"] == pytest.approx(2.0 * 2.0 * 5.0 / ((32 / 48) * (32 / 40)))


def test_misaligned_adc_is_resampled_onto_the_trace_grid(tmp_path):
    aff = np.diag([2.0, 2.0, 5.0, 1.0])
    shifted = aff.copy()
    shifted[:3, 3] += [11.0, 10.0, -3.5]  # sub-235-style displacement
    t, a, m = _subject(tmp_path, adc_affine=shifted)
    r = preprocess_subject(t, a, m, size=32)
    assert r["flags"].get("adc_resampled_to_trace") is True
    assert r["flags"]["adc_affine_max_diff"] == pytest.approx(11.0)
    assert np.isfinite(r["img"].astype(np.float32)).all()
    # aligned ADC keeps the flag clear and leaves the channel untouched
    t2, a2, m2 = _subject(tmp_path, adc_affine=aff)
    r2 = preprocess_subject(t2, a2, m2, size=32)
    assert "adc_resampled_to_trace" not in r2["flags"]


def test_mask_affine_mismatch_is_a_hard_error(tmp_path):
    bad = np.diag([2.0, 2.0, 5.0, 1.0])
    bad[:3, 3] += [7.0, 0.0, 0.0]
    t, a, m = _subject(tmp_path, mask_affine=bad)
    with pytest.raises(AssertionError, match="mask/TRACE affine"):
        preprocess_subject(t, a, m, size=32)


def test_preprocessing_is_deterministic(tmp_path):
    t, a, m = _subject(tmp_path)
    r1 = preprocess_subject(t, a, m, size=32)
    r2 = preprocess_subject(t, a, m, size=32)
    assert np.array_equal(r1["img"], r2["img"]) and np.array_equal(r1["mask"], r2["mask"])
