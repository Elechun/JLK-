import numpy as np

from strokeai.data.preprocess import foreground, robust_zscore
from strokeai.features import FEATURE_NAMES, lesion_features


def test_zscore_uses_foreground_only():
    x = np.zeros((32, 32, 4), np.float32)
    x[8:24, 8:24] = 100 + np.random.default_rng(0).normal(0, 5, (16, 16, 4))
    fg = foreground(x)
    assert fg[16, 16, 0] and not fg[0, 0, 0]
    z = robust_zscore(x, fg)
    assert abs(np.median(z[fg])) < 0.2 and z.min() >= -6 and z.max() <= 6


def test_features_left_right_and_empty():
    """Cache layout is (S, H, W) with H = patient left-right (A5b): a lesion at small H is LEFT."""
    m = np.zeros((10, 32, 32), np.uint8)
    m[3:5, 2:6, 10:14] = 1  # small H -> patient left in RAS ; W (anterior-posterior) centred
    f = lesion_features(m, 10.0)
    assert set(f) == set(FEATURE_NAMES)
    assert f["laterality_left_frac"] == 1.0 and f["n_components"] == 1 and f["volume_ml"] == 32 * 10 / 1000
    assert f["centroid_x_norm"] == (2 + 5) / 2 / 32 and f["centroid_y_norm"] == (10 + 13) / 2 / 32
    # the same lesion mirrored along W (anterior-posterior) must NOT change laterality
    assert lesion_features(m[:, :, ::-1], 10.0)["laterality_left_frac"] == 1.0
    # mirrored along H (left-right) it becomes a right-hemisphere lesion
    assert lesion_features(m[:, ::-1, :], 10.0)["laterality_left_frac"] == 0.0
    assert all(v == 0 for k, v in lesion_features(np.zeros_like(m), 10.0).items())


def test_features_left_right_through_real_preprocessing(tmp_path):
    """A5b 2026-09-09 (A5a item 3): the old `features.py` read W as left-right, but `preprocess_subject`
    stores (S, H, W) with H = RAS x.  Push a NIfTI whose lesion sits at small RAS-x (= patient LEFT)
    through the real preprocessing and require laterality_left_frac == 1."""
    import nibabel as nib

    from strokeai.data.preprocess import preprocess_subject

    rng = np.random.default_rng(0)
    X, Y, Z = 48, 40, 6
    aff = np.diag([2.0, 2.0, 5.0, 1.0])  # RAS: axis0 = x (L->R), axis1 = y (P->A), axis2 = z
    tr = rng.normal(200.0, 30.0, (X, Y, Z)).astype(np.float32)
    ad = rng.normal(900.0, 120.0, (X, Y, Z)).astype(np.float32)
    mk = np.zeros((X, Y, Z), np.float32)
    mk[4:12, 30:38, 2:4] = 1  # LEFT (small x) and ANTERIOR (large y)
    for n, d in [("tr", tr[..., None]), ("ad", ad[..., None]), ("mk", mk)]:
        nib.save(nib.Nifti1Image(d, aff), str(tmp_path / f"{n}.nii.gz"))
    # LAS on disk (what SOOP ships) must give the same answer after canonicalisation
    las = aff.copy(); las[0, 0] = -2.0; las[0, 3] = 2.0 * (X - 1)
    for n, d in [("tr_las", tr[::-1][..., None]), ("ad_las", ad[::-1][..., None]), ("mk_las", mk[::-1])]:
        nib.save(nib.Nifti1Image(np.ascontiguousarray(d), las), str(tmp_path / f"{n}.nii.gz"))
    for suffix in ("", "_las"):
        r = preprocess_subject(tmp_path / f"tr{suffix}.nii.gz", tmp_path / f"ad{suffix}.nii.gz",
                               tmp_path / f"mk{suffix}.nii.gz", size=32)
        f = lesion_features(r["mask"], r["voxel_volume_mm3"])
        assert f["laterality_left_frac"] == 1.0, suffix
        assert f["centroid_x_norm"] < 0.5 < f["centroid_y_norm"], (suffix, f)
