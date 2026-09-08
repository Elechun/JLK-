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
    m = np.zeros((10, 32, 32), np.uint8)
    m[3:5, 10:14, 2:6] = 1  # small x -> patient left in RAS
    f = lesion_features(m, 10.0)
    assert set(f) == set(FEATURE_NAMES)
    assert f["laterality_left_frac"] == 1.0 and f["n_components"] == 1 and f["volume_ml"] == 32 * 10 / 1000
    assert all(v == 0 for k, v in lesion_features(np.zeros_like(m), 10.0).items())
