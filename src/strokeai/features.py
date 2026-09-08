"""Hand-crafted lesion features for etiology classification (JBS-01K-style LAA / CE / SVO / Others).

Features are computed from a binary lesion mask (ground truth *or* model prediction) in canonical RAS space,
so the same code serves both the 'oracle' and the 'deployment-like' classifiers.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

FEATURE_NAMES = [
    "volume_ml", "log_volume_ml", "n_components", "largest_component_ml", "largest_component_frac",
    "n_slices_involved", "z_extent_frac", "laterality_left_frac", "bilateral", "centroid_x_norm", "centroid_y_norm",
    "centroid_z_norm", "spread_x_norm", "spread_y_norm", "multi_territory_proxy",
]


def lesion_features(mask: np.ndarray, voxel_volume_mm3: float) -> dict:
    """mask: (S, H, W) binary in canonical orientation (W axis = left->right). Returns dict of floats."""
    m = mask.astype(bool)
    S, H, W = m.shape
    vol = m.sum() * voxel_volume_mm3 / 1000.0
    if m.sum() == 0:
        return {k: 0.0 for k in FEATURE_NAMES} | {"log_volume_ml": np.log1p(0.0)}
    lab, n = ndimage.label(m)
    sizes = np.bincount(lab.ravel())[1:]
    largest = sizes.max() * voxel_volume_mm3 / 1000.0
    zs, ys, xs = np.nonzero(m)
    left_frac = float((xs < W / 2).mean())  # RAS: x increases to the Right -> small x = patient Left
    bilateral = float(0.1 < left_frac < 0.9 and n > 1)
    # components whose centroids are far apart in-plane (>1/4 image) hint at multiple vascular territories
    cents = ndimage.center_of_mass(m, lab, range(1, n + 1))
    xy = np.array([[c[2] / W, c[1] / H] for c in cents]) if n > 1 else np.zeros((1, 2))
    multi = float(n > 1 and np.max(np.linalg.norm(xy[:, None] - xy[None], axis=-1)) > 0.25)
    return {
        "volume_ml": float(vol), "log_volume_ml": float(np.log1p(vol)), "n_components": float(n),
        "largest_component_ml": float(largest), "largest_component_frac": float(sizes.max() / sizes.sum()),
        "n_slices_involved": float(len(np.unique(zs))), "z_extent_frac": float((zs.max() - zs.min() + 1) / S),
        "laterality_left_frac": left_frac, "bilateral": bilateral,
        "centroid_x_norm": float(xs.mean() / W), "centroid_y_norm": float(ys.mean() / H), "centroid_z_norm": float(zs.mean() / S),
        "spread_x_norm": float(xs.std() / W), "spread_y_norm": float(ys.std() / H), "multi_territory_proxy": multi,
    }
