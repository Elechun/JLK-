"""Per-subject preprocessing: canonical orientation, in-plane resampling, per-volume robust normalisation.

Output per subject (data/cache/<sid>.npz):
  img  : float16 (S, 2, H, W)   channel 0 = TRACE (b1000), channel 1 = ADC, both z-scored on brain foreground
  mask : uint8   (S, H, W)      acute lesion mask, binary
  meta : voxel_volume_mm3 after resampling, original shape, scale factors
Normalisation uses only the subject's own volume -> no train/test statistics leakage by construction.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def foreground(trace: np.ndarray) -> np.ndarray:
    """Brain foreground = voxels above 5 % of the 99th percentile (rejects air/background)."""
    p99 = np.percentile(trace, 99)
    return trace > 0.05 * p99


def robust_zscore(x: np.ndarray, fg: np.ndarray, clip: float = 6.0) -> np.ndarray:
    v = x[fg]
    if v.size < 100:  # degenerate volume
        v = x.ravel()
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 + 1e-6
    return np.clip((x - med) / mad, -clip, clip).astype(np.float32)


def load_canonical(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    img = nib.as_closest_canonical(nib.load(str(path)))  # RAS+: axis0=L->R, axis1=P->A, axis2=I->S
    data = np.asanyarray(img.dataobj).astype(np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data, img.affine, tuple(float(z) for z in img.header.get_zooms()[:3])


def preprocess_subject(trace_p: Path, adc_p: Path, mask_p: Path, size: int) -> dict:
    tr, aff_t, zooms = load_canonical(trace_p)
    ad, aff_a, _ = load_canonical(adc_p)
    mk, aff_m, _ = load_canonical(mask_p)
    assert tr.shape == ad.shape == mk.shape, (tr.shape, ad.shape, mk.shape)
    assert np.allclose(aff_t, aff_m, atol=1e-3), "mask/TRACE affine mismatch after canonicalisation"
    fg = foreground(tr)
    tr_n = robust_zscore(tr, fg)
    ad_n = robust_zscore(ad, fg)
    H, W, S = tr.shape
    sy, sx = size / H, size / W
    zoom = (sy, sx, 1.0)
    tr_r = ndimage.zoom(tr_n, zoom, order=1)
    ad_r = ndimage.zoom(ad_n, zoom, order=1)
    mk_r = ndimage.zoom((mk > 0).astype(np.uint8), zoom, order=0)
    img = np.stack([tr_r, ad_r], axis=0)  # (2, H', W', S)
    img = np.transpose(img, (3, 0, 1, 2)).astype(np.float16)  # (S, 2, H', W')
    mask = np.transpose(mk_r, (2, 0, 1)).astype(np.uint8)  # (S, H', W')
    voxel_volume_mm3 = float(zooms[0] * zooms[1] * zooms[2] / (sy * sx))
    return {"img": img, "mask": mask, "voxel_volume_mm3": voxel_volume_mm3,
            "orig_shape": np.array(tr.shape), "zooms": np.array(zooms), "scale": np.array([sy, sx])}
