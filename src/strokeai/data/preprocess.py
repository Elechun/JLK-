"""Per-subject preprocessing: canonical orientation, in-plane resampling, per-volume robust normalisation.

Output per subject (data/cache/<sid>.npz):
  img  : float16 (S, 2, H, W)   channel 0 = TRACE (b1000), channel 1 = ADC, both z-scored on brain foreground
  mask : uint8   (S, H, W)      acute lesion mask, binary
  meta : voxel_volume_mm3 after resampling, original shape, scale factors
Normalisation uses only the subject's own volume -> no train/test statistics leakage by construction.

Label decisions (A2 review, 2026-09-09):
  * The acute mask is binarised with ``mk > 0``. Exactly 10 SOOP subjects store the label as 2 or 3 instead
    of 1 (subject blocks 503-509 and 1252-1257); A2 showed these are an intensity-coding artefact, not a
    sub-class (every mask carries a single non-zero value, and TRACE/ADC signal is the same as elsewhere),
    so ``> 0`` is the correct binarisation.  `preprocess_subject` reports the observed maximum in `flags`.
  * Only ``*_desc-lesionAcute_mask.nii.gz`` is ever read.  The *combined* ``*_desc-lesion_mask.nii.gz`` is
    NOT used: its value encoding is inconsistent (sub-507 = {1, 4}), see the A2 report.
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


def robust_stats(v: np.ndarray) -> tuple[float, float, bool]:
    """Robust location/scale of a 1-D sample.

    Returns (median, scale, degenerate).  The normal path is the MAD (x1.4826).  In ~3.6 % of SOOP subjects
    the ADC map stores a single constant fill value on more than half of the TRACE brain foreground, so the
    median *is* that fill value and the MAD is exactly 0 -> dividing by it turned the whole ADC channel into
    a saturated +-clip image (measured: 44/1233 train+val subjects, up to 64 % of voxels at the clip).
    When that happens we drop the fill value, and fall back to IQR/1.349 and then to the std.
    """
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)) * 1.4826)
    if mad > 0.0:
        return med, mad, False
    rest = v[v != med]
    if rest.size >= 100:
        med2 = float(np.median(rest))
        mad2 = float(np.median(np.abs(rest - med2)) * 1.4826)
        if mad2 > 0.0:
            return med2, mad2, True
    q75, q25 = np.percentile(v, [75, 25])
    scale = float((q75 - q25) / 1.349)
    if scale <= 0.0:
        scale = float(np.std(v))
    if scale <= 0.0:
        scale = 1.0
    return med, scale, True


def robust_zscore(x: np.ndarray, fg: np.ndarray, clip: float = 6.0) -> np.ndarray:
    v = x[fg]
    if v.size < 100:  # degenerate volume
        v = x.ravel()
    med, scale, _ = robust_stats(v)
    return np.clip((x - med) / (scale + 1e-6), -clip, clip).astype(np.float32)


def load_canonical(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    img = nib.as_closest_canonical(nib.load(str(path)))  # RAS+: axis0=L->R, axis1=P->A, axis2=I->S
    data = np.asanyarray(img.dataobj).astype(np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data, np.asarray(img.affine), tuple(float(z) for z in img.header.get_zooms()[:3])


def resample_to_grid(data: np.ndarray, affine: np.ndarray, shape: tuple[int, ...], target_affine: np.ndarray,
                     order: int = 1) -> np.ndarray:
    """Resample `data` (defined by `affine`) onto the voxel grid (`shape`, `target_affine`)."""
    from nibabel.processing import resample_from_to

    src = nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.asarray(affine))
    out = resample_from_to(src, (tuple(int(s) for s in shape), np.asarray(target_affine)), order=order, cval=0.0)
    return np.asanyarray(out.dataobj).astype(np.float32)


def preprocess_subject(trace_p: Path, adc_p: Path, mask_p: Path, size: int) -> dict:
    """Returns the arrays to cache plus a `flags` dict of per-subject data-quality observations.

    `flags` is *not* meant to be stored in the .npz (callers pop it and aggregate it into the run manifest)
    so that the cache content stays byte-reproducible.
    """
    tr, aff_t, zooms = load_canonical(trace_p)
    ad, aff_a, _ = load_canonical(adc_p)
    mk, aff_m, _ = load_canonical(mask_p)
    flags: dict = {}

    # --- geometry: everything must live on the TRACE voxel grid -------------------------------------
    # The mask is authored in TRACE space, so a mismatch there is a hard error.
    assert tr.shape == mk.shape, (tr.shape, mk.shape)
    assert np.allclose(aff_t, aff_m, atol=1e-3), "mask/TRACE affine mismatch after canonicalisation"
    # The ADC is a separate reconstruction and *can* sit on a shifted grid (measured: 1/1233 train+val
    # subjects, sub-235, ~6.6 mm displacement at the image centre). Resample instead of silently stacking
    # two misaligned channels.
    if ad.shape != tr.shape or not np.allclose(aff_a, aff_t, atol=1e-3):
        flags["adc_resampled_to_trace"] = True
        flags["adc_affine_max_diff"] = float(np.abs(np.asarray(aff_a) - np.asarray(aff_t)).max())
        ad = resample_to_grid(ad, aff_a, tr.shape, aff_t, order=1)
    assert tr.shape == ad.shape == mk.shape, (tr.shape, ad.shape, mk.shape)

    fg = foreground(tr)
    med_t, sc_t, deg_t = robust_stats(tr[fg] if fg.sum() >= 100 else tr.ravel())
    med_a, sc_a, deg_a = robust_stats(ad[fg] if fg.sum() >= 100 else ad.ravel())
    if deg_t:
        flags["trace_degenerate_mad"] = True
    if deg_a:
        flags["adc_degenerate_mad"] = True
    tr_n = robust_zscore(tr, fg)
    ad_n = robust_zscore(ad, fg)

    flags["mask_max_value"] = float(mk.max())  # A2: 10 subjects encode the label as 2 or 3, not 1
    H, W, S = tr.shape
    sy, sx = size / H, size / W
    zoom = (sy, sx, 1.0)
    tr_r = ndimage.zoom(tr_n, zoom, order=1)
    ad_r = ndimage.zoom(ad_n, zoom, order=1)
    mk_r = ndimage.zoom((mk > 0).astype(np.uint8), zoom, order=0)  # nearest neighbour -> stays binary
    assert set(np.unique(mk_r).tolist()) <= {0, 1}, "mask is not binary after resampling"
    assert tr_r.shape == mk_r.shape == (size, size, S), (tr_r.shape, mk_r.shape)
    if mk.any() and not mk_r.any():
        flags["mask_lost_by_resampling"] = True
    img = np.stack([tr_r, ad_r], axis=0)  # (2, H', W', S)
    img = np.transpose(img, (3, 0, 1, 2)).astype(np.float16)  # (S, 2, H', W')
    mask = np.transpose(mk_r, (2, 0, 1)).astype(np.uint8)  # (S, H', W')
    voxel_volume_mm3 = float(zooms[0] * zooms[1] * zooms[2] / (sy * sx))
    return {"img": img, "mask": mask, "voxel_volume_mm3": voxel_volume_mm3,
            "orig_shape": np.array(tr.shape), "zooms": np.array(zooms), "scale": np.array([sy, sx]),
            "flags": flags}
