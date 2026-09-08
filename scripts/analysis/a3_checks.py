#!/usr/bin/env python3
"""A3 checks: cache integrity, resampling safety, geometry agreement, acute/chronic mask duplication.

Reads ONLY the splits named in --which (default: train val). The held-out `test` split is never read
(CLAUDE.md); pass `--which test` at the final-evaluation step if the same audit is wanted there.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/analysis/a3_checks.py --out /tmp/a3
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def cache_check(args):
    sid, cache = args
    z = np.load(cache / f"{sid}.npz")
    img, mask = z["img"], z["mask"]
    u = np.unique(mask)
    return {
        "participant_id": sid,
        "img_shape": tuple(int(x) for x in img.shape),
        "mask_shape": tuple(int(x) for x in mask.shape),
        "img_dtype": str(img.dtype),
        "mask_dtype": str(mask.dtype),
        "mask_unique_ok": bool(set(u.tolist()) <= {0, 1}),
        "mask_unique": str(u.tolist()[:5]),
        "n_slices": int(img.shape[0]),
        "mask_voxels": int(mask.sum()),
        "pos_slices": int((mask.reshape(mask.shape[0], -1).sum(1) > 0).sum()),
        "voxel_volume_mm3": float(z["voxel_volume_mm3"]),
        "orig_shape": tuple(int(x) for x in z["orig_shape"]),
        "zooms": tuple(float(x) for x in z["zooms"]),
        "img_finite": bool(np.isfinite(img.astype(np.float32)).all()),
        "img_min": float(img.min()), "img_max": float(img.max()),
        "trace_mean": float(img[:, 0].astype(np.float32).mean()),
        "trace_std": float(img[:, 0].astype(np.float32).std()),
        "adc_mean": float(img[:, 1].astype(np.float32).mean()),
        "adc_std": float(img[:, 1].astype(np.float32).std()),
        # lesion intensity contrast (measured on the normalised volumes)
        "trace_lesion_mean": float(img[:, 0].astype(np.float32)[mask > 0].mean()) if mask.sum() else np.nan,
        "adc_lesion_mean": float(img[:, 1].astype(np.float32)[mask > 0].mean()) if mask.sum() else np.nan,
    }


def geom_check(args):
    """Header-only geometry audit on the raw NIfTIs (no voxel data loaded except the masks)."""
    sid, raw = args
    d = raw / sid / "dwi"
    md = raw / "derivatives" / "lesion_masks" / sid / "dwi"
    tr = nib.load(str(d / f"{sid}_rec-TRACE_dwi.nii.gz"))
    ad = nib.load(str(d / f"{sid}_rec-ADC_dwi.nii.gz"))
    mk = nib.load(str(md / f"{sid}_space-TRACE_desc-lesionAcute_mask.nii.gz"))
    trc, adc, mkc = (nib.as_closest_canonical(x) for x in (tr, ad, mk))
    out = {
        "participant_id": sid,
        "adc_affine_eq_trace_raw": bool(np.allclose(ad.affine, tr.affine, atol=1e-3)),
        "mask_affine_eq_trace_raw": bool(np.allclose(mk.affine, tr.affine, atol=1e-3)),
        "adc_affine_eq_trace_canon": bool(np.allclose(adc.affine, trc.affine, atol=1e-3)),
        "mask_affine_eq_trace_canon": bool(np.allclose(mkc.affine, trc.affine, atol=1e-3)),
        "orient_trace": "".join(nib.aff2axcodes(tr.affine)),
        "orient_canon": "".join(nib.aff2axcodes(trc.affine)),
        "shape_canon": tuple(int(x) for x in trc.shape[:3]),
        "zooms_canon": tuple(float(x) for x in trc.header.get_zooms()[:3]),
        "adc_max_abs_affine_diff": float(np.abs(np.asarray(ad.affine) - np.asarray(tr.affine)).max()),
    }
    # acute vs chronic mask duplication (A2 hand-off item 3)
    mc_p = md / f"{sid}_space-TRACE_desc-lesionChronic_mask.nii.gz"
    a = np.asanyarray(mk.dataobj) > 0
    out["acute_voxels"] = int(a.sum())
    if mc_p.exists():
        c = np.asanyarray(nib.load(str(mc_p)).dataobj) > 0
        inter = int((a & c).sum())
        out.update(has_chronic=True, chronic_voxels=int(c.sum()),
                   acute_eq_chronic=bool(a.shape == c.shape and np.array_equal(a, c)),
                   acute_subset_chronic=bool(a.shape == c.shape and a.sum() > 0 and inter == int(a.sum())),
                   acute_chronic_overlap_voxels=inter,
                   acute_chronic_dice=float(2 * inter / (a.sum() + c.sum())) if (a.sum() + c.sum()) else 0.0)
    else:
        out.update(has_chronic=False, chronic_voxels=0, acute_eq_chronic=False,
                   acute_subset_chronic=False, acute_chronic_overlap_voxels=0, acute_chronic_dice=np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/ds004889"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--index", type=Path, default=Path("data/index.csv"))
    ap.add_argument("--which", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("results/a3"))
    ap.add_argument("--skip-geom", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sp = json.load(open(a.splits))
    ids = {w: sp[w] for w in a.which}
    flat = [(s, w) for w in a.which for s in sp[w]]
    print(f"auditing {len(flat)} subjects from splits {a.which} (test split NOT read)")

    with cf.ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(cache_check, [(s, a.cache) for s, _ in flat], chunksize=8))
    cc = pd.DataFrame(rows)
    cc["split"] = [w for _, w in flat]
    cc.to_csv(a.out / "cache_check.csv", index=False)

    if not a.skip_geom:
        with cf.ProcessPoolExecutor(a.workers) as ex:
            grows = list(ex.map(geom_check, [(s, a.raw) for s, _ in flat], chunksize=8))
        gc = pd.DataFrame(grows)
        gc["split"] = [w for _, w in flat]
        gc.to_csv(a.out / "geom_check.csv", index=False)
    else:
        gc = pd.read_csv(a.out / "geom_check.csv")

    idx = pd.read_csv(a.index).set_index("participant_id")
    cc = cc.set_index("participant_id")
    cc["native_ml"] = idx.loc[cc.index, "mask_acute_ml"]
    cc["cached_ml"] = cc["mask_voxels"] * cc["voxel_volume_mm3"] / 1000.0
    cc["vol_ratio"] = cc["cached_ml"] / cc["native_ml"].replace(0, np.nan)
    cc["native_voxels"] = idx.loc[cc.index, "mask_acute_voxels"]
    cc.reset_index().to_csv(a.out / "cache_check.csv", index=False)

    rep = {
        "n_subjects": int(len(cc)),
        "splits": {w: len(v) for w, v in ids.items()},
        "img_shapes": cc["img_shape"].astype(str).value_counts().to_dict(),
        "mask_binary_all": bool(cc["mask_unique_ok"].all()),
        "mask_unique_violations": cc.index[~cc["mask_unique_ok"]].tolist(),
        "img_all_finite": bool(cc["img_finite"].all()),
        "img_range": [float(cc["img_min"].min()), float(cc["img_max"].max())],
        "in_plane_128_all": bool(cc["img_shape"].map(lambda t: t[2] == 128 and t[3] == 128).all()),
        "empty_mask_after_resample": int((cc["mask_voxels"] == 0).sum()),
        "empty_mask_natively": int((cc["native_voxels"] == 0).sum()),
        "min_cached_mask_voxels": int(cc["mask_voxels"].min()),
        "vol_ratio_quantiles": {q: float(cc["vol_ratio"].quantile(q)) for q in [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]},
        "voxel_volume_mm3": {"min": float(cc["voxel_volume_mm3"].min()), "median": float(cc["voxel_volume_mm3"].median()),
                             "max": float(cc["voxel_volume_mm3"].max()),
                             "ratio_max_min": float(cc["voxel_volume_mm3"].max() / cc["voxel_volume_mm3"].min())},
        "geometry": {
            "orient_raw_counts": gc["orient_trace"].value_counts().to_dict(),
            "orient_canon_counts": gc["orient_canon"].value_counts().to_dict(),
            "mask_affine_eq_trace_canon_all": bool(gc["mask_affine_eq_trace_canon"].all()),
            "adc_affine_eq_trace_raw_all": bool(gc["adc_affine_eq_trace_raw"].all()),
            "adc_affine_eq_trace_canon_all": bool(gc["adc_affine_eq_trace_canon"].all()),
            "adc_affine_mismatch_subjects": gc.loc[~gc["adc_affine_eq_trace_canon"], "participant_id"].tolist()[:20],
            "adc_affine_mismatch_n": int((~gc["adc_affine_eq_trace_canon"]).sum()),
            "adc_max_abs_affine_diff_max": float(gc["adc_max_abs_affine_diff"].max()),
        },
        "acute_chronic": {
            "n_with_chronic": int(gc["has_chronic"].sum()),
            "n_acute_eq_chronic": int(gc["acute_eq_chronic"].sum()),
            "acute_eq_chronic_subjects": gc.loc[gc["acute_eq_chronic"], "participant_id"].tolist(),
            "n_acute_subset_chronic": int(gc["acute_subset_chronic"].sum()),
            "acute_subset_chronic_subjects": gc.loc[gc["acute_subset_chronic"], "participant_id"].tolist(),
            "n_overlap_any": int((gc["acute_chronic_overlap_voxels"] > 0).sum()),
            "acute_chronic_dice_median_when_overlap": float(
                gc.loc[gc["acute_chronic_overlap_voxels"] > 0, "acute_chronic_dice"].median())
            if (gc["acute_chronic_overlap_voxels"] > 0).any() else None,
        },
    }
    json.dump(rep, open(a.out / "checks.json", "w"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))

    # suspicious-subject flag file (train+val only)
    flag = gc[["participant_id", "split", "acute_eq_chronic", "acute_subset_chronic", "acute_chronic_dice"]].copy()
    flag.to_csv(a.out / "acute_chronic_flags.csv", index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
