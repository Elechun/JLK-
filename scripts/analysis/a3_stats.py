#!/usr/bin/env python3
"""A3 descriptive statistics per split: slices, lesion-positive slice fraction, lesion volume, intensities.

Reads ONLY the splits named in --which (default: train val). The held-out `test` split is never read
(CLAUDE.md); its subject count is reported from splits.json without touching any image.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/analysis/a3_stats.py --out results/a3
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
from strokeai.data.preprocess import foreground, load_canonical  # noqa: E402

Q = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]


def per_subject(args):
    sid, cache, raw = args
    z = np.load(cache / f"{sid}.npz")
    img, mask = z["img"], z["mask"]
    vv = float(z["voxel_volume_mm3"])
    m = mask > 0
    pos = m.reshape(m.shape[0], -1).any(1)
    tr = img[:, 0].astype(np.float32)
    ad = img[:, 1].astype(np.float32)
    # brain foreground on the *native* TRACE, using the pipeline's own definition
    t_raw, _, zooms = load_canonical(raw / sid / "dwi" / f"{sid}_rec-TRACE_dwi.nii.gz")
    a_raw, _, _ = load_canonical(raw / sid / "dwi" / f"{sid}_rec-ADC_dwi.nii.gz")
    fg = foreground(t_raw)
    native_vv = float(np.prod(zooms))
    brain_ml = float(fg.sum() * native_vv / 1000.0)
    tfg, afg = t_raw[fg], a_raw[fg]
    r = {
        "participant_id": sid,
        "n_slices": int(img.shape[0]),
        "n_pos_slices": int(pos.sum()),
        "pos_slice_frac": float(pos.mean()),
        "lesion_voxels": int(m.sum()),
        "lesion_ml": float(m.sum() * vv / 1000.0),
        "voxel_volume_mm3_resampled": vv,
        "voxel_volume_mm3_native": native_vv,
        "brain_ml": brain_ml,
        "lesion_pct_brain": float(m.sum() * vv / 1000.0) / brain_ml * 100.0 if brain_ml else np.nan,
        "native_shape": str(tuple(int(x) for x in z["orig_shape"])),
        # normalised intensities (what the model sees)
        "trace_z_mean": float(tr.mean()), "trace_z_p50": float(np.median(tr)), "trace_z_p99": float(np.percentile(tr, 99)),
        "adc_z_mean": float(ad.mean()), "adc_z_p50": float(np.median(ad)), "adc_z_p99": float(np.percentile(ad, 99)),
        "trace_z_lesion_p50": float(np.median(tr[m])) if m.any() else np.nan,
        "adc_z_lesion_p50": float(np.median(ad[m])) if m.any() else np.nan,
        # native intensity scale (pre-normalisation) on brain foreground
        "trace_raw_med": float(np.median(tfg)), "trace_raw_mad": float(np.median(np.abs(tfg - np.median(tfg))) * 1.4826),
        "adc_raw_med": float(np.median(afg)), "adc_raw_mad": float(np.median(np.abs(afg - np.median(afg))) * 1.4826),
        "trace_raw_max": float(t_raw.max()), "adc_raw_max": float(a_raw.max()),
    }
    return r


def qtab(s: pd.Series) -> dict:
    return {"n": int(s.notna().sum()), "mean": float(s.mean()), **{f"p{int(q*100)}": float(s.quantile(q)) for q in Q}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/ds004889"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--index", type=Path, default=Path("data/index.csv"))
    ap.add_argument("--which", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("results/a3"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sp = json.load(open(a.splits))
    flat = [(s, w) for w in a.which for s in sp[w]]
    with cf.ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(per_subject, [(s, a.cache, a.raw) for s, _ in flat], chunksize=8))
    df = pd.DataFrame(rows)
    df["split"] = [w for _, w in flat]
    idx = pd.read_csv(a.index).set_index("participant_id")
    for c in ["etiology_code", "trace_Manufacturer", "trace_MagneticFieldStrength", "age", "sex", "nihss"]:
        if c in idx.columns:
            df[c] = idx.loc[df["participant_id"], c].values
    df.to_csv(a.out / "per_subject_stats.csv", index=False)

    tab = {}
    for w, g in df.groupby("split"):
        tab[w] = {
            "n_subjects": int(len(g)),
            "n_slices_total": int(g["n_slices"].sum()),
            "slices_per_subject": qtab(g["n_slices"]),
            "n_pos_slices_total": int(g["n_pos_slices"].sum()),
            "pos_slice_frac_pooled": float(g["n_pos_slices"].sum() / g["n_slices"].sum()),
            "pos_slice_frac_per_subject": qtab(g["pos_slice_frac"]),
            "lesion_ml": qtab(g["lesion_ml"]),
            "lesion_pct_brain": qtab(g["lesion_pct_brain"]),
            "lesion_voxels_resampled": qtab(g["lesion_voxels"]),
            "brain_ml": qtab(g["brain_ml"]),
            "voxel_volume_mm3_native": qtab(g["voxel_volume_mm3_native"]),
            "voxel_volume_mm3_resampled": qtab(g["voxel_volume_mm3_resampled"]),
            "trace_z_p50": qtab(g["trace_z_p50"]), "adc_z_p50": qtab(g["adc_z_p50"]),
            "trace_z_lesion_p50": qtab(g["trace_z_lesion_p50"]), "adc_z_lesion_p50": qtab(g["adc_z_lesion_p50"]),
            "trace_raw_med": qtab(g["trace_raw_med"]), "adc_raw_med": qtab(g["adc_raw_med"]),
            "small_lesion_lt2ml_frac": float((g["lesion_ml"] < 2).mean()),
            "etiology_counts": g["etiology_code"].value_counts(dropna=False).to_dict() if "etiology_code" in g else {},
            "manufacturer_counts": g["trace_Manufacturer"].value_counts(dropna=False).to_dict() if "trace_Manufacturer" in g else {},
            "field_counts": g["trace_MagneticFieldStrength"].value_counts(dropna=False).to_dict() if "trace_MagneticFieldStrength" in g else {},
        }
    tab["test"] = {"n_subjects": len(sp["test"]), "note": "held out - no statistics computed (CLAUDE.md)"}
    json.dump(tab, open(a.out / "split_stats.json", "w"), indent=1, default=str)
    print(json.dumps(tab, indent=1, default=str))


if __name__ == "__main__":
    main()
