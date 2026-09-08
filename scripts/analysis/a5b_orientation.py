#!/usr/bin/env python3
"""A5b - which cache axis is the patient's left-right?  (train/val only, never test)

1. Synthetic subject: a NIfTI whose lesion sits at small RAS-x (= patient LEFT) is pushed through the
   real `preprocess_subject`; we then check where the lesion lands in the cached (S, H, W) mask and what
   `lesion_features` reports as `laterality_left_frac`.
2. Real train subjects: `nib.as_closest_canonical` axis codes, plus the distribution of the lesion's
   fraction on the low-H side vs the low-W side over all train+val GT masks.  Strokes are overwhelmingly
   unilateral, so the *true* left-right axis is the one whose fraction piles up at 0 and 1.

    PYTHONPATH=src .venv/bin/python scripts/analysis/a5b_orientation.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from strokeai.data.preprocess import preprocess_subject  # noqa: E402
from strokeai.features import lesion_features  # noqa: E402


def synthetic(tmp: Path) -> dict:
    rng = np.random.default_rng(0)
    X, Y, Z = 48, 40, 6
    aff = np.diag([2.0, 2.0, 5.0, 1.0])  # already RAS: axis0 = x (L->R), axis1 = y (P->A), axis2 = z
    tr = rng.normal(200.0, 30.0, (X, Y, Z)).astype(np.float32)
    ad = rng.normal(900.0, 120.0, (X, Y, Z)).astype(np.float32)
    mk = np.zeros((X, Y, Z), np.float32)
    mk[4:12, 16:24, 2:4] = 1  # small x -> patient LEFT ; centred in y
    for n, d in [("tr", tr[..., None]), ("ad", ad[..., None]), ("mk", mk)]:
        nib.save(nib.Nifti1Image(d, aff), str(tmp / f"{n}.nii.gz"))
    r = preprocess_subject(tmp / "tr.nii.gz", tmp / "ad.nii.gz", tmp / "mk.nii.gz", size=32)
    m = r["mask"]  # (S, H, W)
    zs, hs, ws = np.nonzero(m)
    f = lesion_features(m, r["voxel_volume_mm3"])
    return {"cache_mask_shape": list(m.shape), "lesion_H_range": [int(hs.min()), int(hs.max())], "H": int(m.shape[1]),
            "lesion_W_range": [int(ws.min()), int(ws.max())], "W": int(m.shape[2]),
            "lesion_on_low_H_side": bool(hs.max() < m.shape[1] / 2), "lesion_on_low_W_side": bool(ws.max() < m.shape[2] / 2),
            "laterality_left_frac_reported": f["laterality_left_frac"], "expected_if_left_right_is_H": 1.0}


def real(cache: Path, raw: Path, ids: list[str]) -> dict:
    # axis codes after canonicalisation on a handful of real subjects (header only, no voxels)
    codes = {}
    for sid in ids[:5]:
        img = nib.load(str(raw / sid / "dwi" / f"{sid}_rec-TRACE_dwi.nii.gz"))
        codes[sid] = {"raw": "".join(nib.aff2axcodes(img.affine)),
                      "canonical": "".join(nib.aff2axcodes(nib.as_closest_canonical(img).affine))}
    lowH, lowW, n_bil_H, n_bil_W = [], [], 0, 0
    for sid in ids:
        m = np.load(cache / f"{sid}.npz")["mask"].astype(bool)
        if not m.any():
            continue
        zs, hs, ws = np.nonzero(m)
        fh = float((hs < m.shape[1] / 2).mean())
        fw = float((ws < m.shape[2] / 2).mean())
        lowH.append(fh)
        lowW.append(fw)
    lowH, lowW = np.array(lowH), np.array(lowW)

    def summ(v):
        return {"n": int(len(v)), "frac_within_0.05_of_0_or_1": float(np.mean((v < 0.05) | (v > 0.95))),
                "frac_between_0.2_0.8": float(np.mean((v > 0.2) & (v < 0.8))),
                "hist_10bins": np.histogram(v, bins=10, range=(0, 1))[0].tolist()}

    return {"axis_codes": codes, "low_H_fraction": summ(lowH), "low_W_fraction": summ(lowW)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw/ds004889"))
    ap.add_argument("--out", type=Path, default=Path("results/a5b/orientation.json"))
    ap.add_argument("--tmp", type=Path, required=True)
    a = ap.parse_args()
    sp = json.load(open("data/splits.json"))
    ids = sp["train"] + sp["val"]  # never test
    a.tmp.mkdir(parents=True, exist_ok=True)
    out = {"synthetic": synthetic(a.tmp), "real_train_val": real(a.cache, a.raw, ids)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
