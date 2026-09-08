#!/usr/bin/env python3
"""A2: recompute acute-lesion volumes directly from the mask NIfTIs (independent of data/index.csv).

Reads every derivatives/lesion_masks/<sub>/dwi/<sub>_space-TRACE_desc-lesionAcute_mask.nii.gz,
counts voxels > 0, and multiplies by the voxel volume taken from the *mask* header (index.csv uses
the TRACE header, so this is a genuinely independent path). Writes a CSV.

A5b 2026-09-09 (A5a item 2): the first version of this script had NO split filter and was run on
2026-09-09 02:55 KST, after data/splits.json existed, so it re-read the acute masks of all 1,451 subjects
including the 218 held-out ones (volumes only; nothing was trained, selected or scored on them).  It now
reads the train+val subjects of data/splits.json by default; `--include-test` is required to read test.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/analysis/a2_volumes.py --out /tmp/.../a2_volumes.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/ds004889"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--include-test", action="store_true",
                    help="also read the held-out test masks (only at the final evaluation step, CLAUDE.md)")
    a = ap.parse_args()

    import json

    sp = json.loads(a.splits.read_text())
    allowed = set(sp["train"]) | set(sp["val"]) | (set(sp["test"]) if a.include_test else set())
    mroot = a.raw / "derivatives" / "lesion_masks"
    rows = []
    subs = sorted([p for p in mroot.glob("sub-*") if p.is_dir() and p.name in allowed], key=lambda p: int(p.name[4:]))
    for sub in subs:
        sid = sub.name
        p = sub / "dwi" / f"{sid}_space-TRACE_desc-lesionAcute_mask.nii.gz"
        if not p.exists():
            continue
        img = nib.load(str(p))
        arr = np.asanyarray(img.dataobj)
        vox_mm3 = float(np.prod(img.header.get_zooms()[:3]))
        m = arr > 0
        lab, ncomp = ndimage.label(m)
        comp_sizes = np.bincount(lab.ravel())[1:] if ncomp else np.array([0])
        nz = np.unique(arr[arr != 0])
        rows.append(
            dict(
                participant_id=sid,
                a2_voxels=int(m.sum()),
                a2_vox_mm3=vox_mm3,
                a2_ml=float(m.sum() * vox_mm3 / 1000.0),
                a2_n_components=int(ncomp),
                a2_largest_comp_ml=float(comp_sizes.max() * vox_mm3 / 1000.0),
                a2_n_slices=int((m.sum(axis=(0, 1)) > 0).sum()),
                a2_nonzero_values=str(sorted(int(v) for v in nz)),
                a2_max_extent_mm=float(
                    max(
                        (np.ptp(np.where(m.any(axis=(1, 2)))[0]) + 1) * img.header.get_zooms()[0],
                        (np.ptp(np.where(m.any(axis=(0, 2)))[0]) + 1) * img.header.get_zooms()[1],
                        (np.ptp(np.where(m.any(axis=(0, 1)))[0]) + 1) * img.header.get_zooms()[2],
                    )
                )
                if m.any()
                else 0.0,
            )
        )
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}  n={len(df)}")


if __name__ == "__main__":
    main()
