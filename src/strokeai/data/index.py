"""Build the subject index: joins imaging files, sidecar metadata, mask statistics and participants.tsv."""
from __future__ import annotations

import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

ETIOLOGY_MAP = {  # participants.tsv 'etiology' -> short code (JBS-01K style: LAA / CE / SVO / Others)
    "1": "LAA",
    "2": "CE",
    "3": "SVO",
    "4": "OtherDet",
    "5": "Cryptogenic",
}


def parse_age(v: str) -> float:
    """SOOP clamps age to '89+' (HIPAA safe harbor). Map it to 89.0; 'n/a' -> NaN."""
    if v in ("n/a", "", None):
        return float("nan")
    return float(str(v).rstrip("+"))


def read_participants(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df["age"] = df["age"].map(parse_age)
    df["etiology_code"] = df["etiology"].str[:1].map(ETIOLOGY_MAP).fillna("n/a")
    for c in ["nihss", "bmi", "gs_rankin_6isdeath"]:
        df[c] = pd.to_numeric(df[c].replace("n/a", np.nan), errors="coerce")
    df["age_clamped_89plus"] = df["age"].eq(89.0)
    return df


def _sidecar(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        j = json.load(f)
    keys = ["Manufacturer", "ManufacturersModelName", "MagneticFieldStrength", "SliceThickness",
            "SpacingBetweenSlices", "RepetitionTime", "EchoTime", "ReconMatrixPE", "SeriesDescription"]
    return {f"trace_{k}": j.get(k) for k in keys}


def build_index(raw: Path) -> pd.DataFrame:
    rows = []
    subs = sorted([p for p in raw.glob("sub-*") if p.is_dir()], key=lambda p: int(p.name[4:]))
    for sub in subs:
        sid = sub.name
        trace = sub / "dwi" / f"{sid}_rec-TRACE_dwi.nii.gz"
        adc = sub / "dwi" / f"{sid}_rec-ADC_dwi.nii.gz"
        mdir = raw / "derivatives" / "lesion_masks" / sid / "dwi"
        m_acute = mdir / f"{sid}_space-TRACE_desc-lesionAcute_mask.nii.gz"
        m_chronic = mdir / f"{sid}_space-TRACE_desc-lesionChronic_mask.nii.gz"
        m_all = mdir / f"{sid}_space-TRACE_desc-lesion_mask.nii.gz"
        r = {"participant_id": sid, "has_trace": trace.exists(), "has_adc": adc.exists(),
             "has_mask_acute": m_acute.exists(), "has_mask_chronic": m_chronic.exists(), "has_mask_any": m_all.exists()}
        if trace.exists():
            img = nib.load(str(trace))
            # SOOP stores TRACE/ADC as 4-D (H, W, S, 1); masks are 3-D. Compare spatial dims only.
            r.update(trace_shape=tuple(int(x) for x in img.shape), trace_ndim=img.ndim,
                     trace_shape3=tuple(int(x) for x in img.shape[:3]),
                     trace_extra_dims_singleton=all(int(x) == 1 for x in img.shape[3:]),
                     trace_zooms=tuple(float(z) for z in img.header.get_zooms()[:3]))
            r["voxel_volume_mm3"] = float(np.prod(img.header.get_zooms()[:3]))
            r.update(_sidecar(trace.with_suffix("").with_suffix(".json")))
        if adc.exists():
            img = nib.load(str(adc))
            r.update(adc_shape=tuple(int(x) for x in img.shape), adc_ndim=img.ndim)
            r["adc_shape_matches_trace"] = r.get("trace_shape3") == tuple(int(x) for x in img.shape[:3])
        if m_acute.exists():
            mimg = nib.load(str(m_acute))
            m = np.asanyarray(mimg.dataobj) > 0
            r.update(mask_shape=tuple(int(x) for x in m.shape),
                     mask_shape_matches_trace=r.get("trace_shape3") == tuple(int(x) for x in m.shape[:3]),
                     mask_acute_voxels=int(m.sum()),
                     mask_acute_ml=float(m.sum() * r.get("voxel_volume_mm3", np.nan) / 1000.0),
                     mask_acute_n_components=int(ndimage.label(m)[1]),
                     mask_acute_n_slices=int((m.reshape(m.shape[0], m.shape[1], -1).sum(axis=(0, 1)) > 0).sum()) if m.ndim == 3 else -1,
                     mask_unique_values=str(sorted(np.unique(np.asanyarray(mimg.dataobj)).tolist())[:5]))
            if r.get("trace_shape3") == tuple(int(x) for x in m.shape[:3]):
                # affine agreement between mask and TRACE (should be identical: mask is in TRACE space)
                r["mask_affine_matches_trace"] = bool(np.allclose(mimg.affine, nib.load(str(trace)).affine, atol=1e-3))
        if m_chronic.exists():
            mc = np.asanyarray(nib.load(str(m_chronic)).dataobj) > 0
            r["mask_chronic_voxels"] = int(mc.sum())
        rows.append(r)
    df = pd.DataFrame(rows)
    return df


def join_participants(idx: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    df = idx.merge(participants, on="participant_id", how="left", indicator=True)
    df["in_participants_tsv"] = df["_merge"].eq("both")
    df = df.drop(columns=["_merge"])
    df["subject_num"] = df["participant_id"].str[4:].astype(int)
    return df.sort_values("subject_num").reset_index(drop=True)
