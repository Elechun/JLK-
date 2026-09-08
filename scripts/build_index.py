#!/usr/bin/env python3
"""Scan data/raw/ds004889 and write data/index.csv (one row per subject) + data/index_summary.json."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strokeai.data.index import build_index, join_participants, read_participants  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/ds004889"))
    ap.add_argument("--out", type=Path, default=Path("data/index.csv"))
    a = ap.parse_args()
    idx = build_index(a.raw)
    part = read_participants(a.raw / "participants.tsv")
    df = join_participants(idx, part)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    summ = {
        "n_subjects_with_imaging_dir": int(len(df)),
        "n_in_participants_tsv": int(df["in_participants_tsv"].sum()),
        "n_participants_tsv_rows": int(len(part)),
        "n_tsv_rows_without_imaging": int(len(set(part["participant_id"]) - set(df["participant_id"]))),
        "has_trace": int(df["has_trace"].sum()), "has_adc": int(df["has_adc"].sum()),
        "has_mask_acute": int(df["has_mask_acute"].sum()), "has_mask_chronic": int(df["has_mask_chronic"].sum()),
        "acute_mask_empty": int((df["mask_acute_voxels"] == 0).sum()),
        "adc_shape_mismatch": int((~df["adc_shape_matches_trace"].fillna(True).astype(bool)).sum()),
        "mask_shape_mismatch": int((~df["mask_shape_matches_trace"].fillna(True).astype(bool)).sum()),
        "mask_affine_mismatch": int((~df["mask_affine_matches_trace"].fillna(True).astype(bool)).sum()),
        "trace_ndim_counts": df["trace_ndim"].value_counts(dropna=False).to_dict(),
        "etiology_counts": df["etiology_code"].value_counts(dropna=False).to_dict(),
        "manufacturer_counts": df["trace_Manufacturer"].value_counts(dropna=False).to_dict(),
        "field_strength_counts": df["trace_MagneticFieldStrength"].value_counts(dropna=False).to_dict(),
        "slice_thickness_counts": df["trace_SliceThickness"].value_counts(dropna=False).to_dict(),
        "inplane_shape_top": df["trace_shape"].astype(str).value_counts().head(10).to_dict(),
    }
    with open(a.out.with_name("index_summary.json"), "w") as f:
        json.dump(summ, f, indent=2, default=str)
    print(json.dumps(summ, indent=2, default=str))


if __name__ == "__main__":
    main()
