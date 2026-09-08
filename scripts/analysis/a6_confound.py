#!/usr/bin/env python3
"""A6 - protocol-leakage probe for the etiology classifier (A2 hand-off item 4b).

If scanner metadata alone predicts the etiology label above chance, then any classifier that can see
those variables (directly, or through an image feature that encodes them) is partly reading the
acquisition protocol instead of the disease.  The shipped feature sets deliberately contain no
scanner variable; this script measures how much signal they would have carried.

Also reports the train-vs-val lesion-volume distribution (A2 hand-off 4c) so a split imbalance cannot
be mistaken for a modelling effect.  train + val only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from strokeai.metrics import multiclass_auc  # noqa: E402

CLASSES = ["LAA", "CE", "SVO", "Others"]
CODE2Y = {"LAA": "LAA", "CE": "CE", "SVO": "SVO", "OtherDet": "Others"}


def cv_macro_auc(X: np.ndarray, y: np.ndarray, seed: int) -> dict:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(SimpleImputer(), StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, random_state=seed))
    proba = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")
    classes = sorted(set(y))
    present = [c for c in CLASSES if c in classes]
    return multiclass_auc(y, proba[:, [classes.index(c) for c in present]], present)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/index.csv"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=Path("results/a6/confound.json"))
    a = ap.parse_args()
    idx = pd.read_csv(a.index)
    sp = json.load(open(a.splits))
    dev_ids = set(sp["train"]) | set(sp["val"])
    d = idx[idx["participant_id"].isin(dev_ids)].copy()
    d["y"] = d["etiology_code"].map(CODE2Y)
    lab = d[d["y"].notna()].copy()

    out = {"n_dev": int(len(d)), "n_labelled": int(len(lab)),
           "class_counts": lab["y"].value_counts().to_dict(),
           "scanner_counts": lab["trace_Manufacturer"].value_counts().to_dict(),
           "field_counts": lab["trace_MagneticFieldStrength"].value_counts().to_dict()}

    probes = {
        "field_strength_only": ["trace_MagneticFieldStrength"],
        "voxel_geometry_only": ["voxel_volume_mm3"],
        "field_plus_geometry": ["trace_MagneticFieldStrength", "voxel_volume_mm3"],
    }
    out["protocol_probe_macro_auc"] = {}
    for name, cols in probes.items():
        X = lab[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
        out["protocol_probe_macro_auc"][name] = cv_macro_auc(X, lab["y"].values, a.seed)

    # A2 4c: is the lesion-size distribution comparable across the two dev splits?
    vol = {}
    for w in ("train", "val"):
        v = idx[idx["participant_id"].isin(sp[w])]["mask_acute_ml"].dropna().values
        vol[w] = {"n": int(len(v)), "median_ml": float(np.median(v)),
                  "p25_ml": float(np.percentile(v, 25)), "p75_ml": float(np.percentile(v, 75)),
                  "frac_lt_1ml": float(np.mean(v < 1)), "frac_lt_2ml": float(np.mean(v < 2)),
                  "frac_ge_50ml": float(np.mean(v >= 50))}
    from scipy.stats import mannwhitneyu

    vt = idx[idx["participant_id"].isin(sp["train"])]["mask_acute_ml"].dropna().values
    vv = idx[idx["participant_id"].isin(sp["val"])]["mask_acute_ml"].dropna().values
    out["lesion_volume_by_split"] = vol
    out["train_vs_val_mannwhitney_p"] = float(mannwhitneyu(vt, vv).pvalue)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
