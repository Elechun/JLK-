#!/usr/bin/env python3
"""A2: quantify how much of the etiology-classification AUC comes from lesion SIZE alone.

5-fold stratified CV (seed 2026) of a multinomial logistic regression on the 4-class TOAST subset,
comparing feature sets. Ground-truth masks are used, so this is an upper bound on what a perfect
segmentation model could hand to the classifier.

TEST SET IS NEVER TOUCHED: only splits.json train+val subjects enter this script.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/analysis/a2_leakage.py --volumes /tmp/.../a2_volumes.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLS4 = ["LAA", "CE", "SVO", "OtherDet"]
SEED = 2026

FEATSETS = {
    "clinical_only          (age,sex,NIHSS,priorstroke)": ["age", "sex_M", "nihss", "priorstroke"],
    "size_only              (logV,extent,slices,largestC)": ["logV", "extent", "slices", "logLC"],
    "shape_no_size          (ncomp,LC/V,slice-density)": ["ncomp", "lc_frac", "slice_density"],
    "clinical+shape_no_size": ["age", "sex_M", "nihss", "priorstroke", "ncomp", "lc_frac", "slice_density"],
    "clinical+size          (FULL)": ["age", "sex_M", "nihss", "priorstroke", "logV", "extent", "slices", "logLC"],
    "volume_only            (1 feature)": ["logV"],
}


def build(volumes: Path) -> pd.DataFrame:
    idx = pd.read_csv("data/index.csv")
    idx["etiology_code"] = np.where(idx.etiology_code.notna(), idx.etiology_code,
                                    np.where(idx.in_participants_tsv, "n/a", "NOT_IN_TSV"))
    vol = pd.read_csv(volumes)
    sp = json.loads(Path("data/splits.json").read_text())
    keep = set(sp["train"]) | set(sp["val"])          # test excluded on purpose
    df = idx.merge(vol, on="participant_id")
    df = df[df.participant_id.isin(keep) & df.etiology_code.isin(CLS4)].copy()
    df["logV"] = np.log10(df.a2_ml + 1e-3)
    df["logLC"] = np.log10(df.a2_largest_comp_ml + 1e-3)
    df["extent"] = df.a2_max_extent_mm
    df["slices"] = df.a2_n_slices
    df["ncomp"] = df.a2_n_components
    df["lc_frac"] = df.a2_largest_comp_ml / df.a2_ml          # scale-free: fragmentation
    df["slice_density"] = df.a2_voxels / df.a2_n_slices / (df.a2_voxels ** (2 / 3))  # scale-free-ish
    df["sex_M"] = (df.sex == "M").astype(float)
    df["priorstroke"] = df.priorstroke.astype(float)
    return df


def cv_macro_auc(X: np.ndarray, y: np.ndarray, classes: list[str]) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    proba = np.zeros((len(y), len(classes)))
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        # align columns to `classes`
        cols = {c: i for i, c in enumerate(clf.classes_)}
        for j, c in enumerate(classes):
            proba[te, j] = p[:, cols[c]] if c in cols else 0.0
    out = {}
    aucs = []
    for j, c in enumerate(classes):
        a = roc_auc_score((y == c).astype(int), proba[:, j])
        out[c] = round(a, 3)
        aucs.append(a)
    out["macroOvR"] = round(float(np.mean(aucs)), 3)
    if {"LAA", "CE"} <= set(classes):
        m = np.isin(y, ["LAA", "CE"])
        score = proba[m, classes.index("LAA")] - proba[m, classes.index("CE")]
        out["LAAvsCE"] = round(roc_auc_score((y[m] == "LAA").astype(int), score), 3)
    return out


def run(df: pd.DataFrame, classes: list[str], title: str) -> None:
    d = df[df.etiology_code.isin(classes)].dropna(subset=sorted({f for v in FEATSETS.values() for f in v}))
    y = d.etiology_code.values
    print(f"\n--- {title}  n={len(d)}  {pd.Series(y).value_counts().to_dict()} ---")
    rows = []
    for name, feats in FEATSETS.items():
        r = {"feature_set": name}
        r.update(cv_macro_auc(d[feats].values.astype(float), y, classes))
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes", type=Path, required=True)
    a = ap.parse_args()
    df = build(a.volumes)
    print("train+val 4-class subjects:", len(df), df.etiology_code.value_counts().to_dict())
    run(df, CLS4, "4-class (LAA/CE/SVO/OtherDet)")
    run(df, ["LAA", "CE", "OtherDet"], "3-class, SVO REMOVED (how much size signal survives?)")


if __name__ == "__main__":
    main()
