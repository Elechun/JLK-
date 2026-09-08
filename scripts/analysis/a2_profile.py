#!/usr/bin/env python3
"""A2 data-suitability profiling: cohort counts, missingness, label/volume confounds.

Read-only. Never writes into data/. Prints every table it computes so the numbers in
docs/agents/A2_report.md are reproducible from stdout.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/analysis/a2_profile.py \
        --volumes /tmp/.../a2_volumes.csv [--section all|counts|missing|volume|scanner|auc|leak]

Test-set rule (CLAUDE.md): any split-wise statistic is restricted to train+val.
Cohort-wide descriptive statistics over all 1,715 / 1,451 subjects are allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

CLS4 = ["LAA", "CE", "SVO", "OtherDet"]


def load(volumes: Path):
    idx = pd.read_csv("data/index.csv")
    # pandas maps the literal string "n/a" to NaN, which would merge two very different groups
    # (tsv row exists but etiology is 'n/a'  vs.  subject absent from participants.tsv). Restore it.
    idx["etiology_code"] = np.where(
        idx["etiology_code"].notna(), idx["etiology_code"],
        np.where(idx["in_participants_tsv"], "n/a", "NOT_IN_TSV"))
    vol = pd.read_csv(volumes)
    df = idx.merge(vol, on="participant_id", how="left")
    sp = json.loads(Path("data/splits.json").read_text())
    trainval = set(sp["train"]) | set(sp["val"])
    df["split"] = np.where(
        df["participant_id"].isin(sp["train"]), "train",
        np.where(df["participant_id"].isin(sp["val"]), "val",
                 np.where(df["participant_id"].isin(sp["test"]), "test", "none")))
    df["inplane_mm"] = df["trace_zooms"].str.strip("()").str.split(",").str[0].astype(float)
    df["eti4"] = np.where(df["etiology_code"].isin(CLS4), df["etiology_code"], np.nan)
    return df, sp, trainval


def sec_counts(df, sp):
    print("\n===== 1. COHORT COUNTS (recomputed from data/index.csv) =====")
    n = len(df)
    print(f"n_subject_dirs (imaging)        : {n}")
    print(f"has_trace / has_adc            : {int(df.has_trace.sum())} / {int(df.has_adc.sum())}")
    print(f"has_mask_acute                 : {int(df.has_mask_acute.sum())}")
    print(f"has_mask_chronic               : {int(df.has_mask_chronic.sum())}")
    print(f"has_mask_any (combined file)   : {int(df.has_mask_any.sum())}")
    print(f"acute & chronic both           : {int((df.has_mask_acute & df.has_mask_chronic).sum())}")
    print(f"chronic only (no acute)        : {int((~df.has_mask_acute & df.has_mask_chronic).sum())}")
    print(f"no acute mask                  : {int((~df.has_mask_acute).sum())}")
    print(f"acute mask empty (voxels==0)   : {int((df.mask_acute_voxels == 0).sum())}")
    print(f"in participants.tsv            : {int(df.in_participants_tsv.sum())}  (tsv rows: "
          f"{len(pd.read_csv('data/raw/ds004889/participants.tsv', sep=chr(9)))})")
    print(f"shape/affine mismatches        : adc={int((~df.adc_shape_matches_trace.fillna(True)).sum())} "
          f"mask={int((~df.mask_shape_matches_trace.fillna(True)).sum())} "
          f"affine={int((~df.mask_affine_matches_trace.fillna(True)).sum())}")
    print(f"trace_ndim value counts        : {df.trace_ndim.value_counts().to_dict()}")
    print(f"trace extra dims all singleton : {int(df.trace_extra_dims_singleton.sum())}")
    print("\n-- etiology_code (all 1715) --");  print(df.etiology_code.value_counts(dropna=False))
    print("\n-- manufacturer --");              print(df.trace_Manufacturer.value_counts(dropna=False))
    print("\n-- field strength --");            print(df.trace_MagneticFieldStrength.value_counts(dropna=False))
    print("\n-- slice thickness --");           print(df.trace_SliceThickness.value_counts(dropna=False))
    print("\n-- spacing between slices --");    print(df.trace_SpacingBetweenSlices.value_counts(dropna=False))
    print("\n-- in-plane matrix (top 12) --");  print(df.trace_shape3.value_counts().head(12))
    print("\n-- in-plane voxel size mm (describe) --"); print(df.inplane_mm.describe())
    print("\n-- in-plane voxel size mm (value counts, rounded 3dp) --")
    print(df.inplane_mm.round(3).value_counts().head(12))
    print("\n-- model name (top) --");          print(df.trace_ManufacturersModelName.value_counts(dropna=False).head(12))
    print("\n-- n slices --");                  print(df.trace_shape3.str.split(",").str[2].str.strip(" )").astype(int).value_counts())
    elig = df[df.split != "none"]
    print(f"\neligible for segmentation (in splits.json): {len(elig)}")
    print("split sizes:", elig.split.value_counts().to_dict())
    print("etiology within eligible:"); print(elig.etiology_code.value_counts(dropna=False))
    print(f"eligible with 4-class etiology: {int(elig.eti4.notna().sum())}")


def sec_missing(df):
    print("\n===== 2. CLINICAL VARIABLE MISSINGNESS =====")
    cols = ["sex", "age", "race", "acuteischaemicstroke", "priorstroke", "bmi", "nihss",
            "gs_rankin_6isdeath", "etiology_code"]
    for scope, sub in [("all imaging (1715)", df),
                       ("in participants.tsv", df[df.in_participants_tsv]),
                       ("eligible (mask, 1451)", df[df.split != "none"]),
                       ("eligible & 4-class label", df[(df.split != "none") & df.eti4.notna()])]:
        rows = []
        for c in cols:
            s = sub[c]
            miss = s.isna() if c != "etiology_code" else s.isin(["n/a"]) | s.isna()
            rows.append(dict(var=c, n=len(sub), missing=int(miss.sum()),
                             pct=round(100 * miss.mean(), 1)))
        print(f"\n-- {scope} --")
        print(pd.DataFrame(rows).to_string(index=False))
    print("\n-- age '89+' clamped --", int(df.age_clamped_89plus.fillna(False).sum()))
    print("\n-- clinical distributions (eligible 1451) --")
    e = df[df.split != "none"]
    print(e[["age", "nihss", "bmi", "gs_rankin_6isdeath"]].describe().round(2))
    print("sex:", e.sex.value_counts(dropna=False).to_dict())
    print("race:", e.race.value_counts(dropna=False).to_dict())
    print("priorstroke:", e.priorstroke.value_counts(dropna=False).to_dict())
    print("acuteischaemicstroke:", e.acuteischaemicstroke.value_counts(dropna=False).to_dict())
    print("\n-- 2c. WHO are the label-missing? (all 1715) --")
    df = df.copy()
    df["label_group"] = np.where(~df.in_participants_tsv, "not_in_tsv",
                                 np.where(df.etiology_code == "n/a", "tsv_eti_na",
                                          np.where(df.etiology_code == "Cryptogenic", "Cryptogenic", "4class")))
    print(df.label_group.value_counts())
    agg = df.groupby("label_group").agg(
        n=("participant_id", "size"),
        has_acute=("has_mask_acute", "sum"),
        acute_rate=("has_mask_acute", "mean"),
        median_ml=("a2_ml", "median"),
        mean_ml=("a2_ml", "mean"),
        median_age=("age", "median"),
        median_nihss=("nihss", "median"),
        nihss_missing=("nihss", lambda s: s.isna().mean()),
        pct_3T=("trace_MagneticFieldStrength", lambda s: (s == 3.0).mean()),
        pct_philips=("trace_Manufacturer", lambda s: (s == "Philips").mean()),
        median_subject_num=("subject_num", "median"),
    ).round(3)
    print(agg.to_string())
    print("\n-- acuteischaemicstroke x has_mask_acute (tsv rows only) --")
    t = df[df.in_participants_tsv]
    print(pd.crosstab(t.acuteischaemicstroke, t.has_mask_acute, dropna=False))
    print("\n-- subject_num deciles by label_group (is missingness a time/batch effect?) --")
    df["subj_decile"] = pd.qcut(df.subject_num, 10, labels=False) + 1
    print(pd.crosstab(df.subj_decile, df.label_group))


def sec_volume(df):
    print("\n===== 3. LESION VOLUME BY ETIOLOGY (direct from masks; eligible cohort 1451) =====")
    e = df[df.split != "none"].copy()
    print("index.csv vs A2 recomputation:")
    d = (e.mask_acute_ml - e.a2_ml).abs()
    print(f"  n compared={len(e)}  max |diff| mL = {d.max():.3e}  n(diff>1e-6) = {int((d > 1e-6).sum())}")
    dv = (e.mask_acute_voxels - e.a2_voxels).abs()
    print(f"  voxel counts identical: {bool((dv == 0).all())}")

    def q(s):
        return pd.Series({"n": len(s), "mean": s.mean(), "sd": s.std(), "min": s.min(),
                          "p25": s.quantile(.25), "median": s.median(), "p75": s.quantile(.75),
                          "p90": s.quantile(.90), "max": s.max()})
    g = e.groupby(e.etiology_code)["a2_ml"].apply(q).unstack().round(3)
    print("\n-- volume (mL) by etiology, eligible cohort --")
    print(g.to_string())
    print("\n-- fraction below thresholds --")
    for thr in [1.0, 1.77, 2.0, 4.19, 5.0, 10.0]:
        s = e.groupby(e.etiology_code)["a2_ml"].apply(lambda x: (x < thr).mean()).round(3)
        print(f"  < {thr:>5} mL :", s.to_dict())
    print("\n-- max lesion extent (mm) by etiology; TOAST SVO is defined as <15 mm --")
    print(e.groupby(e.etiology_code)["a2_max_extent_mm"].describe().round(2).to_string())
    for thr in [15.0, 20.0]:
        s = e.groupby(e.etiology_code)["a2_max_extent_mm"].apply(lambda x: (x < thr).mean()).round(3)
        print(f"  extent < {thr} mm :", s.to_dict())
    print("\n-- n components / n slices by etiology --")
    print(e.groupby(e.etiology_code)[["a2_n_components", "a2_n_slices"]].median().to_string())
    print("\n-- Kruskal-Wallis on log volume across 4 classes --")
    groups = [np.log10(e.loc[e.etiology_code == c, "a2_ml"] + 1e-3) for c in CLS4]
    H, p = stats.kruskal(*groups)
    print(f"  H={H:.2f} p={p:.3e}")
    print("  Mann-Whitney SVO vs each:")
    for c in ["LAA", "CE", "OtherDet"]:
        u, pu = stats.mannwhitneyu(e.loc[e.etiology_code == "SVO", "a2_ml"],
                                   e.loc[e.etiology_code == c, "a2_ml"])
        n1 = (e.etiology_code == "SVO").sum(); n2 = (e.etiology_code == c).sum()
        print(f"    SVO vs {c:9s} U={u:.0f} p={pu:.3e}  AUC(volume, SVO smaller)= {1 - u / (n1 * n2):.3f}")


def sec_scanner(df):
    print("\n===== 4. SCANNER / FIELD / RESOLUTION vs LABEL and LESION SIZE (eligible 1451) =====")
    e = df[df.split != "none"].copy()
    e["field"] = e.trace_MagneticFieldStrength.astype(str)
    print("\n-- etiology x manufacturer --")
    ct = pd.crosstab(e.etiology_code, e.trace_Manufacturer)
    print(ct.to_string()); print("chi2 p =", stats.chi2_contingency(ct)[1])
    print("\n-- etiology x field strength --")
    ct = pd.crosstab(e.etiology_code, e.field)
    print(ct.to_string())
    print("row % 3T:", (ct.get("3.0", pd.Series(0, index=ct.index)) / ct.sum(axis=1)).round(3).to_dict())
    print("chi2 p =", stats.chi2_contingency(ct)[1])
    print("\n-- 4-class only, etiology x field --")
    ct4 = pd.crosstab(e.loc[e.eti4.notna(), "etiology_code"], e.loc[e.eti4.notna(), "field"])
    print(ct4.to_string()); print("chi2 p =", stats.chi2_contingency(ct4)[1])
    print("\n-- lesion volume by field strength --")
    print(e.groupby("field")["a2_ml"].describe().round(3).to_string())
    a = e.loc[e.field == "1.5", "a2_ml"]; b = e.loc[e.field == "3.0", "a2_ml"]
    u, p = stats.mannwhitneyu(a, b)
    print(f"  MW p={p:.3e}  AUC(field predicts volume)= {u / (len(a) * len(b)):.3f}")
    print("\n-- lesion volume by manufacturer --")
    print(e.groupby("trace_Manufacturer")["a2_ml"].describe().round(3).to_string())
    print("\n-- in-plane voxel size vs volume (Spearman) --")
    r, p = stats.spearmanr(e.inplane_mm, e.a2_ml)
    print(f"  rho={r:.3f} p={p:.3e}")
    print("  in-plane voxel size by field:"); print(e.groupby("field")["inplane_mm"].describe().round(3).to_string())
    print("\n-- in-plane voxel size by etiology --")
    print(e.groupby(e.etiology_code)["inplane_mm"].median().round(4).to_string())
    print("  Kruskal p =", stats.kruskal(*[e.loc[e.etiology_code == c, "inplane_mm"] for c in CLS4])[1])
    print("\n-- voxel volume mm3 describe (affects mL quantisation) --")
    print(e.a2_vox_mm3.describe().round(4).to_string())
    print("  smallest detectable non-zero volume (1 voxel) range mL:",
          round(e.a2_vox_mm3.min() / 1000, 5), "-", round(e.a2_vox_mm3.max() / 1000, 5))
    print("\n-- 3T rate by subject_num decile (scanner drift over accrual) --")
    e["subj_decile"] = pd.qcut(e.subject_num, 10, labels=False) + 1
    print(e.groupby("subj_decile").agg(n=("participant_id", "size"),
                                       pct3T=("field", lambda s: (s == "3.0").mean()),
                                       median_ml=("a2_ml", "median")).round(3).to_string())


def sec_auc(df):
    print("\n===== 5. LEAKAGE QUANTIFICATION: how much does volume alone buy? =====")
    print("(split-wise: TRAIN+VAL ONLY, test never touched)")
    e = df[df.split.isin(["train", "val"])].copy()
    lab = e[e.eti4.notna()].copy()
    print("n train+val eligible:", len(e), " with 4-class label:", len(lab))
    print(lab.etiology_code.value_counts().to_dict())
    feats = {
        "volume_mL": lab.a2_ml.values,
        "log10_volume": np.log10(lab.a2_ml.values + 1e-3),
        "max_extent_mm": lab.a2_max_extent_mm.values,
        "n_slices": lab.a2_n_slices.values,
        "largest_component_mL": lab.a2_largest_comp_ml.values,
    }
    print("\n-- one-vs-rest AUC of a SINGLE image-derived feature (GT mask) --")
    rows = []
    for fname, x in feats.items():
        r = {"feature": fname}
        aucs = []
        for c in CLS4:
            y = (lab.etiology_code == c).astype(int).values
            a = roc_auc_score(y, x)
            a = max(a, 1 - a)  # direction-free separability
            r[c] = round(a, 3)
            aucs.append(a)
        r["macroOvR"] = round(float(np.mean(aucs)), 3)
        yb = lab.etiology_code.isin(["LAA", "CE"])
        y2 = (lab.loc[yb, "etiology_code"] == "LAA").astype(int).values
        x2 = x[yb.values]
        a2 = roc_auc_score(y2, x2); r["LAAvsCE"] = round(max(a2, 1 - a2), 3)
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n-- same, with Cryptogenic INCLUDED as a 5th class (for reference) --")
    lab5 = e[e.etiology_code.isin(CLS4 + ["Cryptogenic"])]
    rows = []
    for fname, x in {"volume_mL": lab5.a2_ml.values, "log10_volume": np.log10(lab5.a2_ml.values + 1e-3)}.items():
        r = {"feature": fname}
        for c in CLS4 + ["Cryptogenic"]:
            y = (lab5.etiology_code == c).astype(int).values
            a = roc_auc_score(y, x); r[c] = round(max(a, 1 - a), 3)
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n-- clinical-only baselines (age, sex, NIHSS, priorstroke), single feature --")
    for cvar in ["age", "nihss"]:
        s = lab[cvar]
        ok = s.notna().values
        r = {"feature": cvar, "n": int(ok.sum())}
        for c in CLS4:
            y = (lab.etiology_code == c).astype(int).values[ok]
            a = roc_auc_score(y, s.values[ok]); r[c] = round(max(a, 1 - a), 3)
        print(r)
    print("\n-- SVO share below TOAST-implied size cutoffs (train+val, 4-class) --")
    for thr, name in [(1.77, "1.5cm sphere = 1.77 mL"), (4.19, "2.0cm sphere = 4.19 mL")]:
        below = lab.a2_ml < thr
        ct = pd.crosstab(lab.etiology_code, below)
        print(f"  {name}: "); print(ct.to_string())
        if True in ct.columns:
            prec = ct.loc["SVO", True] / ct[True].sum()
            rec = ct.loc["SVO", True] / ct.loc["SVO"].sum()
            print(f"    rule 'volume<{thr} -> SVO': precision={prec:.3f} recall={rec:.3f}")


def sec_leak(df):
    print("\n===== 6. MASK LABEL VALUES (unresolved item) =====")
    e = df[df.has_mask_acute == True]  # noqa: E712
    print(e.a2_nonzero_values.value_counts().to_string())
    odd = e[e.a2_nonzero_values != "[1]"]
    print("\nodd subjects:")
    print(odd[["participant_id", "subject_num", "a2_nonzero_values", "a2_voxels", "a2_ml",
               "a2_n_components", "has_mask_chronic", "etiology_code", "split"]].to_string(index=False))
    print("\nneighbourhood of the odd blocks (subject_num 495-515 and 1245-1265):")
    nb = e[(e.subject_num.between(498, 514)) | (e.subject_num.between(1248, 1262))]
    print(nb[["participant_id", "a2_nonzero_values", "a2_voxels", "trace_ManufacturersModelName",
              "trace_MagneticFieldStrength"]].to_string(index=False))
    print("\nEach odd mask has exactly ONE non-zero value:",
          bool(odd.a2_nonzero_values.str.count(",").eq(0).all()))
    print("Any acute mask with >1 non-zero value in whole cohort:",
          int((e.a2_nonzero_values.str.count(",") > 0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes", type=Path, required=True)
    ap.add_argument("--section", default="all")
    a = ap.parse_args()
    df, sp, _ = load(a.volumes)
    print("index.csv sha256   :", hashlib.sha256(Path("data/index.csv").read_bytes()).hexdigest())
    print("splits.json field  :", sp["sha256"], " n:", sp["n"])
    s = a.section
    if s in ("all", "counts"):
        sec_counts(df, sp)
    if s in ("all", "missing"):
        sec_missing(df)
    if s in ("all", "volume"):
        sec_volume(df)
    if s in ("all", "scanner"):
        sec_scanner(df)
    if s in ("all", "auc"):
        sec_auc(df)
    if s in ("all", "leak"):
        sec_leak(df)


if __name__ == "__main__":
    main()
