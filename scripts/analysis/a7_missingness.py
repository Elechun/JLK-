#!/usr/bin/env python3
"""A7 - is the discharge-mRS label missing at random?

Compares the 620 eligible subjects who HAVE `gs_rankin_6isdeath` against the 831 who do not, on the
covariates that are available for the whole cohort (lesion volume, NIHSS, age, sex, prior stroke,
manufacturer, field strength, TOAST etiology, tsv registration).

Scope note (CLAUDE.md / charter):
  * this reads `data/index.csv` ONLY - no image, no mask voxel, and in particular no mRS *value*.
    Group membership is the *presence* indicator `gs_rankin_6isdeath.notna()`, which is metadata.
  * the charter allows whole-cohort metadata aggregates (A2 already published this table's
    left half); it forbids looking at mRS *values* split by train/val/test.  This script therefore
    never groups an outcome by split.  `--by-split` only prints availability COUNTS.

Reported per variable: group means/medians, the standardised mean difference (SMD, Cohen's d style)
and a two-sided test (Mann-Whitney U for continuous, chi-square for categorical).  |SMD| > 0.10 is the
conventional "imbalanced" flag in the propensity-score literature (Austin 2009).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MRS = "gs_rankin_6isdeath"
CONT = ["mask_acute_ml", "mask_acute_n_components", "mask_acute_n_slices", "nihss", "age", "bmi",
        "voxel_volume_mm3", "trace_SliceThickness"]
CAT = ["sex", "priorstroke", "trace_Manufacturer", "trace_MagneticFieldStrength", "etiology_code",
       "in_participants_tsv", "race"]


def smd_cont(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def smd_bin(p1: float, p2: float) -> float:
    d = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return float((p1 - p2) / d) if d > 0 else float("nan")


def describe_cont(s: pd.Series) -> dict:
    v = s.dropna().to_numpy(float)
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "n_missing": int(s.isna().sum()), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
            "median": float(np.median(v)),
            "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/index.csv"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--out", type=Path, default=Path("results/prognosis/missingness.json"))
    a = ap.parse_args()

    idx = pd.read_csv(a.index)
    sp = json.load(open(a.splits))
    elig = idx[idx["has_mask_acute"].astype(bool)].copy()
    assert len(elig) == sp["n_eligible"], (len(elig), sp["n_eligible"])
    has = elig[MRS].notna()
    g1, g0 = elig[has], elig[~has]

    out = {"cohort": {"n_eligible": int(len(elig)), "n_mrs_present": int(has.sum()),
                      "n_mrs_missing": int((~has).sum()),
                      "availability_counts_by_split": {k: {"n": len(sp[k]),
                                                           "n_mrs_present": int(elig[elig["participant_id"].isin(sp[k])][MRS].notna().sum())}
                                                       for k in ("train", "val", "test")},
                      "note": "only COUNTS by split; no mRS value is grouped by split (charter)"},
           "continuous": {}, "categorical": {}}

    for c in CONT:
        d1, d0 = describe_cont(g1[c]), describe_cont(g0[c])
        v1, v0 = g1[c].dropna().to_numpy(float), g0[c].dropna().to_numpy(float)
        p = float(stats.mannwhitneyu(v1, v0, alternative="two-sided").pvalue) if len(v1) > 1 and len(v0) > 1 else float("nan")
        out["continuous"][c] = {"mrs_present": d1, "mrs_missing": d0, "smd": smd_cont(v1, v0),
                                "mannwhitney_p": p}
        # a log transform for the strongly skewed lesion volume (A2: 0.067-557 mL)
        if c == "mask_acute_ml":
            out["continuous"]["log1p_mask_acute_ml"] = {
                "mrs_present": describe_cont(np.log1p(g1[c])), "mrs_missing": describe_cont(np.log1p(g0[c])),
                "smd": smd_cont(np.log1p(v1), np.log1p(v0)), "mannwhitney_p": p}

    for c in CAT:
        t1 = g1[c].value_counts(dropna=False)
        t0 = g0[c].value_counts(dropna=False)
        levels = sorted({str(x) for x in set(t1.index) | set(t0.index)})
        rows = {}
        for lv in levels:
            n1 = int(sum(v for k, v in t1.items() if str(k) == lv))
            n0 = int(sum(v for k, v in t0.items() if str(k) == lv))
            rows[lv] = {"n_present": n1, "frac_present": n1 / len(g1),
                        "n_missing": n0, "frac_missing": n0 / len(g0),
                        "smd": smd_bin(n1 / len(g1), n0 / len(g0))}
        obs = np.array([[rows[lv]["n_present"] for lv in levels], [rows[lv]["n_missing"] for lv in levels]])
        keep = obs.sum(0) > 0
        chi = stats.chi2_contingency(obs[:, keep]) if keep.sum() > 1 else None
        # An SMD against a group that is 60 % NaN on this very variable is uninterpretable, so every
        # categorical variable is ALSO compared among the subjects where it is observed.
        obs_lv = [lv for lv in levels if lv != "nan"]
        n1o = sum(rows[lv]["n_present"] for lv in obs_lv)
        n0o = sum(rows[lv]["n_missing"] for lv in obs_lv)
        obs_only = {lv: {"n_present": rows[lv]["n_present"], "frac_present": rows[lv]["n_present"] / n1o if n1o else float("nan"),
                         "n_missing": rows[lv]["n_missing"], "frac_missing": rows[lv]["n_missing"] / n0o if n0o else float("nan"),
                         "smd": smd_bin(rows[lv]["n_present"] / n1o, rows[lv]["n_missing"] / n0o) if n1o and n0o else float("nan")}
                    for lv in obs_lv}
        oarr = np.array([[obs_only[lv]["n_present"] for lv in obs_lv], [obs_only[lv]["n_missing"] for lv in obs_lv]])
        okeep = oarr.sum(0) > 0
        ochi = stats.chi2_contingency(oarr[:, okeep]) if okeep.sum() > 1 and oarr.sum(1).min() > 0 else None
        out["categorical"][c] = {"levels": rows, "chi2_p": float(chi.pvalue) if chi else float("nan"),
                                 "observed_only": {"n_present": n1o, "n_missing": n0o, "levels": obs_only,
                                                   "chi2_p": float(ochi.pvalue) if ochi else float("nan")}}

    # ---- the A2 "single block" link: 399 subjects have EVERY clinical variable missing ------------
    core = ["sex", "age", "nihss", "priorstroke", "acuteischaemicstroke"]
    allmiss = idx[core].isna().all(axis=1)
    block = idx[allmiss & idx["has_mask_acute"].astype(bool)]
    out["a2_block"] = {
        "n_all_clinical_missing_in_index": int(allmiss.sum()),
        "n_all_clinical_missing_and_eligible": int(len(block)),
        "n_of_those_with_mrs": int(block[MRS].notna().sum()),
        "median_lesion_ml": float(block["mask_acute_ml"].median()),
        "median_lesion_ml_rest_eligible": float(elig[~elig["participant_id"].isin(block["participant_id"])]["mask_acute_ml"].median()),
        "frac_3T": float((block["trace_MagneticFieldStrength"] == 3.0).mean()),
        "frac_3T_rest_eligible": float((elig[~elig["participant_id"].isin(block["participant_id"])]["trace_MagneticFieldStrength"] == 3.0).mean()),
        "note": "A2 F5: the clinical-variable block is missing for exactly the same subjects; "
                "this quantifies how much of the mRS missingness it explains",
    }
    # how much of the 831 is explained by the all-clinical-missing block?
    miss_ids = set(g0["participant_id"])
    out["a2_block"]["n_mrs_missing_inside_block"] = int(len(miss_ids & set(block["participant_id"])))
    out["a2_block"]["n_mrs_missing_outside_block"] = int(len(miss_ids) - len(miss_ids & set(block["participant_id"])))

    # ---- the mRS-missing group split into "block" vs "clinical present but mRS missing" -----------
    outside = g0[~g0["participant_id"].isin(block["participant_id"])]
    out["mrs_missing_strata"] = {}
    for name, grp in [("mrs_missing_in_all_clinical_block", g0[g0["participant_id"].isin(block["participant_id"])]),
                      ("mrs_missing_with_clinical", outside),
                      ("mrs_present", g1)]:
        out["mrs_missing_strata"][name] = {
            "n": int(len(grp)),
            "median_lesion_ml": float(grp["mask_acute_ml"].median()),
            "median_nihss": float(grp["nihss"].median()) if grp["nihss"].notna().any() else None,
            "median_age": float(grp["age"].median()) if grp["age"].notna().any() else None,
            "frac_3T": float((grp["trace_MagneticFieldStrength"] == 3.0).mean()),
            "frac_philips": float((grp["trace_Manufacturer"] == "Philips").mean()),
            "frac_etiology_labelled": float(grp["etiology_code"].isin(["LAA", "CE", "SVO", "OtherDet"]).mean()),
        }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
