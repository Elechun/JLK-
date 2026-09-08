#!/usr/bin/env python3
"""A6 - how much can the 218-subject test split actually decide?

Analytic standard errors for the charter's test-set criteria, so the thresholds are chosen with an
explicit power statement instead of by eye.

  * AUC        : Hanley & McNeil (1982) exponential approximation
  * Dice mean  : subject bootstrap SD measured on val, rescaled by sqrt(n_val / n_test)
  * ICC(2,1)   : subject bootstrap on val (same n) -> used directly

The etiology subgroup sizes for the test split are ESTIMATED from the dev proportions (the split is
stratified on `etiology_code | lesion-size band`, so proportions carry over); the test labels
themselves are NOT read.  Every number produced here is therefore marked "estimate" in the report.
"""
from __future__ import annotations

import argparse
import json
from math import erf, sqrt
from pathlib import Path

import numpy as np


def hanley_mcneil_se(auc: float, n1: int, n2: int) -> float:
    """SE of an AUC with n1 positives and n2 negatives (exponential-distribution approximation)."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    return sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n2 - 1) * (q2 - auc**2)) / (n1 * n2))


def norm_cdf(z: float) -> float:
    return 0.5 * (1 + erf(z / sqrt(2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-counts", type=json.loads, default='{"LAA":197,"CE":127,"SVO":71,"Others":52}')
    ap.add_argument("--n-dev-subjects", type=int, default=1233)
    ap.add_argument("--n-test-subjects", type=int, default=218)
    ap.add_argument("--cls", type=Path, default=Path("results/cls_pred_logreg.json"))
    ap.add_argument("--sweep", type=Path, default=Path("results/a6/threshold_sweep.json"))
    ap.add_argument("--out", type=Path, default=Path("results/a6/power.json"))
    a = ap.parse_args()
    dev = a.dev_counts if isinstance(a.dev_counts, dict) else json.loads(a.dev_counts)
    scale = a.n_test_subjects / a.n_dev_subjects
    est = {k: v * scale for k, v in dev.items()}
    n_lab = sum(est.values())
    cls = json.load(open(a.cls))

    out = {"note": "test etiology subgroup sizes are ESTIMATED from dev proportions; test labels were not read",
           "estimated_test_counts": {k: round(v, 1) for k, v in est.items()},
           "estimated_test_labelled_subjects": round(n_lab, 1), "criteria": {}}

    # --- LAA vs CE -------------------------------------------------------------------------------
    auc_dev = cls["cv"]["full"]["laa_vs_ce_auc"]
    n1, n2 = est["LAA"], est["CE"]
    se = hanley_mcneil_se(auc_dev, int(round(n1)), int(round(n2)))
    se0 = hanley_mcneil_se(0.5, int(round(n1)), int(round(n2)))
    for thr in (0.50, 0.55, 0.60, 0.65):
        out["criteria"][f"laa_vs_ce_auc > {thr}"] = {
            "dev_estimate": auc_dev, "se_test_est": se,
            "power_if_true_equals_dev": 1 - norm_cdf((thr - auc_dev) / se),
            "false_pass_if_true_is_chance": 1 - norm_cdf((thr - 0.5) / se0),
            "expected_ci95_halfwidth": 1.96 * se,
        }

    # --- macro OvR (4 class), for reference -----------------------------------------------------
    macro_dev = cls["cv"]["full"]["macro_auc_ovr"]
    per = []
    for c, n1c in est.items():
        per.append(hanley_mcneil_se(cls["cv"]["full"]["per_class_auc"][c], int(round(n1c)), int(round(n_lab - n1c))))
    se_macro = float(np.sqrt(np.mean(np.array(per) ** 2) / len(per)))  # optimistic: assumes independence
    out["criteria"]["macro_auc_ovr"] = {"dev_estimate": macro_dev, "se_test_est_lower_bound": se_macro,
                                        "per_class_se": dict(zip(est, [float(x) for x in per])),
                                        "comment": "per-class OvR AUCs are correlated; the macro SE is between se_macro and mean(per_class_se)",
                                        "mean_per_class_se": float(np.mean(per))}

    # --- segmentation: Dice mean and ICC ---------------------------------------------------------
    sw = json.load(open(a.sweep))["sweep"]["thr=0.5|minvox=0"]
    dice_sd = (sw["dice_ci"][1] - sw["dice_ci"][0]) / (2 * 1.96)
    icc_lo, icc_hi = sw["icc_ci"]
    out["criteria"]["dice_pos_mean >= 0.55"] = {
        "val_estimate": sw["dice_pos_mean"], "val_bootstrap_se": dice_sd,
        "test_se_est": dice_sd,  # same n (218 val, 218 test)
        "margin_over_threshold_in_se": (sw["dice_pos_mean"] - 0.55) / dice_sd,
        "power_if_true_equals_val": 1 - norm_cdf((0.55 - sw["dice_pos_mean"]) / dice_sd),
    }
    out["criteria"]["icc21 >= 0.85"] = {
        "val_estimate": sw["icc21"], "val_bootstrap_ci95": [icc_lo, icc_hi],
        "val_bootstrap_se_approx": (icc_hi - icc_lo) / (2 * 1.96),
        "seed_sd_A4": 0.05,
        "margin_over_threshold": sw["icc21"] - 0.85,
        "comment": "the bootstrap CI is strongly asymmetric (ICC is bounded above by 1); the lower bound "
                   "is the number that matters and it sits BELOW the 0.85 threshold",
    }
    out["criteria"]["detection_sensitivity >= 0.90"] = {
        "val_estimate": sw["sens"], "n": 218,
        "binomial_se_at_val_estimate": sqrt(sw["sens"] * (1 - sw["sens"]) / 218),
        "max_false_negatives_allowed_at_218": int(np.floor(218 * 0.10)),
        "val_false_negatives": sw["n_fn"],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
