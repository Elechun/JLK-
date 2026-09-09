#!/usr/bin/env python3
"""A4b - WHAT actually goes wrong in the < 2 mL band, and why the band's Dice is so noisy (val only).

The band mean (0.538 for the shipped run) is the headline number the charter reports, but it is a mean
over 60 subjects whose Dice distribution is bimodal: a cluster of complete misses at 0 and a body
around 0.65.  This script measures
  (a) whether the failure is UNDER-segmentation (which would justify an FN-weighted loss, H5) or
      all-or-nothing localisation,
  (b) how much of the band mean the zeros cost,
  (c) how many subjects flip between Dice = 0 and Dice > 0 across seeds -- i.e. where the band's
      seed noise comes from, and therefore how big an effect has to be to be detectable at n = 60.

    PYTHONPATH=src .venv/bin/python scripts/analysis/small_lesion_failure_modes.py

Writes results/small_lesion/failure_modes.json.  Reads only runs/*/val_best.json (val split).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_RUNS = ["runs/a5b_flip_ap_s2026", "runs/a5b_flip_ap_s7", "runs/a5b_flip_ap_s77"]


def band(run: Path, hi: float = 2.0) -> list[dict]:
    per = json.load(open(run / "val_best.json"))["per_subject"]
    return [r for r in per if r["gt_pos"] and r["gt_ml"] < hi]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, default=[Path(p) for p in DEFAULT_RUNS])
    ap.add_argument("--hi", type=float, default=2.0, help="upper bound of the band in mL")
    ap.add_argument("--out", type=Path, default=Path("results/small_lesion/failure_modes.json"))
    a = ap.parse_args()

    out: dict = {"runs": [r.as_posix() for r in a.runs], "band_ml": [0.0, a.hi], "per_run": {}}
    dice_by_sid: dict[str, list[float]] = {}
    for run in a.runs:
        rows = band(run, a.hi)
        gt = np.array([r["gt_ml"] for r in rows])
        pred = np.array([r["pred_ml"] for r in rows])
        dice = np.array([r["dice"] for r in rows])
        zero = dice == 0
        for r in rows:
            dice_by_sid.setdefault(r["sid"], []).append(r["dice"])
        out["per_run"][run.name] = {
            "n": len(rows), "dice_mean": float(dice.mean()), "dice_median": float(np.median(dice)),
            "n_dice_zero": int(zero.sum()), "n_empty_prediction": int(sum(not r["pred_pos"] for r in rows)),
            "dice_mean_excluding_zeros": float(dice[~zero].mean()),
            "band_mean_cost_of_zeros": float(dice[~zero].mean() - dice.mean()),  # how much the misses cost the mean
            # under- vs over-segmentation among the subjects the model DID find
            "median_pred_over_gt_all": float(np.median(pred / gt)),
            "median_pred_over_gt_found": float(np.median(pred[~zero] / gt[~zero])),
            "frac_under_segmented_found": float(np.mean(pred[~zero] < gt[~zero])),
            "gt_ml_median": float(np.median(gt)),
        }

    # ---- seed stability, subject by subject ---------------------------------------------------------
    sids = sorted(dice_by_sid)
    mat = np.array([dice_by_sid[s] for s in sids])  # (n_subjects, n_runs)
    zero_count = (mat == 0).sum(axis=1)
    n_runs = mat.shape[1]
    out["seed_stability"] = {
        "n_subjects": len(sids), "n_runs": n_runs,
        "always_zero": int((zero_count == n_runs).sum()),
        "never_zero": int((zero_count == 0).sum()),
        "flips_between_zero_and_nonzero": int(((zero_count > 0) & (zero_count < n_runs)).sum()),
        "per_subject_dice_sd_mean": float(mat.std(axis=1, ddof=1).mean()),
        "band_mean_by_run": [float(x) for x in mat.mean(axis=0)],
        "band_mean_sd_across_runs": float(mat.mean(axis=0).std(ddof=1)),
        # how much of the band-mean spread is produced by the all-or-nothing subjects alone?
        "band_mean_sd_excluding_flippers": float(
            mat[zero_count == 0].mean(axis=0).std(ddof=1)) if (zero_count == 0).any() else float("nan"),
        "flipper_sids": [s for s, z in zip(sids, zero_count) if 0 < z < n_runs],
    }
    # one Dice=0 subject moves the 60-subject band mean by its would-be Dice / 60
    out["seed_stability"]["band_mean_shift_per_flipped_subject"] = float(
        np.median(mat[mat > 0]) / len(sids))

    # ---- is the hard floor an input problem?  cross-check against A6's lesion-contrast table --------
    ct = Path("results/a6/failures_val.csv")
    if ct.exists():
        import csv

        tab = {r["sid"]: r for r in csv.DictReader(open(ct))}
        always = [s for s, z in zip(sids, zero_count) if z == n_runs]
        rest = [s for s in sids if s not in always and s in tab]

        def stat(group, col):
            v = [float(tab[s][col]) for s in group if s in tab]
            return float(np.median(v)) if v else float("nan")

        out["hard_floor_vs_input_evidence"] = {
            "source": ct.as_posix(),
            "n_always_zero": len(always), "sids": always,
            "trace_contrast_median_always_zero": stat(always, "trace_contrast"),
            "trace_contrast_median_rest_of_band": stat(rest, "trace_contrast"),
            # A6: a POSITIVE ADC contrast (lesion brighter than the ring) contradicts an acute infarct
            "n_adc_contrast_positive_always_zero": sum(float(tab[s]["adc_contrast"]) > 0 for s in always if s in tab),
            "n_adc_contrast_positive_in_band": sum(float(tab[s]["adc_contrast"]) > 0 for s in sids if s in tab),
        }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
