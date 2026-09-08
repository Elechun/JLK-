#!/usr/bin/env python3
"""A5b - small independent checks behind the A5a-item verdicts (train/val only, never test).

  icc          ICC(2,1) of metrics.py against an explicit sum-of-squares implementation + the val values
  optimism     predicted-mask quality on the seg-TRAIN vs seg-VAL part of the etiology cohort (A5a item 4)
  detection    val subjects with Dice = 0 but a non-empty prediction (A5a item 13)
  flags        union of A3's acute==chronic / acute<chronic flags (A5a item 17)
  centroid_z   Spearman rho(log volume, centroid_z) on GT vs predicted masks (A5a item 10)

    PYTHONPATH=src .venv/bin/python scripts/analysis/a5b_checks.py   -> results/a5b/checks.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from strokeai.features import lesion_features  # noqa: E402
from strokeai.metrics import dice_binary, icc21  # noqa: E402


def icc21_ref(x: np.ndarray, y: np.ndarray) -> float:
    """Shrout & Fleiss ICC(2,1) via explicit sums of squares, written independently of metrics.py."""
    d = np.stack([x, y], 1).astype(float)
    n, k = d.shape
    gm = d.mean()
    ss_rows = k * np.sum((d.mean(1) - gm) ** 2)
    ss_cols = n * np.sum((d.mean(0) - gm) ** 2)
    ss_err = np.sum((d - gm) ** 2) - ss_rows - ss_cols
    msr, msc, mse = ss_rows / (n - 1), ss_cols / (k - 1), ss_err / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def main() -> None:
    out: dict = {}
    run = Path("runs/seg_unet2d")
    per = json.load(open(run / "eval_val.json"))["per_subject"]
    g = np.array([p["gt_ml"] for p in per])
    pr = np.array([p["pred_ml"] for p in per])
    rng = np.random.default_rng(0)
    trials = []
    for _ in range(20):
        x = rng.uniform(0, 100, 50)
        y = x * rng.uniform(0.7, 1.3, 50) + rng.normal(0, 5, 50) + 3
        trials.append(abs(icc21(x, y) - icc21_ref(x, y)))
    out["icc"] = {"max_abs_diff_20_random_trials": float(max(trials)), "val_icc_metrics": float(icc21(pr, g)),
                  "val_icc_reference": icc21_ref(pr, g)}

    # --- optimism of predicted masks on the seg-train part of the etiology cohort -------------------
    sp = json.load(open("data/splits.json"))
    idx = pd.read_csv("data/index.csv")
    idx["etiology_code"] = idx["etiology_code"].fillna("n/a")
    lab = set(idx[idx.etiology_code.isin(["LAA", "CE", "SVO", "OtherDet"])].participant_id)
    rows = []
    for split in ["train", "val"]:
        for sid in sp[split]:
            if sid not in lab:
                continue
            z = np.load(f"data/cache/{sid}.npz")
            gt = z["mask"] > 0
            vv = float(z["voxel_volume_mm3"])
            pm = np.load(run / "pred_masks" / f"{sid}.npz")["mask"] > 0
            fg, fp = lesion_features(gt, vv), lesion_features(pm, vv)
            rows.append(dict(split=split, dice=dice_binary(pm, gt), gt_logv=fg["log_volume_ml"], pr_logv=fp["log_volume_ml"],
                             gt_cz=fg["centroid_z_norm"], pr_cz=fp["centroid_z_norm"], gt_lat=fg["laterality_left_frac"],
                             pr_lat=fp["laterality_left_frac"]))
    df = pd.DataFrame(rows)
    out["optimism"] = {}
    for s, gdf in df.groupby("split"):
        out["optimism"][f"seg_{s}"] = {"n": int(len(gdf)), "dice_mean": float(gdf.dice.mean()), "dice_median": float(gdf.dice.median()),
                                       "rho_logvol_gt_pred": float(spearmanr(gdf.gt_logv, gdf.pr_logv)[0]),
                                       "mae_logvol": float(np.abs(gdf.gt_logv - gdf.pr_logv).mean()),
                                       "mae_laterality": float(np.abs(gdf.gt_lat - gdf.pr_lat).mean())}
    out["centroid_z_rho_logvol"] = {"gt": float(spearmanr(df.gt_logv, df.gt_cz)[0]), "pred": float(spearmanr(df.pr_logv, df.pr_cz)[0])}

    # --- detection semantics ---------------------------------------------------------------------------
    out["detection_val"] = {"n": len(per), "dice_zero": sum(1 for p in per if p["dice"] == 0),
                            "dice_zero_but_pred_nonempty": sum(1 for p in per if p["dice"] == 0 and p["pred_pos"]),
                            "pred_empty": sum(1 for p in per if not p["pred_pos"]),
                            "overlap_hits": sum(1 for p in per if p.get("overlap_pos", p["dice"] > 0))}

    # --- A3 label-quality flags ------------------------------------------------------------------------
    f = pd.read_csv("results/a3/acute_chronic_flags.csv")
    eq = set(f[f.acute_eq_chronic.astype(str) == "True"].participant_id)
    sub = set(f[f.acute_subset_chronic.astype(str) == "True"].participant_id)
    out["flags"] = {"acute_eq_chronic": len(eq), "acute_subset_chronic": len(sub), "union": len(eq | sub),
                    "eq_is_subset_of_subset": eq <= sub, "ids": sorted(eq | sub)}

    Path("results/a5b").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("results/a5b/checks.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
