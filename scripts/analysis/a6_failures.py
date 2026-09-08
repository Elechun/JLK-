#!/usr/bin/env python3
"""A6 - failure-mode analysis on the VAL split (never test).

For every val subject it measures, from the preprocessed cache, how much *evidence* the two input
channels actually carry for the annotated lesion.  The reference region is a **peri-lesional ring**
(4-voxel dilation of the GT minus the GT itself, restricted to the slices the lesion touches), not
"everything outside the lesion": the volume is mostly air, and after the per-subject z-score air sits
far below the brain, which would make every contrast look positive.

    trace_contrast = mean(TRACE_z in GT) - mean(TRACE_z in ring)
    adc_contrast   = mean(ADC_z   in GT) - mean(ADC_z   in ring)

An acute infarct is bright on TRACE and dark on ADC, so `trace_contrast > 0 and adc_contrast < 0` is
the expected signature; `adc_contrast > 0` is the signature of a *chronic* lesion (ADC pseudo-normalised
or elevated) and points at label quality rather than model capacity.
The script joins these against the per-subject Dice and reports the worst cases with their evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs/seg_unet2d"))
    ap.add_argument("--eval", type=Path, default=None)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--split", choices=["train", "val", "test"], default="val",
                    help="test is allowed ONLY in the final, signed-off evaluation step")
    ap.add_argument("--out", type=Path, default=Path("results/a6"))
    a = ap.parse_args()
    ev = a.eval or (a.run / f"eval_{a.split}.json")
    per = {p["sid"]: p for p in json.load(open(ev))["per_subject"]}
    ids = json.load(open(a.splits))[a.split]
    assert set(ids) == set(per), "eval file does not match the requested split"

    rows = []
    for sid in ids:
        z = np.load(a.cache / f"{sid}.npz")
        img = z["img"].astype(np.float32)
        m = z["mask"] > 0
        p = per[sid]
        r = {"sid": sid, "dice": p["dice"], "gt_ml": p["gt_ml"], "pred_ml": p["pred_ml"], "pred_pos": p["pred_pos"]}
        if m.any():
            ring = ndimage.binary_dilation(m, structure=np.ones((1, 3, 3)), iterations=4) & ~m
            ring &= m.any(axis=(1, 2))[:, None, None]  # only slices the lesion touches
            ring &= img[:, 0] > -1.0  # stay inside the brain: air sits far below 0 after the z-score
            if not ring.any():
                ring = ~m & (img[:, 0] > -1.0)
            r["trace_contrast"] = float(img[:, 0][m].mean() - img[:, 0][ring].mean())
            r["adc_contrast"] = float(img[:, 1][m].mean() - img[:, 1][ring].mean())
            r["trace_in_lesion"] = float(img[:, 0][m].mean())
            r["adc_in_lesion"] = float(img[:, 1][m].mean())
            r["n_ring_voxels"] = int(ring.sum())
            r["frac_trace_at_clip"] = float(np.mean(np.abs(img[:, 0][m]) >= 0.999 * np.abs(img[:, 0]).max()))
        rows.append(r)

    d = np.array([r["dice"] for r in rows], float)
    tc = np.array([r.get("trace_contrast", np.nan) for r in rows], float)
    ac = np.array([r.get("adc_contrast", np.nan) for r in rows], float)
    gv = np.array([r["gt_ml"] for r in rows], float)
    from scipy.stats import spearmanr

    ok = ~np.isnan(tc)
    chronic_like = ok & (ac > 0)
    summ = {
        "split": a.split, "n": int(ok.sum()),
        "spearman_dice_vs_trace_contrast": float(spearmanr(tc[ok], d[ok]).statistic),
        "spearman_dice_vs_adc_contrast": float(spearmanr(ac[ok], d[ok]).statistic),
        "spearman_dice_vs_log_gt_ml": float(spearmanr(np.log10(gv[ok]), d[ok]).statistic),
        "n_adc_contrast_positive": int(chronic_like.sum()),
        "frac_adc_contrast_positive": float(chronic_like.mean()),
        "dice_mean_adc_negative": float(np.nanmean(d[ok & ~chronic_like])),
        "dice_mean_adc_positive": float(np.nanmean(d[chronic_like])),
        "median_gt_ml_adc_positive": float(np.median(gv[chronic_like])) if chronic_like.any() else None,
        "n_missed": int(sum(1 for r in rows if not r["pred_pos"])),
        "missed_with_adc_positive": [r["sid"] for r in rows if not r["pred_pos"] and r.get("adc_contrast", -1) > 0],
    }
    worst = sorted([r for r in rows if not np.isnan(r["dice"])], key=lambda r: r["dice"])[:15]
    summ["worst15"] = worst
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(summ, open(a.out / f"failures_{a.split}.json", "w"), indent=1)
    with open(a.out / f"failures_{a.split}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: v for k, v in summ.items() if k != "worst15"}, indent=1))
    print("\nworst 15 (dice, gt_ml, pred_ml, trace_contrast, adc_contrast):")
    for r in worst:
        print(f"  {r['sid']:10s} {r['dice']:.3f} {r['gt_ml']:8.2f} {r['pred_ml']:8.2f} "
              f"{r.get('trace_contrast', float('nan')):+6.2f} {r.get('adc_contrast', float('nan')):+6.2f}")
    print("wrote", a.out / f"failures_{a.split}.json")


if __name__ == "__main__":
    main()
