#!/usr/bin/env python3
"""Vendor / acquisition-condition robustness of the segmentation model (val only).

Motivation: the headline val Dice hides the failure mode. Grouping the stored val predictions by
scanner manufacturer shows the model returns an EMPTY mask for every Siemens subject, including a
107 mL lesion, while averaging 0.69 overall.

Two hypotheses are tested against each other:
  H-ADC   the ADC channel is inverted on Siemens (lesion ADC z is +4.7..+8.7 vs ~0 elsewhere)
          -> refuted: the TRACE-only checkpoint fails on the same subjects.
  H-NORM  the robust z-score saturates on Siemens intensity distributions (8.6-11.7 % of TRACE
          voxels at the clip vs 0.0 % for Philips/GE), pushing the input off-distribution
          -> supported: percentile normalisation recovers the large lesion (Dice 0 -> 0.84).

Usage:
    PYTHONPATH=src python scripts/analysis/vendor_robustness.py [--out results/vendor]

Never touches the test split: the model was already scored on test once (charter section C).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from strokeai.metrics import dice_binary  # noqa: E402
from strokeai.models import UNet2D  # noqa: E402
from strokeai.train import predict_subject  # noqa: E402

ARMS = {
    "mad_clip10": ("runs/a4_stage1_clip6/a4_clip10", "data/cache_clip10"),
    "percentile": ("runs/a4_stage1_clip6/a4_clippct01", "data/cache_pct01"),
}


def score(run: str, cache: str, ids: list[str], device: str) -> pd.DataFrame:
    cfg = json.load(open(f"{run}/config.json"))
    thr = cfg.get("threshold", 0.5)
    model = UNet2D(in_ch=len(cfg.get("channels", [0, 1])), base=cfg["base_channels"], depth=cfg["depth"])
    model.load_state_dict(torch.load(f"{run}/best.pt", map_location="cpu"))
    model.to(device).eval()
    rows = []
    for sid in ids:
        p = f"{cache}/{sid}.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p)
        img = z["img"].astype(np.float16)
        prob = predict_subject(model, img, device=device)
        pred, gt = prob >= thr, z["mask"] > 0
        vv = float(z["voxel_volume_mm3"])
        tr = img[:, 0].astype(np.float32)
        rows.append({"sid": sid, "dice": dice_binary(pred, gt), "detected": bool(pred.any()),
                     "gt_ml": gt.sum() * vv / 1000.0, "pred_ml": pred.sum() * vv / 1000.0,
                     "trace_saturated_frac": float(np.mean(np.abs(tr) >= np.abs(tr).max() - 1e-3))})
    return pd.DataFrame(rows).set_index("sid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/vendor"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    idx = pd.read_csv("data/index.csv").set_index("participant_id")
    val = json.load(open("data/splits.json"))["val"]  # val only - test is spent

    report: dict = {"split": "val", "n": len(val), "arms": {}}
    frames = {}
    for name, (run, cache) in ARMS.items():
        df = score(run, cache, val, a.device).join(idx[["trace_Manufacturer", "trace_MagneticFieldStrength"]])
        frames[name] = df
        groups = {
            "all": df,
            "Philips": df[df.trace_Manufacturer == "Philips"],
            "Siemens": df[df.trace_Manufacturer == "Siemens"],
            "GE": df[df.trace_Manufacturer == "GE"],
            "large_ge50ml": df[df.gt_ml >= 50],
            "small_lt2ml": df[df.gt_ml < 2],
        }
        report["arms"][name] = {
            g: {"n": int(len(x)), "dice_mean": float(np.nanmean(x.dice)) if len(x) else None,
                "detection": float(x.detected.mean()) if len(x) else None}
            for g, x in groups.items()
        }
        df.to_csv(a.out / f"val_{name}.csv")

    sie = frames["mad_clip10"][frames["mad_clip10"].trace_Manufacturer == "Siemens"]
    report["siemens_subjects"] = {
        sid: {"gt_ml": float(r.gt_ml),
              "dice_mad": float(r.dice),
              "dice_percentile": float(frames["percentile"].loc[sid, "dice"]),
              "detected_mad": bool(r.detected),
              "detected_percentile": bool(frames["percentile"].loc[sid, "detected"])}
        for sid, r in sie.iterrows()
    }
    json.dump(report, open(a.out / "vendor_robustness.json", "w"), indent=1)
    print(json.dumps(report["arms"], indent=1))
    print("\nSiemens per subject:", json.dumps(report["siemens_subjects"], indent=1))
    print("wrote", a.out / "vendor_robustness.json")


if __name__ == "__main__":
    main()
