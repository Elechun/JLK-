#!/usr/bin/env python3
"""A4b (H6, exploratory) - can the < 2 mL band be bought at INFERENCE time?  (val only, never test)

A6 swept the binarisation threshold on the OVERALL val Dice (0.1-0.9 moved it by 0.006) and fixed it at
0.5.  Nobody looked at the threshold *inside the small-lesion band*, where the failure mode measured in
`small_lesion_failure_modes.py` is all-or-nothing detection: a lower threshold could convert misses into
partial hits at the cost of false positives elsewhere.  This costs no training at all, so it is worth a
measurement before concluding that the band cannot be moved.

Sweeps the threshold on the three baseline checkpoints (the shipped config at seeds 2026/7/77) and
reports, per threshold: overall Dice, the [0,2) mL band Dice / detection rate, and volume ICC.
The charter's shipped threshold (0.5) is NOT changed by this script -- it only measures.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python scripts/analysis/small_lesion_threshold.py

Writes results/small_lesion/threshold_by_band.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from strokeai.data.dataset import SubjectCache, context_channel_count  # noqa: E402
from strokeai.metrics import dice_binary, segmentation_summary, volume_ml  # noqa: E402
from strokeai.models import UNet2D  # noqa: E402
from strokeai.train import predict_subject  # noqa: E402

DEFAULT_RUNS = ["runs/a5b_flip_ap_s2026", "runs/a5b_flip_ap_s7", "runs/a5b_flip_ap_s77"]
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, default=[Path(p) for p in DEFAULT_RUNS])
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--thresholds", nargs="+", type=float, default=THRESHOLDS)
    ap.add_argument("--out", type=Path, default=Path("results/small_lesion/threshold_by_band.json"))
    a = ap.parse_args()

    sp = json.load(open(a.splits))
    cache = SubjectCache(a.cache, sp["val"])  # val only -- the held-out split is never loaded
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out: dict = {"runs": [r.as_posix() for r in a.runs], "thresholds": a.thresholds, "per_run": {}}

    for run in a.runs:
        cfg = json.load(open(run / "config.json"))
        channels = cfg.get("channels")
        ctx, mode = int(cfg.get("context", 0) or 0), cfg.get("context_mode", "all")
        in_ch = context_channel_count(2 if channels is None else len(channels), ctx, mode)
        model = UNet2D(in_ch=in_ch, base=cfg["base_channels"], depth=cfg["depth"]).to(device)
        model.load_state_dict(torch.load(run / "best.pt", map_location=device))
        probs = {sid: predict_subject(model, cache.img[sid], device=device, channels=channels,
                                      context=ctx, context_mode=mode) for sid in cache.ids}
        rows = {}
        for thr in a.thresholds:
            per = []
            for sid in cache.ids:
                pred = probs[sid] >= thr
                gt = cache.mask[sid] > 0
                vv = cache.meta[sid]["voxel_volume_mm3"]
                per.append({"sid": sid, "dice": dice_binary(pred, gt), "gt_ml": volume_ml(gt, vv),
                            "pred_ml": volume_ml(pred, vv), "gt_pos": bool(gt.any()),
                            "pred_pos": bool(pred.any()), "overlap_pos": bool((pred & gt).any())})
            s = segmentation_summary(per)
            small = s["dice_by_gt_volume_ml"]["[0,2)"]
            n_zero = sum(1 for r in per if r["gt_pos"] and r["gt_ml"] < 2 and r["dice"] == 0)
            rows[f"{thr:g}"] = {"dice_pos_mean": s["dice_pos_mean"], "small_dice": small["dice_mean"],
                                "small_detect": small["detect_frac"], "small_n_dice_zero": n_zero,
                                "small_median_abs_pct_err": small["median_abs_pct_err"],
                                "sens": s["detection_sensitivity"], "sens_overlap": s.get("detection_sensitivity_overlap"),
                                "icc21": s["volume"]["icc21"], "median_abs_pct_err": s["volume"]["median_abs_pct_err"]}
        out["per_run"][run.name] = rows
        print(run.name, "done", flush=True)

    # mean over the three seeds
    keys = list(out["per_run"][a.runs[0].name][f"{a.thresholds[0]:g}"])
    out["mean_over_runs"] = {f"{thr:g}": {k: float(np.mean([out["per_run"][r.name][f"{thr:g}"][k] for r in a.runs]))
                                          for k in keys} for thr in a.thresholds}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)

    print(f"\n{'thr':>5} {'small_dice':>11} {'small_det':>10} {'n_zero':>7} {'dice':>7} {'ICC':>7} {'sens':>6}")
    for thr in a.thresholds:
        m = out["mean_over_runs"][f"{thr:g}"]
        print(f"{thr:5.2f} {m['small_dice']:11.4f} {m['small_detect']:10.3f} {m['small_n_dice_zero']:7.2f} "
              f"{m['dice_pos_mean']:7.4f} {m['icc21']:7.4f} {m['sens']:6.3f}")


if __name__ == "__main__":
    main()
