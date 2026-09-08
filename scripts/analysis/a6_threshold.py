#!/usr/bin/env python3
"""A6 - probability-threshold selection on the VAL split (never test).

Computes the val probability maps once, then sweeps the binarisation threshold (and an optional
minimum-lesion-size rule) recomputing every charter metric, with subject-bootstrap CIs for the ones
whose charter threshold is close (Dice, ICC).  Also reports the same sweep restricted to the
small-lesion (<2 mL) stratum, and a sensitivity analysis excluding the A3 label-quality flags
(acute == chronic / acute subset of chronic).

Usage:
  PYTHONPATH=src .venv/bin/python scripts/analysis/a6_threshold.py --run runs/seg_unet2d
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from strokeai.data.dataset import SubjectCache  # noqa: E402
from strokeai.metrics import dice_binary, icc21, segmentation_summary, volume_ml  # noqa: E402
from strokeai.models import UNet2D  # noqa: E402
from strokeai.train import predict_subject  # noqa: E402


def load_probs(run: Path, cache_dir: Path, ids: list[str], ckpt: str, cache_npz: Path | None):
    if cache_npz and cache_npz.exists():
        z = np.load(cache_npz, allow_pickle=True)
        return {k: z[k] for k in z.files if k != "_meta"}, json.loads(str(z["_meta"]))
    cfg = json.load(open(run / "config.json"))
    channels = cfg.get("channels")
    model = UNet2D(in_ch=2 if channels is None else len(channels), base=cfg["base_channels"], depth=cfg["depth"])
    model.load_state_dict(torch.load(run / ckpt, map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    cache = SubjectCache(cache_dir, ids)
    # fp32, not float16: a float16 round trip rounds e.g. 0.49999 up to 0.5 and flips the 0.5 decision (A5a item 15)
    probs = {sid: predict_subject(model, cache.img[sid], device=device, channels=channels).astype(np.float32) for sid in ids}
    meta = {sid: {"gt_ml": volume_ml(cache.mask[sid] > 0, cache.meta[sid]["voxel_volume_mm3"]),
                  "vv": float(cache.meta[sid]["voxel_volume_mm3"])} for sid in ids}
    gts = {sid: (cache.mask[sid] > 0) for sid in ids}
    if cache_npz:
        cache_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_npz, _meta=json.dumps(meta), **probs)
    return probs, meta, gts


def per_subject_at(probs, gts, meta, thr: float, min_voxels: int = 0) -> list[dict]:
    out = []
    for sid, p in probs.items():
        pred = np.asarray(p, np.float32) >= thr
        if min_voxels and pred.sum() < min_voxels:
            pred = np.zeros_like(pred)
        gt = gts[sid]
        vv = meta[sid]["vv"]
        out.append({"sid": sid, "dice": dice_binary(pred, gt), "gt_ml": volume_ml(gt, vv), "pred_ml": volume_ml(pred, vv),
                    "gt_pos": bool(gt.any()), "pred_pos": bool(pred.any())})
    return out


def boot_icc(pred, gt, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(pred)
    b = np.array([icc21(pred[i], gt[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    return float(icc21(pred, gt)), float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))


def row(per: list[dict], seed: int = 2026) -> dict:
    s = segmentation_summary(per, seed=seed)
    gt = np.array([p["gt_ml"] for p in per if p["gt_pos"]], float)
    pr = np.array([p["pred_ml"] for p in per if p["gt_pos"]], float)
    icc, lo, hi = boot_icc(pr, gt, seed=seed)
    small = [p for p in per if p["gt_pos"] and p["gt_ml"] < 2]
    return {
        "dice_pos_mean": s["dice_pos_mean"], "dice_ci": s["dice_pos_ci95"], "dice_median": s["dice_pos_median"],
        "sens": s["detection_sensitivity"], "n_fn": s["detection_confusion"]["fn"],
        "icc21": icc, "icc_ci": [lo, hi],
        "mae_ml": s["volume"]["mae_ml"], "bias_ml": s["volume"]["mean_diff_ml"],
        "median_abs_pct_err": s["volume"]["median_abs_pct_err"], "mape_pct": s["volume"]["mape_pct"],
        "loa": [s["volume"]["loa_low_ml"], s["volume"]["loa_high_ml"]],
        "dice_by_size": {k: v["dice_mean"] for k, v in s["dice_by_gt_volume_ml"].items()},
        "small_lt2ml": {"n": len(small), "dice_mean": float(np.nanmean([p["dice"] for p in small])),
                        "detect_frac": float(np.mean([p["pred_pos"] for p in small])),
                        "median_abs_pct_err": float(np.median([abs(p["pred_ml"] - p["gt_ml"]) / p["gt_ml"] for p in small]) * 100)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs/seg_unet2d"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--split", choices=["train", "val"], default="val", help="test is forbidden here by design")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--min-voxels", type=int, nargs="+", default=[0])
    ap.add_argument("--flags", type=Path, default=Path("results/a3/acute_chronic_flags.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/a6/threshold_sweep.json"))
    a = ap.parse_args()
    ids = json.load(open(a.splits))[a.split]
    probs, meta, gts = load_probs(a.run, a.cache, ids, a.ckpt, None)

    res = {"run": str(a.run), "ckpt": a.ckpt, "split": a.split, "n": len(ids), "sweep": {}}
    for mv in a.min_voxels:
        for thr in a.thresholds:
            per = per_subject_at(probs, gts, meta, thr, mv)
            res["sweep"][f"thr={thr}|minvox={mv}"] = row(per)

    # label-quality sensitivity analysis at the shipped threshold (A3 F14)
    if a.flags.exists():
        import csv
        bad = set()
        with open(a.flags) as f:
            for r in csv.DictReader(f):
                if str(r.get("acute_eq_chronic", "")).lower() in ("true", "1") or \
                   str(r.get("acute_subset_chronic", "")).lower() in ("true", "1"):
                    bad.add(r["participant_id"])
        cfg = json.load(open(a.run / "config.json"))
        per = per_subject_at(probs, gts, meta, cfg["threshold"], 0)
        inval = sorted(bad & set(ids))
        res["label_quality_sensitivity"] = {
            "flagged_ids_in_split": inval,
            "all": row(per),
            "excluding_flagged": row([p for p in per if p["sid"] not in bad]) if inval else None,
            "flagged_subjects": [p for p in per if p["sid"] in bad],
        }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: {kk: v[kk] for kk in ["dice_pos_mean", "sens", "icc21", "mae_ml", "median_abs_pct_err", "small_lt2ml"]}
                      for k, v in res["sweep"].items()}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
