#!/usr/bin/env python3
"""A5b - does the mirror-augmentation axis matter?  (train/val only, never test)

The shipped run flipped the W axis of the (S, C, H, W) cache, which is anterior-posterior, while the
code comment called it a left-right flip (the left-right axis is H).  Before deciding whether the
model has to be re-trained with the anatomically intended flip, train the shipped config with
`flip_axis` in {ap, lr, none} for several seeds and compare val Dice / ICC against the seed noise.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python scripts/analysis/a5b_flip_experiment.py

Writes runs/a5b_flip_<axis>_s<seed>/ and results/a5b/flip_experiment.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from strokeai.train import train  # noqa: E402
from strokeai.utils import load_yaml, read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/seg_unet2d.yaml"))
    ap.add_argument("--axes", nargs="+", default=["ap", "lr", "none"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[2026, 7, 77])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/a5b/flip_experiment.json"))
    a = ap.parse_args()
    base = load_yaml(a.config)
    if a.epochs:
        base["epochs"] = a.epochs
    splits = json.load(open("data/splits.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    results = json.load(open(a.out)) if a.out.exists() else {}
    plan = [(ax, sd) for sd in a.seeds for ax in a.axes if not (ax == "none" and sd != a.seeds[0])]
    for ax, sd in plan:
        name = f"a5b_flip_{ax}_s{sd}"
        run = Path("runs") / name
        if (run / "best.json").exists():
            print("skip (done):", name)
        else:
            cfg = {**base, "flip_axis": ax, "seed": sd}
            t0 = time.time()
            train(cfg, run, Path("data/cache"), splits, device=device)
            print(name, "done in", round(time.time() - t0), "s", flush=True)
        best = json.load(open(run / "best.json"))
        vb = json.load(open(run / "val_best.json"))["summary"]
        ep = [r for r in read_jsonl(run / "log.jsonl") if r.get("event") == "epoch"]
        results[name] = {"flip_axis": ax, "seed": sd, "epochs": len(ep), "best_epoch": best["epoch"],
                         "val_dice_best": best["val_dice"], "val_dice_last5_mean": sum(r["val_dice_pos"] for r in ep[-5:]) / 5,
                         "val_sens": vb["detection_sensitivity"], "val_icc": vb["volume"]["icc21"],
                         "val_icc_ci": vb["volume"].get("icc21_ci95"), "small_dice": vb["dice_by_gt_volume_ml"]["[0,2)"]["dice_mean"],
                         "small_detect": vb["dice_by_gt_volume_ml"]["[0,2)"].get("detect_frac")}
        json.dump(results, open(a.out, "w"), indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
