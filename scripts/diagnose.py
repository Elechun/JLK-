#!/usr/bin/env python3
"""A6 diagnostics on a finished run (val split only — never test):
  1. split integrity (no subject overlap; every cached subject belongs to exactly one split)
  2. learning-curve sanity from log.jsonl (train/val not identical, val not high at epoch 0, lr/grad-norm as planned)
  3. predicted vs GT volume scatter, Dice by lesion size, worst-N failure cases -> results/diag_<run>/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strokeai.utils import read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--worst", type=int, default=10)
    a = ap.parse_args()
    out = a.out or Path("results") / f"diag_{a.run.name}"
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    sp = json.load(open(a.splits))
    tr, va, te = set(sp["train"]), set(sp["val"]), set(sp["test"])
    report["split"] = {"train": len(tr), "val": len(va), "test": len(te),
                       "overlap_train_val": len(tr & va), "overlap_train_test": len(tr & te), "overlap_val_test": len(va & te)}

    log = read_jsonl(a.run / "log.jsonl")
    ep = [r for r in log if r.get("event") == "epoch"]
    st = [r for r in log if r.get("event") == "step"]
    if ep:
        tl = np.array([r["train_loss"] for r in ep]); vd = np.array([r["val_dice_pos"] for r in ep])
        report["curves"] = {
            "epochs": len(ep), "train_loss_first_last": [float(tl[0]), float(tl[-1])],
            "val_dice_first_last_best": [float(vd[0]), float(vd[-1]), float(vd.max())], "best_epoch": int(vd.argmax()),
            "val_dice_epoch0_suspiciously_high": bool(vd[0] > 0.6),
            "train_loss_monotone_decrease_frac": float(np.mean(np.diff(tl) < 0)) if len(tl) > 1 else None,
            "clip_frac_mean": float(np.mean([r["clip_frac"] for r in ep])),
            "grad_norm_p95_max": float(max(r["grad_norm_p95"] for r in ep)),
            "lr_first_last": [float(st[0]["lr"]), float(st[-1]["lr"])] if st else None,
            "slices_per_s_median": float(np.median([r["slices_per_s"] for r in st])) if st else None,
        }
    vb = a.run / "val_best.json"
    if vb.exists():
        per = json.load(open(vb))["per_subject"]
        pos = [p for p in per if p["gt_pos"]]
        gt = np.array([p["gt_ml"] for p in pos]); pr = np.array([p["pred_ml"] for p in pos]); dc = np.array([p["dice"] for p in pos])
        report["val_volume"] = {"n": len(pos), "pearson_r": float(np.corrcoef(gt, pr)[0, 1]), "mean_bias_ml": float((pr - gt).mean()),
                                "frac_pred_empty": float(np.mean([not p["pred_pos"] for p in pos]))}
        worst = sorted(pos, key=lambda p: p["dice"])[: a.worst]
        report["worst_cases"] = [{k: p[k] for k in ["sid", "dice", "gt_ml", "pred_ml"]} for p in worst]
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].scatter(gt, pr, s=8, alpha=0.6); m = max(gt.max(), pr.max()); ax[0].plot([0, m], [0, m], "k--", lw=1)
            ax[0].set_xlabel("GT volume (mL)"); ax[0].set_ylabel("Pred volume (mL)"); ax[0].set_title("Val: predicted vs GT lesion volume")
            ax[1].scatter(gt, dc, s=8, alpha=0.6); ax[1].set_xscale("log"); ax[1].set_xlabel("GT volume (mL, log)"); ax[1].set_ylabel("Dice")
            ax[1].set_title("Val: Dice vs lesion size")
            fig.tight_layout(); fig.savefig(out / "val_volume_dice.png", dpi=120)
        except Exception as e:  # noqa: BLE001
            report["plot_error"] = str(e)
    json.dump(report, open(out / "diagnostics.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
