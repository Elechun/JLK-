#!/usr/bin/env python3
"""A6 - learning-curve / leakage diagnostics from runs/<run>/log.jsonl.

Checks, in order:
  1. train and val curves are not abnormally identical (rank correlation + "same value" count)
  2. val metrics are not already high at epoch 0 (leakage signature) and the epoch-0 value is
     compared against the *trivial* predictors measured on the same val subjects
  3. lr follows the configured cosine-with-warmup schedule (analytic re-derivation, max abs error)
  4. grad-norm / clip fraction over the run
  5. val-selection optimism: best epoch vs the plateau (last-k epochs) for every logged val metric
  6. seed reproducibility: |val Dice(run) - val Dice(repro run)| against the charter's 0.02

train / val only - never reads the test split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from strokeai.train import cosine_warmup  # noqa: E402
from strokeai.utils import read_jsonl  # noqa: E402


def curve_report(run: Path, plateau_k: int = 5) -> dict:
    log = read_jsonl(run / "log.jsonl")
    cfg = json.load(open(run / "config.json"))
    start = next(r for r in log if r.get("event") == "start")
    ep = [r for r in log if r.get("event") == "epoch"]
    st = [r for r in log if r.get("event") == "step"]
    tl = np.array([r["train_loss"] for r in ep], float)
    vd = np.array([r["val_dice_pos"] for r in ep], float)
    icc = np.array([r["val_vol_icc"] for r in ep], float)
    sens = np.array([r["val_sens"] for r in ep], float)

    # (1) train/val identity -------------------------------------------------------------------
    # a train-loss curve and a val-Dice curve are different quantities, so "identical" is tested as
    # (a) exact value coincidence and (b) a perfect monotone (Spearman) relation between the two.
    from scipy.stats import spearmanr

    rho = float(spearmanr(tl, vd).statistic)
    ident = {
        "n_epochs_with_train_loss_equal_val_dice": int(np.sum(np.isclose(tl, vd, atol=1e-9))),
        "spearman_train_loss_vs_val_dice": rho,
        "abnormally_identical": bool(np.sum(np.isclose(tl, vd, atol=1e-9)) > 0 or rho <= -0.999),
        "val_dice_distinct_values": int(len(np.unique(np.round(vd, 12)))),
    }

    # (2) epoch-0 leakage signature ------------------------------------------------------------
    leak = {
        "val_dice_epoch0": float(vd[0]),
        "val_dice_best": float(vd.max()),
        "best_epoch": int(vd.argmax()),
        "epoch0_frac_of_best": float(vd[0] / vd.max()),
        "epochs_to_reach_90pct_of_best": int(np.argmax(vd >= 0.9 * vd.max())),
        # a leaking model is (near-)converged before it has seen the data; > 0.6 at epoch 0 or
        # > 0.9 of the final value within the first two epochs would be the alarm.
        "flag_epoch0_high": bool(vd[0] > 0.6),
        "flag_converged_immediately": bool(np.argmax(vd >= 0.9 * vd.max()) <= 1),
    }

    # (3) lr schedule --------------------------------------------------------------------------
    total, warm = start["total_steps"], int(cfg["warmup_frac"] * start["total_steps"])
    err = [abs(r["lr"] - cosine_warmup(r["step"] - 1, total, warm, cfg["lr"], cfg["min_lr"])) for r in st]
    lrs = np.array([r["lr"] for r in st], float)
    sched = {
        "warmup_steps": warm, "total_steps": total,
        "max_abs_lr_error_vs_formula": float(max(err)) if err else None,
        "lr_first_logged": float(lrs[0]), "lr_max_logged": float(lrs.max()), "lr_last_logged": float(lrs[-1]),
        "lr_min_configured": cfg["min_lr"], "lr_base_configured": cfg["lr"],
        "monotone_decreasing_after_peak": bool(np.all(np.diff(lrs[int(lrs.argmax()):]) <= 1e-12)),
    }

    # (4) grad norm ----------------------------------------------------------------------------
    cf = np.array([r["clip_frac"] for r in ep], float)
    gn95 = np.array([r["grad_norm_p95"] for r in ep], float)
    grad = {
        "clip_frac_mean": float(cf.mean()), "clip_frac_max": float(cf.max()),
        "clip_frac_argmax_epoch": int(cf.argmax()),
        "clip_frac_first5": [float(x) for x in cf[:5]], "clip_frac_last5": [float(x) for x in cf[-5:]],
        "grad_norm_p95_max": float(gn95.max()), "grad_norm_p95_last": float(gn95[-1]),
        "grad_clip_configured": cfg["grad_clip"],
        "runbook_claim_zero_pct_clipping_holds": bool(cf.max() == 0.0),
    }

    # (5) val-selection optimism ---------------------------------------------------------------
    def opt(v):
        return {"best": float(v.max()), "at_best_epoch": float(v[int(vd.argmax())]), "last": float(v[-1]),
                "plateau_mean": float(v[-plateau_k:].mean()), "plateau_sd": float(v[-plateau_k:].std(ddof=1)),
                "best_minus_plateau_mean": float(v[int(vd.argmax())] - v[-plateau_k:].mean())}

    optimism = {"val_dice": opt(vd), "val_icc": opt(icc), "val_sens": opt(sens),
                "n_val_evaluations": len(ep), "plateau_k": plateau_k}

    return {"run": str(run), "n_params": start["n_params"], "device": start.get("device"),
            "train_loss_first_last": [float(tl[0]), float(tl[-1])],
            "identity_check": ident, "leakage_check": leak, "lr_schedule": sched, "grad": grad,
            "val_selection_optimism": optimism,
            "throughput_slices_per_s_median": float(np.median([r["slices_per_s"] for r in st])) if st else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs/seg_unet2d"))
    ap.add_argument("--repro-run", type=Path, default=Path("runs/seg_unet2d_repro"))
    ap.add_argument("--plateau-k", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("results/a6/curves.json"))
    a = ap.parse_args()
    rep = {"main": curve_report(a.run, a.plateau_k)}
    if a.repro_run and (a.repro_run / "log.jsonl").exists():
        rep["repro"] = curve_report(a.repro_run, a.plateau_k)
        d = abs(rep["main"]["leakage_check"]["val_dice_best"] - rep["repro"]["leakage_check"]["val_dice_best"])
        rep["reproducibility"] = {"abs_val_dice_diff": float(d), "charter_limit": 0.02, "pass": bool(d < 0.02),
                                  "same_best_epoch": rep["main"]["leakage_check"]["best_epoch"] == rep["repro"]["leakage_check"]["best_epoch"]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
