#!/usr/bin/env python3
"""A6 diagnostics on a finished run (val split only — never test):
  1. split integrity (no subject overlap) + the stratification caveat that makes the holdout not fully blind
  2. cache provenance: the preprocessing knobs the cache was built with vs the ones the run asked for
     (`size` is asserted at train time, `clip` is NOT — A4 hand-off item 4)
  3. learning-curve sanity from log.jsonl (train/val not identical, val not high at epoch 0, lr/grad-norm
     as planned, how much val-selection optimism the best-epoch pick buys)
  4. predicted vs GT volume scatter + Bland-Altman, Dice/detection/volume-error by lesion size,
     worst-N failure cases with their data-quality flags -> results/diag_<run>/

Deeper A6 analyses live next to this file: scripts/analysis/a6_curves.py (curves),
a6_threshold.py (threshold sweep, needs the model), a6_power.py (what the test split can decide).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strokeai.metrics import bootstrap_icc_ci, icc21  # noqa: E402
from strokeai.utils import read_jsonl  # noqa: E402

PREPROC_KEYS = ["size", "clip", "clip_mode"]


def load_flags(path: Path) -> dict:
    """A3's acute-vs-chronic label-quality flags, keyed by subject."""
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("acute_eq_chronic") == "True" or r.get("acute_subset_chronic") == "True":
                out[r["participant_id"]] = ("acute==chronic" if r.get("acute_eq_chronic") == "True"
                                            else "acute subset of chronic")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--eval", type=Path, default=None,
                    help="evaluation json to analyse (default: <run>/eval_val.json, else <run>/val_best.json)")
    ap.add_argument("--flags", type=Path, default=Path("results/a3/acute_chronic_flags.csv"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--worst", type=int, default=10)
    a = ap.parse_args()
    out = a.out or Path("results") / f"diag_{a.run.name}"
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    cfg = json.load(open(a.run / "config.json"))

    # --- 1. split integrity ----------------------------------------------------------------------
    sp = json.load(open(a.splits))
    tr, va, te = set(sp["train"]), set(sp["val"]), set(sp["test"])
    report["split"] = {
        "train": len(tr), "val": len(va), "test": len(te),
        "overlap_train_val": len(tr & va), "overlap_train_test": len(tr & te), "overlap_val_test": len(va & te),
        "duplicate_ids": len(sp["train"]) + len(sp["val"]) + len(sp["test"]) - len(tr | va | te),
        "cached_subjects": len(list(a.cache.glob("sub-*.npz"))) if a.cache.exists() else None,
        "test_subjects_in_cache": len([p for p in a.cache.glob("sub-*.npz") if p.stem in te]) if a.cache.exists() else None,
        # A3 F9 + A6: `split.stratum()` keys on etiology_code AND the lesion-size band, so *test metadata*
        # (not images) shaped the split boundaries. The holdout is blind to pixels, not to labels.
        "caveat_stratification_used_test_metadata": ["etiology_code", "mask_acute_ml (<2 mL band)"],
    }

    # --- 2. cache provenance (A4 hand-off 4: a stale `clip` is silent) ---------------------------
    man = a.cache / "_manifest.json"
    if man.exists():
        m = json.load(open(man))
        report["cache_provenance"] = {
            **{k: m.get(k) for k in PREPROC_KEYS + ["n", "which"]},
            "config_values": {k: cfg.get(k) for k in PREPROC_KEYS},
            "matches_run_config": all(m.get(k) == cfg.get(k) for k in PREPROC_KEYS if cfg.get(k) is not None),
            "flag_counts": m.get("flag_counts"),
            "n_errors": len(m.get("errors", [])),
        }

    # --- 3. learning curves ----------------------------------------------------------------------
    log = read_jsonl(a.run / "log.jsonl")
    ep = [r for r in log if r.get("event") == "epoch"]
    st = [r for r in log if r.get("event") == "step"]
    if ep:
        tl = np.array([r["train_loss"] for r in ep]); vd = np.array([r["val_dice_pos"] for r in ep])
        icc = np.array([r["val_vol_icc"] for r in ep], float)
        best = int(vd.argmax())
        report["curves"] = {
            "epochs": len(ep), "train_loss_first_last": [float(tl[0]), float(tl[-1])],
            "val_dice_first_last_best": [float(vd[0]), float(vd[-1]), float(vd.max())], "best_epoch": best,
            "val_dice_epoch0_suspiciously_high": bool(vd[0] > 0.6),
            "train_val_curves_identical": bool(np.any(np.isclose(tl, vd, atol=1e-9))),
            "train_loss_monotone_decrease_frac": float(np.mean(np.diff(tl) < 0)) if len(tl) > 1 else None,
            "clip_frac_mean": float(np.mean([r["clip_frac"] for r in ep])),
            "clip_frac_max": float(np.max([r["clip_frac"] for r in ep])),
            "grad_norm_p95_max": float(max(r["grad_norm_p95"] for r in ep)),
            "lr_first_last": [float(st[0]["lr"]), float(st[-1]["lr"])] if st else None,
            "slices_per_s_median": float(np.median([r["slices_per_s"] for r in st])) if st else None,
            # how much of the reported val score is "best of N evaluations" luck
            "val_selection_optimism": {
                "n_val_evaluations": len(ep),
                "dice_best_minus_last5_mean": float(vd[best] - vd[-5:].mean()),
                "icc_best_minus_last5_mean": float(icc[best] - icc[-5:].mean()),
            },
        }

    # --- 4. per-subject analysis -----------------------------------------------------------------
    src = a.eval or (a.run / "eval_val.json" if (a.run / "eval_val.json").exists() else a.run / "val_best.json")
    if src.exists():
        blob = json.load(open(src))
        report["eval_source"] = {"file": str(src), "threshold": blob.get("threshold", cfg.get("threshold")),
                                 "tta": blob.get("tta", False), "min_voxels": blob.get("min_voxels", 0)}
        per = blob["per_subject"]
        if "summary" in blob:
            report["summary"] = blob["summary"]
        pos = [p for p in per if p["gt_pos"]]
        gt = np.array([p["gt_ml"] for p in pos]); pr = np.array([p["pred_ml"] for p in pos]); dc = np.array([p["dice"] for p in pos])
        _, ilo, ihi = bootstrap_icc_ci(pr, gt, seed=cfg.get("seed", 0))
        report["val_volume"] = {
            "n": len(pos), "pearson_r": float(np.corrcoef(gt, pr)[0, 1]), "mean_bias_ml": float((pr - gt).mean()),
            "icc21": float(icc21(pr, gt)), "icc21_ci95": [ilo, ihi],
            "frac_pred_empty": float(np.mean([not p["pred_pos"] for p in pos])),
            "frac_underestimated": float(np.mean(pr < gt)),
        }
        flags = load_flags(a.flags)
        report["label_quality"] = {
            "flag_file": str(a.flags), "n_flagged_in_this_split": sum(p["sid"] in flags for p in pos),
            "flagged": {p["sid"]: {"flag": flags[p["sid"]], "dice": p["dice"], "gt_ml": p["gt_ml"]} for p in pos if p["sid"] in flags},
        }
        if report["label_quality"]["n_flagged_in_this_split"]:
            keep = np.array([p["sid"] not in flags for p in pos])
            report["label_quality"]["dice_excluding_flagged"] = float(np.nanmean(dc[keep]))
            report["label_quality"]["dice_including_flagged"] = float(np.nanmean(dc))
        worst = sorted(pos, key=lambda p: (p["dice"], -p["gt_ml"]))[: a.worst]
        report["worst_cases"] = [{"sid": p["sid"], "dice": p["dice"], "gt_ml": p["gt_ml"], "pred_ml": p["pred_ml"],
                                  "pred_pos": p["pred_pos"], "label_flag": flags.get(p["sid"])} for p in worst]
        report["missed_subjects"] = [{"sid": p["sid"], "gt_ml": p["gt_ml"]} for p in pos if not p["pred_pos"]]
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            ax[0].scatter(gt, pr, s=8, alpha=0.6); m = max(gt.max(), pr.max()); ax[0].plot([0, m], [0, m], "k--", lw=1)
            ax[0].set_xlabel("GT volume (mL)"); ax[0].set_ylabel("Pred volume (mL)"); ax[0].set_title("Predicted vs GT lesion volume")
            ax[1].scatter(gt, dc, s=8, alpha=0.6); ax[1].set_xscale("log"); ax[1].axvline(2, color="r", ls=":", lw=1)
            ax[1].set_xlabel("GT volume (mL, log)"); ax[1].set_ylabel("Dice"); ax[1].set_title("Dice vs lesion size (red = 2 mL)")
            mean_v, diff = (pr + gt) / 2, pr - gt
            ax[2].scatter(mean_v, diff, s=8, alpha=0.6)
            for y_, s_ in [(diff.mean(), "bias"), (diff.mean() + 1.96 * diff.std(ddof=1), "+1.96 SD"), (diff.mean() - 1.96 * diff.std(ddof=1), "-1.96 SD")]:
                ax[2].axhline(y_, color="r", ls="--" if s_ == "bias" else ":", lw=1)
                ax[2].annotate(f"{s_} {y_:.1f}", (mean_v.max(), y_), fontsize=7, ha="right", va="bottom")
            ax[2].set_xlabel("mean of pred and GT (mL)"); ax[2].set_ylabel("pred - GT (mL)"); ax[2].set_title("Bland-Altman")
            fig.tight_layout(); fig.savefig(out / "val_volume_dice.png", dpi=120)
        except Exception as e:  # noqa: BLE001
            report["plot_error"] = str(e)
    json.dump(report, open(out / "diagnostics.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
