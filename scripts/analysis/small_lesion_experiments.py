#!/usr/bin/env python3
"""A4b - can the < 2 mL band be improved?  (train/val only, never test)

Every variant is the SHIPPED config (configs/seg_unet2d.yaml: 40 epochs, size 128, clip 10, flip_axis ap)
with one group of keys overridden, trained for the same seeds, so the only thing that moves is the
intervention.  The baseline is not re-trained: runs/a5b_flip_ap_s{2026,7,77} already are that config at
those seeds (config.json is byte-identical to the yaml) -- `--variants base_extra` only adds two more
seeds so the seed SD of the small-lesion band is estimated from five runs instead of three.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python scripts/analysis/small_lesion_experiments.py \
        --variants h2_subj h2_vol05 --seeds 2026 7 77

Writes runs/small_<variant>_s<seed>/ and results/small_lesion/experiments.json (one entry per run;
existing entries are kept, so the script is resumable and can be sharded across processes).
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

# --- the hypotheses ------------------------------------------------------------------------------
# H2  lesion-size-aware slice sampling.  Measured representation under the shipped sampler
#     (scripts/analysis/small_lesion_stats.py): < 2 mL subjects are 28.0 % of the train split but own
#     only 10.2 % of the positive slices; >= 50 mL are 18.4 % of subjects and 33.3 % of slices.
# H4  2.5-D context: neighbouring slices as extra input channels.
# H5  small-lesion-weighted / FN-weighted region loss.
VARIANTS: dict[str, dict] = {
    # H2 --------------------------------------------------------------------------------------
    "h2_subj":     {"slice_weight": "inv_subject", "slice_weight_power": 1.0},   # small band 10.2 -> 28.0 %
    "h2_vol05":    {"slice_weight": "inv_volume", "slice_weight_power": 0.5},    # small band -> 36.2 %
    "h2_vol10":    {"slice_weight": "inv_volume", "slice_weight_power": 1.0},    # small band -> 69.2 %, ESS 10 %
    # H4 --------------------------------------------------------------------------------------
    "h4_trace":    {"context": 1, "context_mode": "trace"},                      # 4 ch: TRACE t-1,t,t+1 + ADC t
    "h4_all":      {"context": 1, "context_mode": "all"},                        # 6 ch: TRACE and ADC t-1,t,t+1
    # H5 --------------------------------------------------------------------------------------
    "h5_tversky":  {"region_loss": "tversky", "tversky_alpha": 0.7, "tversky_beta": 0.3, "tversky_gamma": 1.0},
    "h5_ftversky": {"region_loss": "tversky", "tversky_alpha": 0.7, "tversky_beta": 0.3, "tversky_gamma": 1.33},
    "h5_invsize":  {"dice_reduction": "sample_invsize", "invsize_power": 0.5},
    # extra baseline seeds (no override at all) -- only to widen the seed-noise estimate ---------
    "base_extra":  {},
}
# baseline runs that already exist with exactly the shipped config
BASELINE_RUNS = {2026: "a5b_flip_ap_s2026", 7: "a5b_flip_ap_s7", 77: "a5b_flip_ap_s77"}


def summarise(run: Path, variant: str, seed: int) -> dict:
    best = json.load(open(run / "best.json"))
    vb = json.load(open(run / "val_best.json"))["summary"]
    ep = [r for r in read_jsonl(run / "log.jsonl") if r.get("event") == "epoch"]
    bands = {k: {"dice": v["dice_mean"], "n": v["n"], "detect_frac": v.get("detect_frac"),
                 "median_abs_pct_err": v.get("median_abs_pct_err")}
             for k, v in vb["dice_by_gt_volume_ml"].items()}
    return {"variant": variant, "seed": seed, "run": run.as_posix(), "epochs": len(ep),
            "best_epoch": best["epoch"], "val_dice": best["val_dice"],
            "val_dice_ci95": vb["dice_pos_ci95"], "val_dice_median": vb["dice_pos_median"],
            "val_sens": vb["detection_sensitivity"], "val_sens_overlap": vb.get("detection_sensitivity_overlap"),
            "val_icc": vb["volume"]["icc21"], "val_icc_ci95": vb["volume"].get("icc21_ci95"),
            "val_mae_ml": vb["volume"]["mae_ml"], "val_median_abs_pct_err": vb["volume"]["median_abs_pct_err"],
            "small_dice": bands["[0,2)"]["dice"], "small_detect": bands["[0,2)"]["detect_frac"],
            "small_median_abs_pct_err": bands["[0,2)"]["median_abs_pct_err"], "bands": bands,
            "train_time_s": sum(r["train_time_s"] for r in ep) / max(len(ep), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/seg_unet2d.yaml"))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[2026, 7, 77])
    ap.add_argument("--epochs", type=int, default=None, help="override (default: the config's 40)")
    ap.add_argument("--out", type=Path, default=Path("results/small_lesion/experiments.json"))
    a = ap.parse_args()

    base = load_yaml(a.config)
    if a.epochs:
        base["epochs"] = a.epochs
    splits = json.load(open("data/splits.json"))  # train/val only; `train()` never touches splits["test"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a.out.parent.mkdir(parents=True, exist_ok=True)

    for variant in a.variants:
        if variant not in VARIANTS:
            raise SystemExit(f"unknown variant {variant!r}; known: {sorted(VARIANTS)}")
        for seed in a.seeds:
            name = f"small_{variant}_s{seed}"
            run = Path("runs") / name
            if (run / "best.json").exists():
                print("skip (done):", name, flush=True)
            elif run.exists():
                # another shard has claimed this run (mkdir below is the atomic claim); leave it alone
                print("skip (claimed by another shard):", name, flush=True)
                continue
            else:
                try:
                    run.mkdir(parents=True, exist_ok=False)  # atomic claim
                except FileExistsError:
                    print("skip (claimed by another shard):", name, flush=True)
                    continue
                cfg = {**base, **VARIANTS[variant], "seed": seed}
                t0 = time.time()
                train(cfg, run, Path("data/cache"), splits, device=device)
                print(name, "done in", round(time.time() - t0), "s", flush=True)
            # one file per run: several shards can run concurrently without clobbering each other
            rec = summarise(run, variant, seed)
            per_run = a.out.parent / "runs"
            per_run.mkdir(parents=True, exist_ok=True)
            json.dump(rec, open(per_run / f"{name}.json", "w"), indent=1)

    # always (re)record the pre-existing baseline runs, then merge every per-run file
    per_run = a.out.parent / "runs"
    per_run.mkdir(parents=True, exist_ok=True)
    for seed, rn in BASELINE_RUNS.items():
        p = Path("runs") / rn
        if (p / "best.json").exists():
            json.dump(summarise(p, "base", seed), open(per_run / f"base_s{seed}.json", "w"), indent=1)
    res = {f.stem: json.load(open(f)) for f in sorted(per_run.glob("*.json"))}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"{len(res)} runs -> {a.out}")


if __name__ == "__main__":
    main()
