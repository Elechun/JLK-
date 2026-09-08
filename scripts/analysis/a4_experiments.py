#!/usr/bin/env python3
"""A4 experiment harness.

Loads `configs/seg_unet2d.yaml` and applies a *named, documented* delta on top of it, then trains into
`runs/a4_<name>/`.  Nothing is hard-coded in the library: every run writes its effective config to
`runs/a4_<name>/config.json`, and the winning values are copied back into `configs/seg_unet2d.yaml`.

Only train/val are ever read (CLAUDE.md: the test split stays untouched).

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python scripts/analysis/a4_experiments.py --group infra
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

# Stage-1 (2026-09-09) ran against the clip=6 cache; the clip sweep then moved the shipped cache to
# clip=10, so every later group was re-run against the regenerated `data/cache`.  Stage-1 runs are kept
# in runs/a4_stage1_clip6/ and results/a4/a4_exp_*_stage1clip6.json.
# infra settings A4 measured as best on one RTX A6000 (see docs/agents/A4_report.md)
OPT = {"batch_size": 64, "num_workers": 8, "amp": "bf16", "channels_last": True,
       "pin_memory": True, "cudnn_benchmark": False, "prefetch_factor": 4}

GROUPS: dict[str, list[dict]] = {
    # (a) infrastructure: does the fast path change the result, and is it reproducible?
    "infra": [
        {"name": "infra_baseline_cpuconfig", "cfg": {"epochs": 12}},
        {"name": "infra_opt", "cfg": {**OPT, "epochs": 12}},
        {"name": "infra_opt_rep", "cfg": {**OPT, "epochs": 12}},          # identical -> reproducibility
        {"name": "infra_opt_nw0", "cfg": {**OPT, "num_workers": 0, "epochs": 12}},
        {"name": "infra_opt_nw16", "cfg": {**OPT, "num_workers": 16, "epochs": 12}},
        {"name": "infra_opt_fp32", "cfg": {**OPT, "amp": None, "channels_last": False, "epochs": 12}},
    ],
    # (b) how many epochs does this model actually need, and does the batch size matter?
    "epochs": [
        {"name": "ep24_bs64", "cfg": {**OPT, "epochs": 24}},
        {"name": "ep40_bs64", "cfg": {**OPT, "epochs": 40}},
        {"name": "ep40_bs32", "cfg": {**OPT, "epochs": 40, "batch_size": 32}},
        {"name": "ep60_bs64", "cfg": {**OPT, "epochs": 60}},
        {"name": "ep60_bs32", "cfg": {**OPT, "epochs": 60, "batch_size": 32}},
        {"name": "ep100_bs64", "cfg": {**OPT, "epochs": 100}},
        {"name": "ep150_bs64", "cfg": {**OPT, "epochs": 150}},
    ],
    # (b2) grad clip: the 12-epoch run clips ~26 % of steps, not the 0 % of the CPU smoke test
    "clipgrad": [
        {"name": "gc1", "cfg": {**OPT, "epochs": 24}},
        {"name": "gc5", "cfg": {**OPT, "epochs": 24, "grad_clip": 5.0}},
        {"name": "gc100", "cfg": {**OPT, "epochs": 24, "grad_clip": 100.0}},
    ],
    # (c) normalisation clip (A3 hand-off F3): each variant has its own cache
    "clip": [
        {"name": "clip06", "cfg": {**OPT, "epochs": 24}, "cache": "data/cache_clip06"},
        {"name": "clip10", "cfg": {**OPT, "epochs": 24}, "cache": "data/cache"},
        {"name": "clip14", "cfg": {**OPT, "epochs": 24}, "cache": "data/cache_clip14"},
        {"name": "clippct01", "cfg": {**OPT, "epochs": 24}, "cache": "data/cache_pct01"},
    ],
    # (d) ADC channel ablation (A3 hand-off): lesion ADC z median is only -0.19
    "adc": [
        {"name": "adc_2ch", "cfg": {**OPT, "epochs": 24}},
        {"name": "adc_traceonly", "cfg": {**OPT, "epochs": 24, "channels": [0]}},
        {"name": "adc_adconly", "cfg": {**OPT, "epochs": 24, "channels": [1]}},
    ],
    "capacity": [
        {"name": "cap_base16", "cfg": {**OPT, "epochs": 24}},
        {"name": "cap_base32", "cfg": {**OPT, "epochs": 24, "base_channels": 32}},
    ],
    # (e) in-plane resolution (A3 hand-off item 4): needs its own preprocessed cache
    "size": [
        {"name": "size128", "cfg": {**OPT, "epochs": 24, "size": 128}, "cache": "data/cache"},
        {"name": "size192", "cfg": {**OPT, "epochs": 24, "size": 192}, "cache": "data/cache_192_clip10"},
    ],
    "lr": [
        {"name": "lr1e3", "cfg": {**OPT, "epochs": 24}},
        {"name": "lr3e3", "cfg": {**OPT, "epochs": 24, "lr": 0.003}},
        {"name": "lr3e4", "cfg": {**OPT, "epochs": 24, "lr": 0.0003}},
    ],
    "loss": [
        {"name": "loss_default", "cfg": {**OPT, "epochs": 24}},
        {"name": "loss_dice_sample", "cfg": {**OPT, "epochs": 24, "dice_reduction": "sample"}},
        {"name": "loss_posw8", "cfg": {**OPT, "epochs": 24, "pos_weight": 8.0}},
        {"name": "loss_negpos2", "cfg": {**OPT, "epochs": 24, "neg_pos_ratio": 2.0}},
    ],
}


def run_one(base: dict, spec: dict, splits: dict, device: str, runs_dir: Path, default_cache: Path,
            seed: int | None = None) -> dict:
    cfg = {**base, **spec["cfg"]}
    if seed is not None:
        cfg["seed"] = seed
    cache = Path(spec.get("cache", default_cache))
    name = spec["name"] + (f"_seed{seed}" if seed is not None else "")
    run_dir = runs_dir / f"a4_{name}"
    t0 = time.time()
    best = train(cfg, run_dir, cache, splits, device=device)
    wall = time.time() - t0
    recs = read_jsonl(run_dir / "log.jsonl")
    ep = [r for r in recs if r.get("event") == "epoch"]
    start = next(r for r in recs if r.get("event") == "start")
    out = {"name": name, "cache": str(cache), "epochs": cfg["epochs"], "batch_size": cfg["batch_size"],
           "num_workers": cfg.get("num_workers", 0), "amp": cfg.get("amp") or "fp32",
           "channels_last": bool(cfg.get("channels_last", False)),
           "channels": cfg.get("channels"), "seed": cfg["seed"], "n_params": start["n_params"],
           "steps_per_epoch": start["steps_per_epoch"],
           "best_val_dice": best["val_dice"], "best_epoch": best["epoch"],
           "final_val_dice": ep[-1]["val_dice_pos"],
           "val_dice_curve": [round(r["val_dice_pos"], 4) for r in ep],
           "train_s_per_epoch_median": float(sorted(r["train_time_s"] for r in ep)[len(ep) // 2]),
           "val_s_per_epoch_median": float(sorted(r["val_time_s"] for r in ep)[len(ep) // 2]),
           "wall_s": round(wall, 1), "clip_frac_mean": sum(r["clip_frac"] for r in ep) / len(ep),
           "grad_norm_p95_first_epoch": ep[0]["grad_norm_p95"], "grad_norm_p95_last_epoch": ep[-1]["grad_norm_p95"],
           "train_loss_first": ep[0]["train_loss"], "train_loss_last": ep[-1]["train_loss"],
           "val_sens_best": max(r["val_sens"] for r in ep), "val_sens_last": ep[-1]["val_sens"],
           "val_icc_last": ep[-1].get("val_vol_icc")}
    print(json.dumps({k: v for k, v in out.items() if k != "val_dice_curve"}), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/seg_unet2d.yaml"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("results/a4"))
    ap.add_argument("--group", required=True, choices=sorted(GROUPS) + ["custom"])
    ap.add_argument("--only", nargs="*", default=None, help="subset of run names inside the group")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="repeat every run with these seeds")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    base = load_yaml(a.config)
    splits = json.load(open(a.splits))
    specs = [s for s in GROUPS[a.group] if a.only is None or s["name"] in a.only]
    rows = []
    for s in specs:
        for seed in (a.seeds or [None]):
            rows.append(run_one(base, s, splits, a.device, a.runs, a.cache, seed))
    p = a.out / f"a4_exp_{a.group}.json"
    prev = json.load(open(p)) if p.exists() else []
    json.dump(prev + rows, open(p, "w"), indent=1, default=float)
    print("written", p)
