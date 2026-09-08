#!/usr/bin/env python3
"""A4 benchmarks: data-loader throughput, forward/backward time, AMP / channels_last / cudnn settings,
grad-norm distribution and clip rate, and val-evaluation cost.

Only train/val caches are read (CLAUDE.md: test must not be touched).

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python scripts/analysis/a4_bench.py --part all
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from strokeai.data.dataset import SliceDataset, SubjectCache  # noqa: E402
from strokeai.losses import bce_dice_loss  # noqa: E402
from strokeai.models import UNet2D  # noqa: E402
from strokeai.train import evaluate_subjects  # noqa: E402
from strokeai.utils import load_yaml, seed_everything  # noqa: E402


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


# ------------------------------------------------------------------ data loader
def bench_loader(ds, batch_sizes, workers, n_batches, pin_memory, seed):
    rows = []
    for bs in batch_sizes:
        for nw in workers:
            g = torch.Generator().manual_seed(seed)
            kw = {}
            if nw > 0:
                kw["prefetch_factor"] = 4
            dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw, generator=g,
                            drop_last=True, pin_memory=pin_memory, **kw)
            it = iter(dl)
            t_first = time.time()
            next(it)
            first = time.time() - t_first
            t0 = time.time()
            n = 0
            for _ in range(n_batches - 1):
                try:
                    next(it)
                except StopIteration:
                    break
                n += 1
            dt = time.time() - t0
            del it, dl
            rows.append({"batch_size": bs, "num_workers": nw, "pin_memory": pin_memory,
                         "first_batch_s": round(first, 3), "batches": n,
                         "slices_per_s": round(n * bs / dt, 1), "s_per_batch": round(dt / max(n, 1), 4)})
            print(rows[-1], flush=True)
    return rows


# ------------------------------------------------------------------ compute only (synthetic tensors)
def bench_compute(cfg, device, batch_sizes, modes, size, iters=30, warmup=10):
    rows = []
    for bs in batch_sizes:
        for mode in modes:
            amp, ch_last, bench = mode["amp"], mode["channels_last"], mode["cudnn_benchmark"]
            torch.backends.cudnn.deterministic = not bench
            torch.backends.cudnn.benchmark = bench
            model = UNet2D(in_ch=cfg.get("in_channels", 2), base=cfg["base_channels"], depth=cfg["depth"]).to(device)
            if ch_last:
                model = model.to(memory_format=torch.channels_last)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=cfg["weight_decay"])
            scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16"))
            x = torch.randn(bs, cfg.get("in_channels", 2), size, size, device=device)
            y = (torch.rand(bs, 1, size, size, device=device) < 0.007).float()
            if ch_last:
                x = x.contiguous(memory_format=torch.channels_last)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            times = []
            for i in range(iters + warmup):
                if i == warmup:
                    _sync(device)
                    t0 = time.time()
                with torch.autocast("cuda", dtype={"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp, torch.float32),
                                    enabled=amp in ("bf16", "fp16")):
                    logits = model(x)
                    loss, _ = bce_dice_loss(logits.float(), y, dice_weight=cfg["dice_weight"])
                opt.zero_grad(set_to_none=True)
                if amp == "fp16":
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    opt.step()
            _sync(device)
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
            rows.append({"batch_size": bs, "size": size, "amp": amp or "fp32", "channels_last": ch_last,
                         "cudnn_benchmark": bench, "s_per_step": round(dt / iters, 5),
                         "slices_per_s": round(iters * bs / dt, 1), "peak_mem_MiB": round(peak, 1)})
            print(rows[-1], flush=True)
            del model, opt, x, y
            if device == "cuda":
                torch.cuda.empty_cache()
    return rows


# ------------------------------------------------------------------ end-to-end partial epoch
def bench_epoch(cfg, cache_dir, splits, device, num_workers, batch_size, amp, ch_last, bench,
                pin_memory, max_steps, do_val=True):
    seed_everything(cfg["seed"])
    torch.backends.cudnn.deterministic = not bench
    torch.backends.cudnn.benchmark = bench
    torch.set_num_threads(cfg.get("threads", 4))
    tc = SubjectCache(cache_dir, splits["train"])
    ds = SliceDataset(tc, neg_pos_ratio=cfg["neg_pos_ratio"], augment=cfg["augment"], seed=cfg["seed"])
    g = torch.Generator().manual_seed(cfg["seed"])
    kw = {"prefetch_factor": 4} if num_workers > 0 else {}
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=g,
                    drop_last=True, pin_memory=pin_memory, **kw)
    model = UNet2D(in_ch=2, base=cfg["base_channels"], depth=cfg["depth"]).to(device)
    if ch_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16"))
    gnorms, clipped, losses = [], 0, []
    _sync(device)
    t0 = time.time()
    steps = 0
    for x, y in dl:
        x = x.to(device, non_blocking=pin_memory)
        y = y.to(device, non_blocking=pin_memory)
        if ch_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype={"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp, torch.float32),
                            enabled=amp in ("bf16", "fp16")):
            logits = model(x)
        loss, _ = bce_dice_loss(logits.float(), y, dice_weight=cfg["dice_weight"], pos_weight=cfg.get("pos_weight"))
        opt.zero_grad(set_to_none=True)
        if amp == "fp16":
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]))
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]))
            opt.step()
        gnorms.append(gn)
        clipped += gn > cfg["grad_clip"]
        losses.append(float(loss.detach()))
        steps += 1
        if max_steps and steps >= max_steps:
            break
    _sync(device)
    train_dt = time.time() - t0
    out = {"num_workers": num_workers, "batch_size": batch_size, "amp": amp or "fp32", "channels_last": ch_last,
           "cudnn_benchmark": bench, "pin_memory": pin_memory, "steps": steps,
           "train_s": round(train_dt, 2), "slices_per_s": round(steps * batch_size / train_dt, 1),
           "s_per_step": round(train_dt / steps, 4),
           "steps_per_epoch": len(ds) // batch_size,
           "est_train_epoch_s": round(train_dt / steps * (len(ds) // batch_size), 1),
           "grad_norm_mean": round(float(np.mean(gnorms)), 4), "grad_norm_p50": round(float(np.percentile(gnorms, 50)), 4),
           "grad_norm_p95": round(float(np.percentile(gnorms, 95)), 4), "grad_norm_max": round(float(np.max(gnorms)), 4),
           "clip_frac": round(clipped / len(gnorms), 4), "loss_first": round(losses[0], 4),
           "loss_last10_mean": round(float(np.mean(losses[-10:])), 4)}
    if do_val:
        vc = SubjectCache(cache_dir, splits["val"])
        _sync(device)
        t1 = time.time()
        summ, _ = evaluate_subjects(model, vc, threshold=cfg["threshold"], device=device)
        _sync(device)
        out["val_s"] = round(time.time() - t1, 2)
        out["val_dice_pos"] = round(summ["dice_pos_mean"], 4)
    print(json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/seg_unet2d.yaml"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--out", type=Path, default=Path("results/a4"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--part", default="all", choices=["all", "loader", "compute", "epoch"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(a.config)
    splits = json.load(open(a.splits))
    res = {}

    if a.part in ("all", "compute"):
        modes = [
            {"amp": None, "channels_last": False, "cudnn_benchmark": False},   # current default (deterministic)
            {"amp": None, "channels_last": False, "cudnn_benchmark": True},
            {"amp": None, "channels_last": True, "cudnn_benchmark": True},
            {"amp": "bf16", "channels_last": False, "cudnn_benchmark": False},
            {"amp": "bf16", "channels_last": False, "cudnn_benchmark": True},
            {"amp": "bf16", "channels_last": True, "cudnn_benchmark": True},
            {"amp": "fp16", "channels_last": True, "cudnn_benchmark": True},
        ]
        res["compute"] = bench_compute(cfg, a.device, [32, 64, 128, 256], modes, cfg["size"])

    if a.part in ("all", "loader"):
        seed_everything(cfg["seed"])
        tc = SubjectCache(a.cache, splits["train"])
        ds = SliceDataset(tc, neg_pos_ratio=cfg["neg_pos_ratio"], augment=cfg["augment"], seed=cfg["seed"])
        res["loader"] = bench_loader(ds, [32, 64, 128], [0, 2, 4, 8, 12, 16], 60, True, cfg["seed"])
        del tc, ds

    if a.part in ("all", "epoch"):
        combos = [
            dict(num_workers=0, batch_size=32, amp=None, ch_last=False, bench=False, pin_memory=False),
            dict(num_workers=8, batch_size=32, amp=None, ch_last=False, bench=False, pin_memory=True),
            dict(num_workers=8, batch_size=64, amp="bf16", ch_last=True, bench=True, pin_memory=True),
            dict(num_workers=12, batch_size=64, amp="bf16", ch_last=True, bench=True, pin_memory=True),
            dict(num_workers=12, batch_size=64, amp=None, ch_last=False, bench=False, pin_memory=True),
        ]
        res["epoch"] = [bench_epoch(cfg, a.cache, splits, a.device, max_steps=120, **c) for c in combos]

    p = a.out / f"a4_bench{('_' + a.tag) if a.tag else ''}_{a.part}.json"
    json.dump(res, open(p, "w"), indent=1, default=float)
    print("written", p)
