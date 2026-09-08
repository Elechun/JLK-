#!/usr/bin/env python3
"""A4 sanity checks: LR schedule, loss numerical stability, batch-composition sensitivity,
and an over-fit check (a model that cannot drive a 1-2 batch loss to ~0 is mis-wired).

Reads only train/val (CLAUDE.md: the test split must not be touched).

    PYTHONPATH=src .venv/bin/python scripts/analysis/a4_sanity.py --out results/a4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from strokeai.data.dataset import SliceDataset, SubjectCache  # noqa: E402
from strokeai.losses import bce_dice_loss, soft_dice_loss  # noqa: E402
from strokeai.models import UNet2D  # noqa: E402
from strokeai.train import cosine_warmup  # noqa: E402
from strokeai.utils import load_yaml, seed_everything  # noqa: E402


def lr_schedule_check(cfg: dict, steps_per_epoch: int) -> dict:
    total = steps_per_epoch * cfg["epochs"]
    warmup = int(cfg["warmup_frac"] * total)
    lrs = [cosine_warmup(s, total, warmup, cfg["lr"], cfg["min_lr"]) for s in range(total)]
    a = np.asarray(lrs)
    peak = int(a.argmax())
    return {
        "steps_per_epoch": steps_per_epoch, "total_steps": total, "warmup_steps": warmup,
        "lr_step0": lrs[0], "lr_at_warmup_end": lrs[warmup - 1] if warmup else None,
        "lr_first_after_warmup": lrs[warmup] if warmup < total else None,
        "peak_step": peak, "peak_lr": float(a.max()), "lr_last": lrs[-1],
        "monotone_decreasing_after_peak": bool(np.all(np.diff(a[peak:]) <= 1e-12)),
        "monotone_increasing_before_peak": bool(np.all(np.diff(a[: peak + 1]) >= -1e-12)),
        "lr_per_epoch_start": [lrs[e * steps_per_epoch] for e in range(cfg["epochs"])],
        "min_lr_target": cfg["min_lr"], "reaches_min_lr_within_1pct": bool(abs(lrs[-1] - cfg["min_lr"]) < 0.01 * cfg["lr"]),
    }


def loss_stability() -> dict:
    """Edge cases: all-empty target, all-positive target, extreme logits, and batch-composition drift."""
    torch.manual_seed(0)
    out = {}
    B, H, W = 8, 32, 32
    empty = torch.zeros(B, 1, H, W)
    full = torch.ones(B, 1, H, W)

    for name, logits, target in [
        ("empty_target_zero_logits", torch.zeros(B, 1, H, W), empty),
        ("empty_target_confident_neg", torch.full((B, 1, H, W), -20.0), empty),
        ("empty_target_confident_pos", torch.full((B, 1, H, W), 20.0), empty),
        ("full_target_confident_pos", torch.full((B, 1, H, W), 20.0), full),
        ("full_target_confident_neg", torch.full((B, 1, H, W), -20.0), full),
        ("extreme_logits_1e4", torch.full((B, 1, H, W), 1e4), empty),
    ]:
        lg = logits.clone().requires_grad_(True)
        loss, parts = bce_dice_loss(lg, target)
        loss.backward()
        out[name] = {"loss": float(loss), "bce": parts["bce"], "dice": parts["dice"],
                     "finite": bool(torch.isfinite(loss)), "grad_finite": bool(torch.isfinite(lg.grad).all()),
                     "grad_absmax": float(lg.grad.abs().max())}

    # smooth term: how large is the "free" Dice reward on an all-empty batch of realistic size?
    n_px = 32 * 1 * 128 * 128
    out["smooth_term_scale"] = {
        "smooth": 1.0, "batch_pixels_at_bs32_128sq": n_px,
        "dice_loss_empty_batch_perfect_pred": float(soft_dice_loss(torch.full((1, 1, 128, 128), -20.0), torch.zeros(1, 1, 128, 128))),
        "note": "with an all-empty target and p->0, dice = 1 - smooth/smooth = 0 (no gradient pathology)",
    }

    # batch-level Dice vs per-sample Dice: value depends on how many positives land in the batch.
    rows = []
    g = torch.Generator().manual_seed(1)
    for n_pos in [0, 1, 2, 4, 8, 16]:
        tgt = torch.zeros(16, 1, 64, 64)
        for i in range(n_pos):
            tgt[i, 0, 20:30, 20:30] = 1.0  # 100 positive px per positive slice
        lg = torch.randn(16, 1, 64, 64, generator=g) * 0.5
        batch_dice = float(soft_dice_loss(lg, tgt))
        per_sample = float(torch.stack([soft_dice_loss(lg[i:i + 1], tgt[i:i + 1]) for i in range(16)]).mean())
        rows.append({"n_pos_slices": n_pos, "batch_level_dice_loss": batch_dice, "per_sample_mean_dice_loss": per_sample})
    out["batch_composition"] = rows
    return out


def real_batch_composition(cache_dir: Path, splits: dict, cfg: dict, n_batches: int = 200) -> dict:
    """How much does the batch-level (micro) Dice term move with the batch composition, on real batches?"""
    seed_everything(cfg["seed"])
    cache = SubjectCache(cache_dir, splits["train"])
    ds = SliceDataset(cache, neg_pos_ratio=cfg["neg_pos_ratio"], augment=False, seed=cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    pos_px, dice_batch, dice_sample = [], [], []
    g = torch.Generator().manual_seed(0)
    for _ in range(n_batches):
        idx = rng.choice(len(ds), size=cfg["batch_size"], replace=False)
        y = torch.stack([ds[int(i)][1] for i in idx])
        logits = torch.randn(y.shape, generator=g) * 0.5 - 2.0  # a fixed, mediocre "prediction"
        pos_px.append(float(y.sum()))
        dice_batch.append(float(soft_dice_loss(logits, y, reduction="batch")))
        dice_sample.append(float(soft_dice_loss(logits, y, reduction="sample")))
    pos_px = np.asarray(pos_px)
    db, dsm = np.asarray(dice_batch), np.asarray(dice_sample)
    return {"n_batches": n_batches, "batch_size": cfg["batch_size"],
            "pos_pixels_per_batch": {"p5": float(np.percentile(pos_px, 5)), "p50": float(np.median(pos_px)),
                                     "p95": float(np.percentile(pos_px, 95)), "min": float(pos_px.min()),
                                     "max": float(pos_px.max()),
                                     "p95_over_p5": float(np.percentile(pos_px, 95) / max(np.percentile(pos_px, 5), 1))},
            "dice_loss_batch_reduction": {"mean": float(db.mean()), "sd": float(db.std()), "min": float(db.min()), "max": float(db.max())},
            "dice_loss_sample_reduction": {"mean": float(dsm.mean()), "sd": float(dsm.std()), "min": float(dsm.min()), "max": float(dsm.max())},
            "corr_dice_batch_vs_pos_pixels": float(np.corrcoef(db, pos_px)[0, 1]),
            "corr_dice_sample_vs_pos_pixels": float(np.corrcoef(dsm, pos_px)[0, 1])}


def overfit_check(cache_dir: Path, splits: dict, cfg: dict, device: str, n_batches: int, steps: int) -> dict:
    seed_everything(cfg["seed"])
    cache = SubjectCache(cache_dir, splits["train"][:24])
    ds = SliceDataset(cache, neg_pos_ratio=cfg["neg_pos_ratio"], augment=False, seed=cfg["seed"])
    xs, ys = [], []
    for b in range(n_batches):
        idx = range(b * cfg["batch_size"], (b + 1) * cfg["batch_size"])
        batch = [ds[i] for i in idx]
        xs.append(torch.stack([b_[0] for b_ in batch]).to(device))
        ys.append(torch.stack([b_[1] for b_ in batch]).to(device))
    pos_frac = float(torch.cat(ys).mean())
    model = UNet2D(in_ch=2, base=cfg["base_channels"], depth=cfg["depth"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.0)
    hist = []
    t0 = time.time()
    for s in range(steps):
        tot = 0.0
        for x, y in zip(xs, ys):
            logits = model(x)
            loss, parts = bce_dice_loss(logits, y, dice_weight=cfg["dice_weight"], pos_weight=cfg.get("pos_weight"))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        if s % 25 == 0 or s == steps - 1:
            with torch.no_grad():
                p = (torch.sigmoid(model(xs[0])) >= 0.5).float()
                d = float((2 * (p * ys[0]).sum() + 1) / (p.sum() + ys[0].sum() + 1))
            hist.append({"step": s, "loss": tot / len(xs), "hard_dice_batch0": d})
    return {"n_batches": n_batches, "batch_size": cfg["batch_size"], "steps": steps, "device": device,
            "positive_pixel_fraction": pos_frac, "seconds": round(time.time() - t0, 1),
            "loss_first": hist[0]["loss"], "loss_last": hist[-1]["loss"],
            "hard_dice_last": hist[-1]["hard_dice_batch0"], "history": hist}


def data_stats(cache_dir: Path, splits: dict, cfg: dict) -> dict:
    cache = SubjectCache(cache_dir, splits["train"])
    tab = cache.slice_table()
    pos = sum(1 for t in tab if t[2])
    ds = SliceDataset(cache, neg_pos_ratio=cfg["neg_pos_ratio"], augment=False, seed=cfg["seed"])
    # positive pixel fraction over the sampled epoch
    tot_px, tot_pos = 0, 0
    for sid in cache.ids:
        m = cache.mask[sid]
        tot_px += m.size
        tot_pos += int(m.sum())
    return {"n_train_subjects": len(cache.ids), "n_slices_total": len(tab), "n_slices_positive": pos,
            "pos_slice_frac": pos / len(tab), "epoch_items": len(ds),
            "steps_per_epoch_bs{}".format(cfg["batch_size"]): len(ds) // cfg["batch_size"],
            "lesion_pixel_frac_all_slices": tot_pos / tot_px}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/seg_unet2d.yaml"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--out", type=Path, default=Path("results/a4"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overfit-batches", type=int, default=2)
    ap.add_argument("--overfit-steps", type=int, default=300)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(a.config)
    splits = json.load(open(a.splits))

    res = {"config": cfg}
    res["data_stats"] = data_stats(a.cache, splits, cfg)
    res["lr_schedule"] = lr_schedule_check(cfg, res["data_stats"][f"steps_per_epoch_bs{cfg['batch_size']}"])
    res["loss_stability"] = loss_stability()
    res["real_batch_composition"] = real_batch_composition(a.cache, splits, cfg)
    res["overfit"] = overfit_check(a.cache, splits, cfg, a.device, a.overfit_batches, a.overfit_steps)
    json.dump(res, open(a.out / "a4_sanity.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=1, default=float)[:6000])
