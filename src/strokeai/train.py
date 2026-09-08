"""Training loop for the 2-D U-Net. Config-driven (configs/*.yaml). Logs JSONL; selects best epoch on val Dice."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.dataset import SliceDataset, SubjectCache
from .losses import bce_dice_loss
from .metrics import dice_binary, segmentation_summary, volume_ml
from .models import UNet2D
from .utils import JsonlLogger, seed_everything


def cosine_warmup(step: int, total: int, warmup: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def predict_subject(model: torch.nn.Module, img: np.ndarray, batch: int = 64, device="cpu", tta_flip: bool = False) -> np.ndarray:
    """img: (S, 2, H, W) float16 -> prob (S, H, W) float32."""
    model.eval()
    out = []
    for i in range(0, len(img), batch):
        x = torch.from_numpy(img[i:i + batch].astype(np.float32)).to(device)
        p = torch.sigmoid(model(x))
        if tta_flip:
            p = 0.5 * (p + torch.flip(model(torch.flip(x, dims=[3])).sigmoid(), dims=[3]))
        out.append(p[:, 0].cpu().numpy())
    return np.concatenate(out, axis=0)


def evaluate_subjects(model, cache: SubjectCache, threshold: float = 0.5, device="cpu", tta_flip=False, min_voxels: int = 0) -> tuple[dict, list[dict]]:
    per = []
    for sid in cache.ids:
        prob = predict_subject(model, cache.img[sid], device=device, tta_flip=tta_flip)
        pred = prob >= threshold
        if min_voxels and pred.sum() < min_voxels:
            pred[:] = False
        gt = cache.mask[sid] > 0
        vv = cache.meta[sid]["voxel_volume_mm3"]
        per.append({"sid": sid, "dice": dice_binary(pred, gt), "gt_ml": volume_ml(gt, vv), "pred_ml": volume_ml(pred, vv),
                    "gt_pos": bool(gt.any()), "pred_pos": bool(pred.any())})
    return segmentation_summary(per), per


def train(cfg: dict, run_dir: Path, cache_dir: Path, splits: dict, device: str = "cpu", max_steps: int | None = None) -> dict:
    seed_everything(cfg["seed"])
    torch.set_num_threads(cfg.get("threads", 4))
    run_dir.mkdir(parents=True, exist_ok=True)
    log = JsonlLogger(run_dir / "log.jsonl")
    json.dump(cfg, open(run_dir / "config.json", "w"), indent=1)

    train_cache = SubjectCache(cache_dir, splits["train"])
    val_cache = SubjectCache(cache_dir, splits["val"])
    assert not set(train_cache.ids) & set(val_cache.ids), "train/val overlap"
    ds = SliceDataset(train_cache, neg_pos_ratio=cfg["neg_pos_ratio"], augment=cfg["augment"], seed=cfg["seed"])
    g = torch.Generator().manual_seed(cfg["seed"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 0), generator=g, drop_last=True)

    model = UNet2D(in_ch=2, base=cfg["base_channels"], depth=cfg["depth"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = len(dl)
    total = steps_per_epoch * cfg["epochs"]
    warmup = int(cfg["warmup_frac"] * total)
    log.log(event="start", n_params=n_params, n_train_subjects=len(train_cache.ids), n_val_subjects=len(val_cache.ids),
            n_train_slices_per_epoch=len(ds), steps_per_epoch=steps_per_epoch, total_steps=total)

    best = {"val_dice": -1.0, "epoch": -1}
    step = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        ds.resample()
        t0 = time.time()
        losses, gnorms, clipped = [], [], 0
        for x, y in dl:
            lr = cosine_warmup(step, total, warmup, cfg["lr"], cfg["min_lr"])
            for pg in opt.param_groups:
                pg["lr"] = lr
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss, parts = bce_dice_loss(logits, y, dice_weight=cfg["dice_weight"], pos_weight=cfg.get("pos_weight"))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]))
            clipped += gn > cfg["grad_clip"]
            opt.step()
            losses.append(float(loss.detach()))
            gnorms.append(gn)
            step += 1
            if step % cfg.get("log_every", 50) == 0:
                log.log(event="step", epoch=epoch, step=step, loss=float(loss.detach()), bce=parts["bce"], dice_loss=parts["dice"], lr=lr, grad_norm=gn,
                        slices_per_s=len(losses) * cfg["batch_size"] / (time.time() - t0))
            if max_steps and step >= max_steps:
                break
        train_time = time.time() - t0
        t1 = time.time()
        val_summary, per = evaluate_subjects(model, val_cache, threshold=cfg["threshold"], device=device)
        val_dice = val_summary["dice_pos_mean"]
        rec = dict(event="epoch", epoch=epoch, train_loss=float(np.mean(losses)), grad_norm_mean=float(np.mean(gnorms)),
                   grad_norm_p95=float(np.percentile(gnorms, 95)), clip_frac=clipped / max(len(gnorms), 1), lr_end=lr,
                   train_time_s=train_time, val_time_s=time.time() - t1, val_dice_pos=val_dice,
                   val_dice_ci=val_summary["dice_pos_ci95"], val_sens=val_summary["detection_sensitivity"],
                   val_spec=val_summary["detection_specificity"], val_vol_icc=val_summary.get("volume", {}).get("icc21"))
        log.log(**rec)
        print(json.dumps(rec), flush=True)
        torch.save(model.state_dict(), run_dir / "last.pt")
        if val_dice > best["val_dice"]:
            best = {"val_dice": val_dice, "epoch": epoch}
            torch.save(model.state_dict(), run_dir / "best.pt")
            json.dump({"summary": val_summary, "per_subject": per}, open(run_dir / "val_best.json", "w"), indent=1)
        if max_steps and step >= max_steps:
            break
    json.dump(best, open(run_dir / "best.json", "w"))
    log.log(event="end", **best)
    return best
