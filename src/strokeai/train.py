"""Training loop for the 2-D U-Net. Config-driven (configs/*.yaml). Logs JSONL; selects best epoch on val Dice."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.dataset import FLIP_AXES, SliceDataset, SubjectCache, context_channel_count, stack_context
from .losses import bce_dice_loss
from .metrics import dice_binary, segmentation_summary, volume_ml
from .models import UNet2D
from .utils import JsonlLogger, seed_everything


def cosine_warmup(step: int, total: int, warmup: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


_AMP_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}


def amp_context(amp: str | None, device: str):
    """autocast context for `amp` in {None, 'bf16', 'fp16'}; a no-op on CPU or when amp is falsy."""
    enabled = bool(amp) and str(device).startswith("cuda")
    return torch.autocast("cuda", dtype=_AMP_DTYPE.get(amp, torch.float32), enabled=enabled)


@torch.no_grad()
def predict_subject(model: torch.nn.Module, img: np.ndarray, batch: int = 64, device="cpu", tta_flip: bool = False,
                    channels: list[int] | None = None, amp: str | None = None, channels_last: bool = False,
                    flip_axis: str = "lr", context: int = 0, context_mode: str = "all") -> np.ndarray:
    """img: (S, C, H, W) float16 -> prob (S, H, W) float32.

    `tta_flip` averages with the mirrored input; the mirror axis follows `flip_axis` (same convention as
    `SliceDataset`: "lr" = H axis = tensor dim 2, "ap" = W axis = dim 3).  Not used by the shipped config.
    `context` > 0 builds the same 2.5-D stack the training set builds (A4b H4), so train and inference
    cannot drift apart: both go through `stack_context`.
    """
    model.eval()
    tta_dims = [ax + 1 for ax in FLIP_AXES[flip_axis]]  # (C,H,W) axis -> (B,C,H,W) dim
    out = []
    for i in range(0, len(img), batch):
        if context > 0:
            arr = np.stack([stack_context(img, s, context, context_mode, channels)
                            for s in range(i, min(i + batch, len(img)))])
        else:
            arr = img[i:i + batch].astype(np.float32)
            if channels is not None:
                arr = arr[:, channels]
        x = torch.from_numpy(arr).to(device)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with amp_context(amp, device):
            p = torch.sigmoid(model(x).float())
            if tta_flip and tta_dims:
                p = 0.5 * (p + torch.flip(model(torch.flip(x, dims=tta_dims)).float().sigmoid(), dims=tta_dims))
        out.append(p[:, 0].float().cpu().numpy())
    return np.concatenate(out, axis=0)


def evaluate_subjects(model, cache: SubjectCache, threshold: float = 0.5, device="cpu", tta_flip=False, min_voxels: int = 0,
                      channels: list[int] | None = None, amp: str | None = None, channels_last: bool = False,
                      context: int = 0, context_mode: str = "all") -> tuple[dict, list[dict]]:
    per = []
    for sid in cache.ids:
        prob = predict_subject(model, cache.img[sid], device=device, tta_flip=tta_flip, channels=channels,
                               amp=amp, channels_last=channels_last, context=context, context_mode=context_mode)
        pred = prob >= threshold
        if min_voxels and pred.sum() < min_voxels:
            pred[:] = False
        gt = cache.mask[sid] > 0
        vv = cache.meta[sid]["voxel_volume_mm3"]
        per.append({"sid": sid, "dice": dice_binary(pred, gt), "gt_ml": volume_ml(gt, vv), "pred_ml": volume_ml(pred, vv),
                    "gt_pos": bool(gt.any()), "pred_pos": bool(pred.any()), "overlap_pos": bool((pred & gt).any())})
    return segmentation_summary(per), per


def train(cfg: dict, run_dir: Path, cache_dir: Path, splits: dict, device: str = "cpu", max_steps: int | None = None) -> dict:
    seed_everything(cfg["seed"])
    # seed_everything sets cudnn.deterministic=True / benchmark=False.  `cudnn_benchmark: true` in the
    # config trades that reproducibility guarantee for speed; A4 measured the gain at ~2 %, so the
    # shipped config leaves it off.
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(cfg.get("threads", 4))
    run_dir.mkdir(parents=True, exist_ok=True)
    log = JsonlLogger(run_dir / "log.jsonl")
    json.dump(cfg, open(run_dir / "config.json", "w"), indent=1)

    channels = cfg.get("channels")  # None = both cached channels (TRACE, ADC)
    # A4b H4: `context` neighbours on each side are stacked as extra input channels (0 = plain 2-D).
    context = int(cfg.get("context", 0) or 0)
    context_mode = cfg.get("context_mode", "all")
    in_ch = context_channel_count(2 if channels is None else len(channels), context, context_mode)
    amp = cfg.get("amp")  # None | "bf16" | "fp16"
    ch_last = bool(cfg.get("channels_last", False))
    pin = bool(cfg.get("pin_memory", False)) and str(device).startswith("cuda")
    num_workers = int(cfg.get("num_workers", 0))

    train_cache = SubjectCache(cache_dir, splits["train"])
    val_cache = SubjectCache(cache_dir, splits["val"])
    assert not set(train_cache.ids) & set(val_cache.ids), "train/val overlap"
    # Guard against a stale cache: `size` (and `clip`) live in the config, but the cache is materialised by
    # scripts/preprocess.py.  Training on a 128^2 cache while the config asks for 192^2 used to pass silently.
    cached_size = int(train_cache.img[train_cache.ids[0]].shape[-1])
    assert cached_size == int(cfg.get("size", cached_size)), (
        f"cache at {cache_dir} is {cached_size}^2 but the config asks for {cfg.get('size')}^2 -- "
        f"re-run scripts/preprocess.py (it reads size/clip from the same config)")
    # `flip_axis` (A5b): "lr" = anatomical left-right mirror.  runs/seg_unet2d was trained before the key
    # existed, with the W-axis flip = "ap"; reproduce it with `flip_axis: ap` in the config.
    ds = SliceDataset(train_cache, neg_pos_ratio=cfg["neg_pos_ratio"], augment=cfg["augment"], seed=cfg["seed"],
                      channels=channels, flip_axis=cfg.get("flip_axis", "lr"),
                      slice_weight=cfg.get("slice_weight", "none"),
                      slice_weight_power=float(cfg.get("slice_weight_power", 1.0)),
                      context=context, context_mode=context_mode)
    assert ds.in_channels == in_ch, (ds.in_channels, in_ch)
    g = torch.Generator().manual_seed(cfg["seed"])
    # NOTE: persistent_workers must stay False -- `ds.resample()` runs in the parent between epochs and
    # persistent workers would keep serving the *first* epoch's item list.
    dl_kw = {"prefetch_factor": int(cfg.get("prefetch_factor", 4))} if num_workers > 0 else {}
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=num_workers, generator=g,
                    drop_last=True, pin_memory=pin, **dl_kw)

    model = UNet2D(in_ch=in_ch, base=cfg["base_channels"], depth=cfg["depth"]).to(device)
    if ch_last:
        model = model.to(memory_format=torch.channels_last)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16" and str(device).startswith("cuda")))
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = len(dl)
    total = steps_per_epoch * cfg["epochs"]
    warmup = int(cfg["warmup_frac"] * total)
    log.log(event="start", n_params=n_params, n_train_subjects=len(train_cache.ids), n_val_subjects=len(val_cache.ids),
            n_train_slices_per_epoch=len(ds), steps_per_epoch=steps_per_epoch, total_steps=total, in_channels=in_ch,
            amp=amp or "fp32", channels_last=ch_last, num_workers=num_workers, pin_memory=pin,
            cudnn_benchmark=bool(torch.backends.cudnn.benchmark), device=str(device), flip_axis=ds.flip_axis,
            slice_weight=ds.slice_weight, slice_weight_power=ds.slice_weight_power, context=context,
            context_mode=context_mode, region_loss=cfg.get("region_loss", "dice"),
            dice_reduction=cfg.get("dice_reduction", "batch"))

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
            x = x.to(device, non_blocking=pin)
            y = y.to(device, non_blocking=pin)
            if ch_last:
                x = x.contiguous(memory_format=torch.channels_last)
            with amp_context(amp, device):
                logits = model(x)
            # the loss is always computed in fp32: BCE-with-logits and the Dice sum over ~5e5 pixels are
            # the numerically sensitive parts, and they cost nothing next to the convolutions.
            loss, parts = bce_dice_loss(logits.float(), y, dice_weight=cfg["dice_weight"], pos_weight=cfg.get("pos_weight"),
                                        dice_reduction=cfg.get("dice_reduction", "batch"),
                                        region_loss=cfg.get("region_loss", "dice"),
                                        tversky_alpha=float(cfg.get("tversky_alpha", 0.7)),
                                        tversky_beta=float(cfg.get("tversky_beta", 0.3)),
                                        tversky_gamma=float(cfg.get("tversky_gamma", 1.0)),
                                        invsize_power=float(cfg.get("invsize_power", 0.5)))
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]))
                clipped += gn > cfg["grad_clip"]
                scaler.step(opt)
                scaler.update()
            else:
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
        val_summary, per = evaluate_subjects(model, val_cache, threshold=cfg["threshold"], device=device,
                                             channels=channels, amp=amp, channels_last=ch_last,
                                             context=context, context_mode=context_mode)
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
