"""A4 regression tests for the config-driven training options added on 2026-09-09.

Covered: normalisation clip modes, the ADC channel ablation switch, the AMP context helper,
and the LR schedule (warm-up -> cosine) actually producing the values the config asks for.
"""
import json

import numpy as np
import pytest
import torch

from strokeai.data.dataset import SliceDataset, SubjectCache
from strokeai.data.preprocess import robust_zscore
from strokeai.train import amp_context, cosine_warmup, train


# ---------------------------------------------------------------- normalisation clip
def test_clip_mode_mad_saturates_at_the_requested_sigma():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, (8, 16, 16))
    x[0, 0, 0] = 500.0  # outlier that must be clipped
    fg = np.ones_like(x, bool)
    for clip in (6.0, 10.0):
        z = robust_zscore(x, fg, clip=clip)
        assert z.max() <= clip + 1e-5 and z.min() >= -clip - 1e-5
        assert np.isclose(z.max(), clip, atol=1e-4)  # the outlier lands exactly on the clip


def test_wider_clip_keeps_more_of_the_bright_tail():
    """A4/F3: the acute lesion is the bright tail of TRACE, so a wider clip must preserve more of it."""
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, (4, 32, 32))
    x[0, :8, :8] = rng.normal(9.0, 1.0, (8, 8))  # synthetic "lesion" around +9 sigma
    fg = np.ones_like(x, bool)
    z6, z10 = robust_zscore(x, fg, clip=6.0), robust_zscore(x, fg, clip=10.0)
    les = np.zeros_like(x, bool)
    les[0, :8, :8] = True
    sat6 = float((z6[les] >= 6.0 - 1e-4).mean())
    sat10 = float((z10[les] >= 10.0 - 1e-4).mean())
    assert sat6 > 0.9 and sat10 < 0.3
    assert z10[les].std() > z6[les].std()  # intra-lesion contrast survives


def test_clip_mode_pct_is_per_subject_and_bounded():
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, (4, 32, 32))
    x[0, 0, 0] = 1e6
    fg = np.ones_like(x, bool)
    z = robust_zscore(x, fg, clip=0.1, clip_mode="pct")
    assert np.isfinite(z).all() and z.max() <= 40.0 and z.min() >= -40.0
    assert z.max() < 20.0  # the 1e6 outlier is cut at the p99.9 of the foreground
    with pytest.raises(ValueError):
        robust_zscore(x, fg, clip_mode="nope")


# ---------------------------------------------------------------- channel ablation
def _tiny_cache(tmp_path, n_sub=4, n_slices=6, size=16):
    rng = np.random.default_rng(0)
    ids = []
    for k in range(n_sub):
        sid = f"sub-{k + 1}"
        img = rng.normal(0, 1, (n_slices, 2, size, size)).astype(np.float16)
        mask = np.zeros((n_slices, size, size), np.uint8)
        mask[1:3, 4:8, 4:8] = 1
        np.savez(tmp_path / f"{sid}.npz", img=img, mask=mask, voxel_volume_mm3=20.0)
        ids.append(sid)
    return ids


def test_channel_subset_selects_the_requested_channels(tmp_path):
    ids = _tiny_cache(tmp_path)
    cache = SubjectCache(tmp_path, ids)
    full = SliceDataset(cache, neg_pos_ratio=None, augment=False, seed=3)
    trace = SliceDataset(cache, neg_pos_ratio=None, augment=False, seed=3, channels=[0])
    adc = SliceDataset(cache, neg_pos_ratio=None, augment=False, seed=3, channels=[1])
    assert full[0][0].shape[0] == 2 and trace[0][0].shape[0] == 1 and adc[0][0].shape[0] == 1
    assert torch.equal(trace[0][0][0], full[0][0][0])
    assert torch.equal(adc[0][0][0], full[0][0][1])
    # the augmentation RNG must stay item-keyed (and therefore worker independent) for 1-channel input
    a = SliceDataset(cache, neg_pos_ratio=None, augment=True, seed=3, channels=[0])
    assert torch.equal(a[2][0], a[2][0])


def test_train_runs_with_one_channel(tmp_path):
    ids = _tiny_cache(tmp_path, n_sub=6)
    splits = {"train": ids[:4], "val": ids[4:]}
    cfg = {"seed": 1, "threads": 1, "size": 16, "base_channels": 4, "depth": 2, "epochs": 1,
           "batch_size": 4, "lr": 1e-3, "min_lr": 1e-5, "warmup_frac": 0.1, "weight_decay": 0.0,
           "grad_clip": 1.0, "dice_weight": 1.0, "pos_weight": None, "neg_pos_ratio": 1.0,
           "augment": True, "threshold": 0.5, "num_workers": 0, "log_every": 1, "channels": [0]}
    best = train(cfg, tmp_path / "run", tmp_path, splits, device="cpu")
    assert "val_dice" in best
    start = json.loads(open(tmp_path / "run" / "log.jsonl").readline())
    assert start["in_channels"] == 1


# ---------------------------------------------------------------- amp helper / lr schedule
def test_amp_context_is_a_noop_on_cpu():
    with amp_context("bf16", "cpu"):
        assert not torch.is_autocast_enabled()
    with amp_context(None, "cpu"):
        assert not torch.is_autocast_enabled()


def test_lr_schedule_warms_up_then_decays_to_min_lr():
    total, warmup, base, mn = 1000, 50, 1e-3, 2e-5
    lrs = [cosine_warmup(s, total, warmup, base, mn) for s in range(total)]
    assert lrs[0] < base / 10 and np.isclose(lrs[warmup - 1], base)
    assert int(np.argmax(lrs)) == warmup - 1
    assert all(b <= a + 1e-12 for a, b in zip(lrs[warmup:], lrs[warmup + 1:]))  # monotone decay
    assert abs(lrs[-1] - mn) < 0.01 * base
    # a zero-length warm-up must not divide by zero
    assert np.isfinite(cosine_warmup(0, 100, 0, base, mn))
