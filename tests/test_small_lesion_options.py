"""A4b regression tests for the small-lesion (< 2 mL) options added on 2026-09-09.

Covered: the lesion-size-aware slice sampler (H2), the 2.5-D context stacking (H4), the
Tversky / inverse-size Dice region losses (H5), and -- most important -- that every one of them
DEFAULTS to the historical behaviour, because `runs/seg_unet2d` must stay reproducible.
"""
import json

import numpy as np
import pytest
import torch

from strokeai.data.dataset import (SliceDataset, SubjectCache, context_channel_count, context_groups,
                                   stack_context)
from strokeai.losses import bce_dice_loss, soft_dice_loss, tversky_loss
from strokeai.train import predict_subject, train
from strokeai.models import UNet2D


def _cache(tmp_path, sizes=(1, 4, 12), n_slices=8, size=16):
    """One subject per entry of `sizes`: `sizes[k]` positive slices, each with a k-dependent lesion area."""
    rng = np.random.default_rng(0)
    ids = []
    for k, n_pos in enumerate(sizes):
        sid = f"sub-{k + 1}"
        img = rng.normal(0, 1, (n_slices, 2, size, size)).astype(np.float16)
        img[:, 0] += np.arange(n_slices, dtype=np.float16)[:, None, None]  # slice-identifiable TRACE
        mask = np.zeros((n_slices, size, size), np.uint8)
        side = 1 + k  # bigger index -> bigger lesion per slice as well as more slices
        mask[:n_pos, :side, :side] = 1
        np.savez(tmp_path / f"{sid}.npz", img=img, mask=mask, voxel_volume_mm3=1000.0)
        ids.append(sid)
    return ids


# ---------------------------------------------------------------- H2: slice weighting
def test_default_sampler_is_unchanged_and_takes_every_positive_slice_once(tmp_path):
    cache = SubjectCache(tmp_path, _cache(tmp_path))
    a = SliceDataset(cache, neg_pos_ratio=1.0, augment=False, seed=5)
    b = SliceDataset(cache, neg_pos_ratio=1.0, augment=False, seed=5, slice_weight="none")
    assert a.items == b.items  # the default must not consume any extra RNG draw
    pos = [t for t in cache.slice_table() if t[2]]
    assert sorted(t for t in a.items if t[2]) == sorted(pos)


def test_inv_subject_weighting_equalises_subjects_and_keeps_the_epoch_length(tmp_path):
    ids = _cache(tmp_path, sizes=(1, 4, 12))
    cache = SubjectCache(tmp_path, ids)
    plain = SliceDataset(cache, neg_pos_ratio=1.0, augment=False, seed=1)
    weighted = SliceDataset(cache, neg_pos_ratio=1.0, augment=False, seed=1, slice_weight="inv_subject")
    assert len(plain) == len(weighted)  # same steps/epoch -> the LR schedule is untouched
    pos = [t for t in cache.slice_table() if t[2]]
    w = weighted._positive_weights(pos)
    share = {sid: float(w[[t[0] == sid for t in pos]].sum()) for sid in ids}
    assert all(abs(v - 1 / len(ids)) < 1e-9 for v in share.values())
    # and the smallest subject really is drawn more often than under the uniform sampler
    n_small = sum(1 for t in weighted.items if t[2] and t[0] == ids[0])
    assert n_small > sum(1 for t in plain.items if t[2] and t[0] == ids[0])


def test_inv_volume_weighting_shifts_mass_to_the_small_lesions(tmp_path):
    ids = _cache(tmp_path, sizes=(1, 4, 12))
    cache = SubjectCache(tmp_path, ids)
    pos = [t for t in cache.slice_table() if t[2]]
    ml = {sid: cache.lesion_volume_ml(sid) for sid in ids}
    assert ml[ids[0]] < ml[ids[1]] < ml[ids[2]]
    prev = None
    for power in (0.0, 0.5, 1.0):
        ds = SliceDataset(cache, neg_pos_ratio=1.0, augment=False, seed=2, slice_weight="inv_volume",
                          slice_weight_power=power)
        w = ds._positive_weights(pos)
        small = float(w[[t[0] == ids[0] for t in pos]].sum())
        if prev is not None:
            assert small > prev  # a stronger exponent gives the smallest lesion strictly more mass
        prev = small
    ds = SliceDataset(cache, neg_pos_ratio=1.0, augment=False, seed=2, slice_weight="inv_area")
    w = ds._positive_weights(pos)
    assert np.isclose(w.sum(), 1.0) and (w > 0).all()
    with pytest.raises(ValueError):
        SliceDataset(cache, slice_weight="nope")


# ---------------------------------------------------------------- H4: 2.5-D context
def test_context_channel_bookkeeping():
    assert context_channel_count(2, 0) == 2 and context_groups(2, 0) == [1, 1]
    assert context_channel_count(2, 1, "all") == 6 and context_groups(2, 1, "all") == [3, 3]
    assert context_channel_count(2, 1, "trace") == 4 and context_groups(2, 1, "trace") == [3, 1]
    assert context_channel_count(2, 2, "all") == 10
    with pytest.raises(ValueError):
        context_channel_count(2, 1, "nope")


def test_stack_context_order_and_replicate_padding(tmp_path):
    cache = SubjectCache(tmp_path, _cache(tmp_path, n_slices=5))
    vol = cache.img["sub-1"]
    mid = stack_context(vol, 2, context=1, mode="all")
    assert mid.shape[0] == 6
    assert np.array_equal(mid[0], vol[1, 0]) and np.array_equal(mid[1], vol[2, 0]) and np.array_equal(mid[2], vol[3, 0])
    assert np.array_equal(mid[3], vol[1, 1]) and np.array_equal(mid[4], vol[2, 1]) and np.array_equal(mid[5], vol[3, 1])
    tr = stack_context(vol, 2, context=1, mode="trace")
    assert tr.shape[0] == 4 and np.array_equal(tr[3], vol[2, 1])  # ADC keeps the centre slice only
    # replicate padding at both ends: the out-of-volume neighbour repeats the edge slice
    lo = stack_context(vol, 0, context=1, mode="all")
    hi = stack_context(vol, 4, context=1, mode="all")
    assert np.array_equal(lo[0], vol[0, 0]) and np.array_equal(lo[1], vol[0, 0])
    assert np.array_equal(hi[1], vol[4, 0]) and np.array_equal(hi[2], vol[4, 0])
    # context = 0 reproduces the plain 2-D sample exactly
    assert np.array_equal(stack_context(vol, 2, context=0), vol[2].astype(np.float32))
    assert np.array_equal(stack_context(vol, 2, context=1, mode="all", channels=[1])[1], vol[2, 1])


def test_context_dataset_and_augmentation_shape(tmp_path):
    cache = SubjectCache(tmp_path, _cache(tmp_path))
    ds = SliceDataset(cache, neg_pos_ratio=None, augment=True, seed=3, context=1, context_mode="all",
                      flip_axis="none")
    x, y = ds[0]
    assert x.shape[0] == 6 and y.shape[0] == 1 and ds.in_channels == 6
    assert torch.equal(ds[0][0], ds[0][0])  # item-keyed RNG -> worker independent
    # the intensity jitter is drawn per SOURCE channel, so the three TRACE planes share scale/shift:
    # differences between neighbouring planes are preserved up to the common scale factor.
    plain = SliceDataset(cache, neg_pos_ratio=None, augment=False, seed=3, context=1, context_mode="all",
                         flip_axis="none")
    xa, x0 = ds[0][0].numpy(), plain[0][0].numpy()
    r = (xa[1] - xa[0]) / (x0[1] - x0[0] + 1e-9)
    assert np.allclose(r, r.mean(), atol=1e-3)


def test_predict_subject_uses_the_same_context_stack(tmp_path):
    cache = SubjectCache(tmp_path, _cache(tmp_path, n_slices=5))
    torch.manual_seed(0)
    model = UNet2D(in_ch=4, base=4, depth=2)
    prob = predict_subject(model, cache.img["sub-1"], batch=2, context=1, context_mode="trace")
    assert prob.shape == (5, 16, 16) and np.isfinite(prob).all()
    with torch.no_grad():
        x = torch.from_numpy(np.stack([stack_context(cache.img["sub-1"], s, 1, "trace") for s in range(5)]))
        ref = torch.sigmoid(model(x))[:, 0].numpy()
    assert np.allclose(prob, ref, atol=1e-6)


# ---------------------------------------------------------------- H5: region losses
def test_tversky_at_half_half_equals_soft_dice():
    torch.manual_seed(0)
    logits = torch.randn(4, 1, 8, 8)
    target = (torch.rand(4, 1, 8, 8) > 0.7).float()
    for red in ("batch", "sample"):
        d = soft_dice_loss(logits, target, reduction=red)
        t = tversky_loss(logits, target, alpha=0.5, beta=0.5, reduction=red)
        assert torch.allclose(d, t, atol=1e-6)


def test_tversky_alpha_penalises_false_negatives_harder():
    logits = torch.full((1, 1, 8, 8), -4.0)  # confident background -> pure false negatives
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, :4] = 1.0
    fn_heavy = tversky_loss(logits, target, alpha=0.7, beta=0.3)
    fp_heavy = tversky_loss(logits, target, alpha=0.3, beta=0.7)
    assert fn_heavy > fp_heavy
    logits_fp = torch.full((1, 1, 8, 8), 4.0)  # confident foreground on an empty target -> pure FP
    empty = torch.zeros(1, 1, 8, 8)
    assert tversky_loss(logits_fp, empty, alpha=0.7, beta=0.3) < tversky_loss(logits_fp, empty, alpha=0.3, beta=0.7)
    # focal gamma keeps the loss in [0, 1] and shrinks it for gamma > 1
    assert 0.0 <= float(tversky_loss(logits, target, gamma=1.33)) <= 1.0
    assert tversky_loss(logits, target, gamma=1.33) < fn_heavy


def test_sample_invsize_gives_the_small_slice_more_weight():
    """Two slices, both predicted background; the loss must move more when the SMALL lesion is missed."""
    target = torch.zeros(2, 1, 16, 16)
    target[0, 0, :2, :2] = 1.0    # 4 px
    target[1, 0, :12, :12] = 1.0  # 144 px
    base = torch.full((2, 1, 16, 16), -4.0)
    good_small = base.clone()
    good_small[0, 0, :2, :2] = 4.0
    good_big = base.clone()
    good_big[1, 0, :12, :12] = 4.0
    plain_s = soft_dice_loss(good_small, target, reduction="sample")
    plain_b = soft_dice_loss(good_big, target, reduction="sample")
    inv_s = soft_dice_loss(good_small, target, reduction="sample_invsize", invsize_power=0.5)
    inv_b = soft_dice_loss(good_big, target, reduction="sample_invsize", invsize_power=0.5)
    # the inverse-size weighting moves the balance towards the small lesion: segmenting it correctly is
    # worth strictly more (relative to segmenting the big one) than it is under the plain "sample" mean
    assert (inv_s - inv_b) < (plain_s - plain_b)
    # empty slices inherit the mean positive weight, so the weights still sum to 1 and stay finite
    with_empty = torch.zeros(3, 1, 16, 16)
    with_empty[:2] = target
    v = soft_dice_loss(torch.cat([base, base[:1]]), with_empty, reduction="sample_invsize")
    assert torch.isfinite(v) and 0.0 <= float(v) <= 1.0
    assert torch.isfinite(soft_dice_loss(base, torch.zeros(2, 1, 16, 16), reduction="sample_invsize"))


def test_bce_dice_loss_defaults_are_unchanged():
    torch.manual_seed(1)
    logits = torch.randn(4, 1, 8, 8)
    target = (torch.rand(4, 1, 8, 8) > 0.7).float()
    a, pa = bce_dice_loss(logits, target)
    b, pb = bce_dice_loss(logits, target, region_loss="dice", dice_reduction="batch")
    assert torch.equal(a, b) and pa == pb
    t, pt = bce_dice_loss(logits, target, region_loss="tversky", tversky_alpha=0.5, tversky_beta=0.5)
    assert torch.allclose(a, t, atol=1e-6) and set(pt) == {"bce", "dice"}
    with pytest.raises(ValueError):
        bce_dice_loss(logits, target, region_loss="nope")


# ---------------------------------------------------------------- end to end
@pytest.mark.parametrize("extra,in_ch", [
    ({}, 2),
    ({"slice_weight": "inv_volume", "slice_weight_power": 0.5}, 2),
    ({"context": 1, "context_mode": "trace"}, 4),
    ({"context": 1, "context_mode": "all"}, 6),
    ({"region_loss": "tversky", "tversky_alpha": 0.7, "tversky_beta": 0.3}, 2),
    ({"dice_reduction": "sample_invsize"}, 2),
])
def test_train_runs_with_each_small_lesion_option(tmp_path, extra, in_ch):
    ids = _cache(tmp_path, sizes=(1, 3, 6, 9, 12, 2))
    splits = {"train": ids[:4], "val": ids[4:]}
    cfg = {"seed": 1, "threads": 1, "size": 16, "base_channels": 4, "depth": 2, "epochs": 1,
           "batch_size": 4, "lr": 1e-3, "min_lr": 1e-5, "warmup_frac": 0.1, "weight_decay": 0.0,
           "grad_clip": 1.0, "dice_weight": 1.0, "pos_weight": None, "neg_pos_ratio": 1.0,
           "augment": True, "threshold": 0.5, "num_workers": 0, "log_every": 1, **extra}
    best = train(cfg, tmp_path / "run", tmp_path, splits, device="cpu")
    assert np.isfinite(best["val_dice"])
    start = json.loads(open(tmp_path / "run" / "log.jsonl").readline())
    assert start["in_channels"] == in_ch
