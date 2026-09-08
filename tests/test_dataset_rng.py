"""A3 regression test: slice sampling / augmentation must be reproducible and DataLoader-worker independent."""
import numpy as np
import pytest

from strokeai.data.dataset import SliceDataset, SubjectCache


def _make_cache(tmp_path, n_sub=3, n_slices=6, size=16):
    rng = np.random.default_rng(0)
    ids = []
    for k in range(n_sub):
        sid = f"sub-{k + 1}"
        img = rng.normal(0, 1, (n_slices, 2, size, size)).astype(np.float16)
        mask = np.zeros((n_slices, size, size), np.uint8)
        mask[1:3, 4:8, 4:8] = 1
        np.savez(tmp_path / f"{sid}.npz", img=img, mask=mask, voxel_volume_mm3=20.0)
        ids.append(sid)
    return SubjectCache(tmp_path, ids)


def test_slice_table_is_patient_level(tmp_path):
    cache = _make_cache(tmp_path)
    rows = cache.slice_table()
    assert len(rows) == 3 * 6
    assert {sid for sid, _, _ in rows} == set(cache.ids)
    assert sum(int(p) for _, _, p in rows) == 3 * 2  # two positive slices per subject


def test_augmentation_is_reproducible_and_order_independent(tmp_path):
    cache = _make_cache(tmp_path)
    ds = SliceDataset(cache, neg_pos_ratio=1.0, augment=True, seed=11)
    forward = [ds[i][0].numpy().copy() for i in range(len(ds))]
    backward = [ds[i][0].numpy().copy() for i in reversed(range(len(ds)))][::-1]
    # fetching in a different order (what several DataLoader workers do) must give the same samples
    assert all(np.array_equal(a, b) for a, b in zip(forward, backward))
    again = [ds[i][0].numpy().copy() for i in range(len(ds))]
    assert all(np.array_equal(a, b) for a, b in zip(forward, again))


def test_augmentation_changes_between_epochs(tmp_path):
    cache = _make_cache(tmp_path)
    ds = SliceDataset(cache, neg_pos_ratio=None, augment=True, seed=11)
    first = [ds[i][0].numpy().copy() for i in range(len(ds))]
    ds.resample()
    second = [ds[i][0].numpy().copy() for i in range(len(ds))]
    assert not all(np.array_equal(a, b) for a, b in zip(first, second))


def test_same_seed_gives_the_same_dataset(tmp_path):
    cache = _make_cache(tmp_path)
    a = SliceDataset(cache, neg_pos_ratio=1.0, augment=True, seed=5)
    b = SliceDataset(cache, neg_pos_ratio=1.0, augment=True, seed=5)
    assert a.items == b.items
    assert all(np.array_equal(a[i][0].numpy(), b[i][0].numpy()) for i in range(len(a)))
    c = SliceDataset(cache, neg_pos_ratio=1.0, augment=True, seed=6)
    assert c.items != a.items


def test_mask_and_image_are_flipped_together(tmp_path):
    """Augmentation must not desynchronise the label from the image (whatever the mirror axis)."""
    cache = _make_cache(tmp_path)
    for axis, flips in [("lr", [lambda m: m[::-1, :]]), ("ap", [lambda m: m[:, ::-1]]),
                        ("both", [lambda m: m[::-1, :], lambda m: m[:, ::-1], lambda m: m[::-1, ::-1]])]:
        ds = SliceDataset(cache, neg_pos_ratio=None, augment=True, seed=3, flip_axis=axis)
        for i in range(len(ds)):
            sid, s, _ = ds.items[i]
            x, y = ds[i]
            ref = cache.mask[sid][s]
            got = y.numpy()[0]
            assert np.array_equal(got, ref) or any(np.array_equal(got, f(ref)) for f in flips)


def test_flip_axis_lr_mirrors_the_left_right_axis_and_ap_the_old_one(tmp_path):
    """A5b 2026-09-09: the cache is (S, C, H, W) with H = patient left-right.  The old code flipped W
    (anterior-posterior) while calling it 'left-right'.  `flip_axis="lr"` must flip H, `"ap"` must
    reproduce the old W flip bit-for-bit, and `"none"` must never flip."""
    cache = _make_cache(tmp_path)
    sid = cache.ids[0]
    cache.mask[sid][:] = 0
    cache.mask[sid][:, 0, 3] = 1          # single column at H = 0, W = 3
    for axis, expect in [("lr", [(0, 3), (15, 3)]), ("ap", [(0, 3), (0, 12)]), ("none", [(0, 3)])]:
        ds = SliceDataset(cache, neg_pos_ratio=None, augment=True, seed=5, flip_axis=axis)
        seen = set()
        for i in range(len(ds)):
            s, sl, _ = ds.items[i]
            if s != sid:
                continue
            y = ds[i][1].numpy()[0]
            hs, ws = np.nonzero(y)
            seen.add((int(hs[0]), int(ws[0])))
        assert seen == set(expect), (axis, seen)
    with pytest.raises(ValueError):
        SliceDataset(cache, augment=True, flip_axis="diagonal")
