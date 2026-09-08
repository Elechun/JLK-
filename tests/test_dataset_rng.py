"""A3 regression test: slice sampling / augmentation must be reproducible and DataLoader-worker independent."""
import numpy as np

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
    """Augmentation must not desynchronise the label from the image."""
    cache = _make_cache(tmp_path)
    ds = SliceDataset(cache, neg_pos_ratio=None, augment=True, seed=3)
    for i in range(len(ds)):
        sid, s, _ = ds.items[i]
        x, y = ds[i]
        ref = cache.mask[sid][s]
        got = y.numpy()[0]
        assert np.array_equal(got, ref) or np.array_equal(got, ref[:, ::-1])
