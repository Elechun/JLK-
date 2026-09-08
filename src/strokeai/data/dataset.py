from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SubjectCache:
    """Loads all preprocessed subjects of a split into RAM (float16). ~1.6 GB for the train split at 128²."""

    def __init__(self, cache_dir: Path, subject_ids: list[str]):
        self.ids = list(subject_ids)
        self.img, self.mask, self.meta = {}, {}, {}
        for sid in self.ids:
            z = np.load(cache_dir / f"{sid}.npz")
            self.img[sid] = z["img"]
            self.mask[sid] = z["mask"]
            self.meta[sid] = {"voxel_volume_mm3": float(z["voxel_volume_mm3"])}

    def slice_table(self) -> list[tuple[str, int, bool]]:
        rows = []
        for sid in self.ids:
            pos = self.mask[sid].reshape(self.mask[sid].shape[0], -1).any(axis=1)
            rows += [(sid, i, bool(pos[i])) for i in range(len(pos))]
        return rows


class SliceDataset(Dataset):
    """2-D slices with optional positive/negative balancing (resampled every epoch via `resample`)."""

    def __init__(self, cache: SubjectCache, neg_pos_ratio: float | None = 1.0, augment: bool = False, seed: int = 0,
                 channels: list[int] | None = None):
        self.cache = cache
        self.table = cache.slice_table()
        self.neg_pos_ratio = neg_pos_ratio
        self.augment = augment
        # `channels` selects a subset of the cached input channels (0 = TRACE, 1 = ADC).  None = both.
        # Used for the ADC ablation (A4); the cache itself is never re-written.
        self.channels = list(channels) if channels is not None else None
        self.seed = int(seed)
        self.epoch = -1
        self.rng = np.random.default_rng(seed)
        self.resample()

    def resample(self) -> None:
        self.epoch += 1
        pos = [t for t in self.table if t[2]]
        neg = [t for t in self.table if not t[2]]
        if self.neg_pos_ratio is None or len(pos) == 0:
            self.items = list(self.table)
        else:
            k = min(len(neg), int(round(len(pos) * self.neg_pos_ratio)))
            idx = self.rng.choice(len(neg), size=k, replace=False)
            self.items = pos + [neg[i] for i in idx]
        self.rng.shuffle(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        sid, s, _ = self.items[i]
        x = self.cache.img[sid][s].astype(np.float32)  # (2, H, W)
        if self.channels is not None:
            x = x[self.channels]
        y = self.cache.mask[sid][s].astype(np.float32)[None]  # (1, H, W)
        if self.augment:
            # Per-item generator keyed by (seed, epoch, index): reproducible and independent of
            # DataLoader `num_workers` (a single self.rng would be *forked* into every worker, so each
            # worker would replay the same augmentation stream and the result would depend on worker count).
            rng = np.random.default_rng((self.seed, self.epoch, i))
            if rng.random() < 0.5:  # left-right flip (axis W)
                x = x[:, :, ::-1]
                y = y[:, :, ::-1]
            scale = rng.uniform(0.9, 1.1, size=(x.shape[0], 1, 1)).astype(np.float32)
            shift = rng.uniform(-0.1, 0.1, size=(x.shape[0], 1, 1)).astype(np.float32)
            x = x * scale + shift
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))
