"""Slice dataset over the preprocessed cache.

Cache layout (A5b 2026-09-09, verified against `preprocess_subject` + `nib.as_closest_canonical`):
  img  : (S, C, H, W)   mask : (S, H, W)
  S = canonical axis 2 (I -> S), H = canonical axis 0 (patient Left -> Right, RAS x),
  W = canonical axis 1 (Posterior -> Anterior, RAS y).
So in a (C, H, W) sample the LEFT-RIGHT axis is H (index 1) and the A-P axis is W (index 2).
The historical augmentation flipped W, i.e. it was an anterior-posterior mirror, not the
left-right mirror its comment claimed. `flip_axis` makes the choice explicit and configurable.

A4b 2026-09-09 added two options for the small-lesion (< 2 mL) follow-up.  Both default to the
historical behaviour, so `runs/seg_unet2d` stays reproducible:
  * `slice_weight`  -- lesion-size-aware resampling of the positive slices (H2).
  * `context`       -- 2.5-D input: neighbouring slices stacked as extra channels (H4).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# axis of a (C, H, W) slice sample
LR_AXIS = 1  # H : patient left -> right (RAS x)
AP_AXIS = 2  # W : posterior -> anterior (RAS y)
FLIP_AXES = {"lr": (LR_AXIS,), "ap": (AP_AXIS,), "both": (LR_AXIS, AP_AXIS), "none": ()}

# H2 (A4b): how positive slices are resampled each epoch.
#   "none"        -- every positive slice exactly once (the historical behaviour)
#   "inv_subject" -- w ∝ (number of positive slices of that subject)^-p : every *subject* contributes
#                    the same expected number of slices regardless of how tall its lesion is
#   "inv_volume"  -- w ∝ (subject lesion volume in mL)^-p : the physically motivated version
#   "inv_area"    -- w ∝ (lesion pixel count in THIS slice)^-p : per-slice, so a small cranial cap of a
#                    big lesion is also up-weighted
SLICE_WEIGHTS = ("none", "inv_subject", "inv_volume", "inv_area")

# H4 (A4b): which cached channels receive the +-context neighbours.
#   "all"   -- every channel is stacked  -> C * (2k+1) input channels (TRACE and ADC context)
#   "trace" -- only channel 0 is stacked -> (2k+1) + (C-1) channels (TRACE context + centre ADC)
CONTEXT_MODES = ("all", "trace")


def context_groups(n_base: int, context: int = 0, mode: str = "all") -> list[int]:
    """Sizes of the channel groups produced by `stack_context`, in output-channel order.

    A group = the planes that come from the same source channel.  The intensity augmentation draws one
    scale/shift per group so that the neighbouring slices of a channel keep a consistent intensity
    (drawing per plane would inject noise into exactly the context H4 is trying to add).
    With `context = 0` this is `[1] * n_base`, i.e. the historical per-channel draw.
    """
    if mode not in CONTEXT_MODES:
        raise ValueError(f"context_mode must be one of {sorted(CONTEXT_MODES)}, got {mode!r}")
    if context <= 0:
        return [1] * n_base
    k = 2 * context + 1
    if mode == "all":
        return [k] * n_base
    return [k] + [1] * (n_base - 1)


def context_channel_count(n_base: int, context: int = 0, mode: str = "all") -> int:
    """Number of model input channels for a (n_base-channel) cache under this 2.5-D setting."""
    return int(sum(context_groups(n_base, context, mode)))


def stack_context(vol: np.ndarray, s: int, context: int = 0, mode: str = "all",
                  channels: list[int] | None = None) -> np.ndarray:
    """(S, C, H, W) volume -> one (C', H, W) float32 sample centred on slice `s`.

    Slices outside the volume are replicate-padded (the first/last slice repeats), so the model never
    sees a zero plane it could use as a "you are at the edge" shortcut.
    Output order for mode "all", context 1, channels (TRACE, ADC):
        [TRACE t-1, TRACE t, TRACE t+1, ADC t-1, ADC t, ADC t+1]
    and for mode "trace":
        [TRACE t-1, TRACE t, TRACE t+1, ADC t]
    """
    chans = list(range(vol.shape[1])) if channels is None else list(channels)
    if context <= 0:
        return vol[s][chans].astype(np.float32)
    if mode not in CONTEXT_MODES:
        raise ValueError(f"context_mode must be one of {sorted(CONTEXT_MODES)}, got {mode!r}")
    n_s = vol.shape[0]
    idx = [min(max(s + o, 0), n_s - 1) for o in range(-context, context + 1)]  # replicate padding
    nb = vol[idx]  # (K, C, H, W)
    if mode == "all":
        out = np.moveaxis(nb[:, chans], 0, 1)  # (C, K, H, W)
        return out.reshape(-1, out.shape[2], out.shape[3]).astype(np.float32)
    parts = [nb[:, chans[0]]] + [vol[s][c][None] for c in chans[1:]]
    return np.concatenate(parts, axis=0).astype(np.float32)


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

    def slice_lesion_voxels(self, sid: str) -> np.ndarray:
        """Lesion voxel count per slice, shape (S,)."""
        m = self.mask[sid]
        return (m > 0).reshape(m.shape[0], -1).sum(axis=1).astype(np.float64)

    def lesion_volume_ml(self, sid: str) -> float:
        return float((self.mask[sid] > 0).sum() * self.meta[sid]["voxel_volume_mm3"] / 1000.0)


class SliceDataset(Dataset):
    """2-D slices with optional positive/negative balancing (resampled every epoch via `resample`)."""

    def __init__(self, cache: SubjectCache, neg_pos_ratio: float | None = 1.0, augment: bool = False, seed: int = 0,
                 channels: list[int] | None = None, flip_axis: str = "lr", slice_weight: str = "none",
                 slice_weight_power: float = 1.0, context: int = 0, context_mode: str = "all"):
        self.cache = cache
        self.table = cache.slice_table()
        self.neg_pos_ratio = neg_pos_ratio
        self.augment = augment
        # Mirror augmentation: "lr" = anatomical left-right (the intended one), "ap" = anterior-posterior
        # (what runs/seg_unet2d was actually trained with), "both", or "none".  Each axis is flipped
        # independently with p = 0.5.
        if flip_axis not in FLIP_AXES:
            raise ValueError(f"flip_axis must be one of {sorted(FLIP_AXES)}, got {flip_axis!r}")
        self.flip_axis = flip_axis
        # `channels` selects a subset of the cached input channels (0 = TRACE, 1 = ADC).  None = both.
        # Used for the ADC ablation (A4); the cache itself is never re-written.
        self.channels = list(channels) if channels is not None else None
        # H2: lesion-size-aware sampling of the positive slices (see SLICE_WEIGHTS).
        if slice_weight not in SLICE_WEIGHTS:
            raise ValueError(f"slice_weight must be one of {sorted(SLICE_WEIGHTS)}, got {slice_weight!r}")
        self.slice_weight = slice_weight
        self.slice_weight_power = float(slice_weight_power)
        # H4: 2.5-D context.  `context` = number of neighbours on EACH side (0 = plain 2-D).
        self.context = int(context)
        if context_mode not in CONTEXT_MODES:
            raise ValueError(f"context_mode must be one of {sorted(CONTEXT_MODES)}, got {context_mode!r}")
        self.context_mode = context_mode
        n_base = len(self.channels) if self.channels is not None else int(cache.img[cache.ids[0]].shape[1])
        self.groups = context_groups(n_base, self.context, self.context_mode)
        self.in_channels = int(sum(self.groups))
        self.seed = int(seed)
        self.epoch = -1
        self.rng = np.random.default_rng(seed)
        self._weight_cache: dict[str, np.ndarray] = {}
        self.resample()

    # ------------------------------------------------------------------ H2: positive-slice weighting
    def _positive_weights(self, pos: list[tuple[str, int, bool]]) -> np.ndarray:
        """Sampling probability of each positive slice; the small lesions get the mass."""
        p = self.slice_weight_power
        if self.slice_weight == "inv_subject":
            n_pos = {}
            for sid, _, _ in pos:
                n_pos[sid] = n_pos.get(sid, 0) + 1
            raw = np.array([n_pos[sid] for sid, _, _ in pos], float)
        elif self.slice_weight == "inv_volume":
            ml = {sid: self.cache.lesion_volume_ml(sid) for sid in {t[0] for t in pos}}
            raw = np.array([ml[sid] for sid, _, _ in pos], float)
        else:  # inv_area
            counts = {sid: self.cache.slice_lesion_voxels(sid) for sid in {t[0] for t in pos}}
            raw = np.array([counts[sid][s] for sid, s, _ in pos], float)
        raw = np.maximum(raw, 1e-6)
        w = raw ** (-p)
        return w / w.sum()

    def resample(self) -> None:
        self.epoch += 1
        pos = [t for t in self.table if t[2]]
        neg = [t for t in self.table if not t[2]]
        if self.slice_weight != "none" and len(pos) > 0:
            # Draw the SAME number of positive slices, with replacement, proportional to the weights:
            # the epoch length (and therefore the LR schedule) is unchanged, only the composition moves.
            w = self._positive_weights(pos)
            pos = [pos[i] for i in self.rng.choice(len(pos), size=len(pos), replace=True, p=w)]
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
        x = stack_context(self.cache.img[sid], s, self.context, self.context_mode, self.channels)  # (C', H, W)
        y = self.cache.mask[sid][s].astype(np.float32)[None]  # (1, H, W)
        if self.augment:
            # Per-item generator keyed by (seed, epoch, index): reproducible and independent of
            # DataLoader `num_workers` (a single self.rng would be *forked* into every worker, so each
            # worker would replay the same augmentation stream and the result would depend on worker count).
            rng = np.random.default_rng((self.seed, self.epoch, i))
            for ax in FLIP_AXES[self.flip_axis]:
                if rng.random() < 0.5:
                    x = np.flip(x, axis=ax)
                    y = np.flip(y, axis=ax)
            # one draw per source channel, broadcast over its context planes (identity when context = 0)
            scale = rng.uniform(0.9, 1.1, size=(len(self.groups), 1, 1)).astype(np.float32)
            shift = rng.uniform(-0.1, 0.1, size=(len(self.groups), 1, 1)).astype(np.float32)
            if self.context > 0:
                scale = np.repeat(scale, self.groups, axis=0)
                shift = np.repeat(shift, self.groups, axis=0)
            x = x * scale + shift
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))
