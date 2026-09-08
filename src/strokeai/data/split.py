"""Patient-level, stratified, seeded train/val/test split. Slice-level splitting is forbidden (CLAUDE.md)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import sha256_of


def _flag(idx: pd.DataFrame, col: str) -> pd.Series:
    """Index columns round-trip through CSV as object dtype; NaN -> False, no silent-downcast warning."""
    return idx[col].isin([True, "True", 1, 1.0]).to_numpy(dtype=bool)


def eligible_for_segmentation(idx: pd.DataFrame) -> pd.DataFrame:
    """Subjects with TRACE + ADC + acute mask whose shapes/affines agree. Empty acute masks are kept
    (they are legitimate negatives) but flagged.

    NOTE (A3, 2026-09-09): the ADC *affine* is deliberately NOT part of this filter. One train subject
    (sub-235) has an ADC grid displaced ~6.6 mm from TRACE; adding the condition here would drop it and
    change `data/splits.json`'s sha256, which the charter pins. `preprocess_subject` resamples that ADC
    onto the TRACE grid instead, which fixes the data without moving anybody between splits.
    """
    ok = (_flag(idx, "has_trace") & _flag(idx, "has_adc") & _flag(idx, "has_mask_acute")
          & _flag(idx, "adc_shape_matches_trace")
          & _flag(idx, "mask_shape_matches_trace")
          & _flag(idx, "mask_affine_matches_trace")
          & _flag(idx, "trace_extra_dims_singleton"))
    return idx[ok].copy()


def stratum(row: pd.Series) -> str:
    size = "empty" if row["mask_acute_voxels"] == 0 else ("small" if row["mask_acute_ml"] < 2 else "large")
    return f"{row.get('etiology_code', 'n/a')}|{size}"


def make_split(df: pd.DataFrame, seed: int, frac_val: float = 0.15, frac_test: float = 0.15) -> dict:
    """Deterministic: subjects are sorted, then shuffled with a seeded Generator inside each stratum,
    then assigned by round-robin proportions. Returns dict with lists and a content hash."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["stratum"] = df.apply(stratum, axis=1)
    split = {"train": [], "val": [], "test": []}
    for _, g in df.sort_values("participant_id").groupby("stratum", sort=True):
        ids = sorted(g["participant_id"].tolist())
        rng.shuffle(ids)
        n = len(ids)
        n_test = int(round(n * frac_test))
        n_val = int(round(n * frac_val))
        split["test"] += ids[:n_test]
        split["val"] += ids[n_test:n_test + n_val]
        split["train"] += ids[n_test + n_val:]
    for k in split:
        split[k] = sorted(split[k], key=lambda s: int(s[4:]))
    assert not (set(split["train"]) & set(split["val"]))
    assert not (set(split["train"]) & set(split["test"]))
    assert not (set(split["val"]) & set(split["test"]))
    return {"seed": seed, "frac_val": frac_val, "frac_test": frac_test, "n": {k: len(v) for k, v in split.items()},
            "sha256": sha256_of(split), **split}
