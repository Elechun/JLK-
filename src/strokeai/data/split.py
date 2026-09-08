"""Patient-level, stratified, seeded train/val/test split. Slice-level splitting is forbidden (CLAUDE.md)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import sha256_of


def eligible_for_segmentation(idx: pd.DataFrame) -> pd.DataFrame:
    """Subjects with TRACE + ADC + acute mask whose shapes/affines agree. Empty acute masks are kept
    (they are legitimate negatives) but flagged."""
    ok = (idx["has_trace"] & idx["has_adc"] & idx["has_mask_acute"]
          & idx["adc_shape_matches_trace"].fillna(False).astype(bool)
          & idx["mask_shape_matches_trace"].fillna(False).astype(bool)
          & idx["mask_affine_matches_trace"].fillna(False).astype(bool)
          & idx["trace_extra_dims_singleton"].fillna(False).astype(bool))
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
