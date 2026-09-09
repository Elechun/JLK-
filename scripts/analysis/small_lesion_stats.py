#!/usr/bin/env python3
"""A4b - how much of the training signal do small lesions actually get?  (train split only, never test)

Motivation for H2: `SliceDataset` balances positive vs negative slices but nothing else, so a subject
contributes as many positive slices as its lesion is tall.  This script measures the resulting
representation per lesion-size band and what the `slice_weight` options do to it (effective sample
size included, so an over-aggressive weight that collapses onto a handful of subjects is visible).

    PYTHONPATH=src .venv/bin/python scripts/analysis/small_lesion_stats.py

Writes results/small_lesion/slice_representation.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from strokeai.data.dataset import SliceDataset, SubjectCache  # noqa: E402

BANDS = [(0.0, 2.0), (2.0, 10.0), (10.0, 50.0), (50.0, np.inf)]


def band_of(ml: float) -> str:
    for a, b in BANDS:
        if a <= ml < b:
            return f"[{a:g},{b:g})"
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--out", type=Path, default=Path("results/small_lesion/slice_representation.json"))
    a = ap.parse_args()

    sp = json.load(open(a.splits))  # train only: the held-out split is never read here
    cache = SubjectCache(a.cache, sp["train"])
    ml = {sid: cache.lesion_volume_ml(sid) for sid in cache.ids}
    table = cache.slice_table()
    pos = [t for t in table if t[2]]

    out: dict = {"n_train_subjects": len(cache.ids), "n_slices": len(table), "n_positive_slices": len(pos),
                 "lesion_ml_quantiles": {str(q): float(np.quantile(list(ml.values()), q))
                                         for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)}}

    # ---- how many positive slices does each band own under the shipped sampler? -------------------
    per_band: dict = {}
    for a_, b_ in BANDS:
        key = f"[{a_:g},{b_:g})"
        subs = [s for s in cache.ids if a_ <= ml[s] < b_]
        slices = [t for t in pos if a_ <= ml[t[0]] < b_]
        per_band[key] = {"n_subjects": len(subs), "pos_slices": len(slices),
                         "pos_slices_per_subject": len(slices) / max(len(subs), 1),
                         "share_of_subjects_pct": 100 * len(subs) / len(cache.ids),
                         "share_of_pos_slices_pct": 100 * len(slices) / max(len(pos), 1)}
    out["bands_uniform_sampler"] = per_band

    # ---- what the weighting options do -------------------------------------------------------------
    variants = [("none", 1.0), ("inv_subject", 1.0), ("inv_volume", 0.5), ("inv_volume", 1.0),
                ("inv_area", 0.5), ("inv_area", 1.0)]
    out["weighted"] = {}
    for name, power in variants:
        ds = SliceDataset(cache, neg_pos_ratio=1.0, augment=False, seed=0, slice_weight=name,
                          slice_weight_power=power)
        w = np.full(len(pos), 1.0 / len(pos)) if name == "none" else ds._positive_weights(pos)
        shares = {f"[{a_:g},{b_:g})": float(w[[a_ <= ml[t[0]] < b_ for t in pos]].sum() * 100) for a_, b_ in BANDS}
        # effective sample size of a weighted draw of len(pos) items (Kish): 1 / sum(w^2) per draw
        ess = float(1.0 / np.square(w).sum())
        sub_w: dict[str, float] = {}
        for wi, (sid, _, _) in zip(w, pos):
            sub_w[sid] = sub_w.get(sid, 0.0) + float(wi)
        sw = np.array(list(sub_w.values()))
        out["weighted"][f"{name}^{power:g}"] = {
            "slice_weight": name, "power": power, "band_share_of_draws_pct": shares,
            "ess_slices": ess, "ess_frac_of_pos": ess / len(pos),
            "ess_subjects": float(1.0 / np.square(sw).sum()), "n_subjects": len(sw),
            "max_subject_share_pct": float(sw.max() * 100),
            "expected_draws_per_epoch_small": float(shares["[0,2)"] / 100 * len(pos)),
        }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
