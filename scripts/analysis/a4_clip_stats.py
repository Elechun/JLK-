#!/usr/bin/env python3
"""A4: how much lesion signal does the +-6 MAD clip destroy, and where would a wider clip sit?

Recomputes the *unclipped* robust z-score from the raw NIfTIs for a random sample of TRAIN subjects
(never test) and reports the z distribution inside / outside the lesion for TRACE and ADC.

    PYTHONPATH=src .venv/bin/python scripts/analysis/a4_clip_stats.py --n 150
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from strokeai.data.preprocess import foreground, load_canonical, robust_stats  # noqa: E402


def one(sid: str, raw: Path) -> dict | None:
    d = raw / sid / "dwi"
    tr, _, _ = load_canonical(d / f"{sid}_rec-TRACE_dwi.nii.gz")
    ad, _, _ = load_canonical(d / f"{sid}_rec-ADC_dwi.nii.gz")
    mk, _, _ = load_canonical(raw / "derivatives" / "lesion_masks" / sid / "dwi" / f"{sid}_space-TRACE_desc-lesionAcute_mask.nii.gz")
    if tr.shape != mk.shape or ad.shape != tr.shape:
        return None
    fg = foreground(tr)
    if fg.sum() < 100:
        return None
    out = {"sid": sid}
    for name, vol in (("trace", tr), ("adc", ad)):
        med, sc, _ = robust_stats(vol[fg])
        z = (vol - med) / (sc + 1e-6)
        les = z[mk > 0]
        bg = z[fg & (mk == 0)]
        if les.size == 0:
            return None
        out[name] = {
            "lesion_p50": float(np.median(les)), "lesion_p90": float(np.percentile(les, 90)),
            "lesion_p99": float(np.percentile(les, 99)), "lesion_max": float(les.max()),
            "frac_les_gt6": float((les > 6).mean()), "frac_les_gt10": float((les > 10).mean()),
            "frac_les_gt14": float((les > 14).mean()), "frac_les_lt_m6": float((les < -6).mean()),
            "frac_les_lt_m10": float((les < -10).mean()),
            "bg_p50": float(np.median(bg)), "bg_p99": float(np.percentile(bg, 99)),
            "bg_p999": float(np.percentile(bg, 99.9)), "bg_max": float(bg.max()),
            "frac_bg_gt6": float((bg > 6).mean()), "frac_bg_gt10": float((bg > 10).mean()),
            "fg_p995_z": float(np.percentile(z[fg], 99.5)), "fg_p05_z": float(np.percentile(z[fg], 0.5)),
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/ds004889"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--which", nargs="+", default=["train"], choices=["train", "val"])  # never test
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", type=Path, default=Path("results/a4"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sp = json.load(open(a.splits))
    ids = [s for w in a.which for s in sp[w]]
    rng = np.random.default_rng(2026)
    ids = [ids[i] for i in rng.choice(len(ids), size=min(a.n, len(ids)), replace=False)]
    rows = []
    for i, sid in enumerate(ids):
        try:
            r = one(sid, a.raw)
        except Exception as e:  # noqa: BLE001
            print("skip", sid, e)
            continue
        if r:
            rows.append(r)
        if (i + 1) % 25 == 0:
            print(f"[{i+1}/{len(ids)}]", flush=True)
    summ = {"n_subjects": len(rows), "which": a.which}
    for name in ("trace", "adc"):
        keys = rows[0][name].keys()
        summ[name] = {k: {"mean": float(np.mean([r[name][k] for r in rows])),
                          "p50": float(np.median([r[name][k] for r in rows])),
                          "p90": float(np.percentile([r[name][k] for r in rows], 90))} for k in keys}
    json.dump({"summary": summ, "per_subject": rows}, open(a.out / "a4_clip_stats.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
