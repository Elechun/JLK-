#!/usr/bin/env python3
"""Preprocess every subject in data/splits.json into data/cache/<sid>.npz (parallel, idempotent)."""
import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strokeai.data.preprocess import preprocess_subject  # noqa: E402


def one(args):
    sid, raw, out, size = args
    dest = out / f"{sid}.npz"
    if dest.exists():
        return sid, "cached"
    try:
        r = preprocess_subject(raw / sid / "dwi" / f"{sid}_rec-TRACE_dwi.nii.gz",
                               raw / sid / "dwi" / f"{sid}_rec-ADC_dwi.nii.gz",
                               raw / "derivatives" / "lesion_masks" / sid / "dwi" / f"{sid}_space-TRACE_desc-lesionAcute_mask.nii.gz",
                               size)
        np.savez(dest, **r)
        return sid, "ok"
    except Exception as e:  # noqa: BLE001
        return sid, f"ERROR {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/ds004889"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--out", type=Path, default=Path("data/cache"))
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sp = json.load(open(a.splits))
    ids = sp["train"] + sp["val"] + sp["test"]
    errs = []
    with cf.ProcessPoolExecutor(a.workers) as ex:
        for i, (sid, st) in enumerate(ex.map(one, [(s, a.raw, a.out, a.size) for s in ids], chunksize=8)):
            if st.startswith("ERROR"):
                errs.append((sid, st))
            if (i + 1) % 200 == 0:
                print(f"[{i+1}/{len(ids)}] errors={len(errs)}", flush=True)
    json.dump({"size": a.size, "n": len(ids), "errors": errs}, open(a.out / "_manifest.json", "w"), indent=1)
    print("done, errors:", errs[:10], len(errs))


if __name__ == "__main__":
    main()
