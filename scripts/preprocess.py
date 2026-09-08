#!/usr/bin/env python3
"""Preprocess subjects listed in data/splits.json into data/cache/<sid>.npz (parallel, idempotent).

By default only `train` and `val` are preprocessed: CLAUDE.md forbids *any* script from reading the
test split before the single final evaluation. Pass `--which test` (or `--which train val test`) at
that point to materialise the test cache.
"""
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
        return sid, "cached", {}
    try:
        r = preprocess_subject(raw / sid / "dwi" / f"{sid}_rec-TRACE_dwi.nii.gz",
                               raw / sid / "dwi" / f"{sid}_rec-ADC_dwi.nii.gz",
                               raw / "derivatives" / "lesion_masks" / sid / "dwi" / f"{sid}_space-TRACE_desc-lesionAcute_mask.nii.gz",
                               size)
        flags = r.pop("flags", {})  # kept out of the .npz so the cache stays byte-reproducible
        np.savez(dest, **r)
        return sid, "ok", flags
    except Exception as e:  # noqa: BLE001
        return sid, f"ERROR {e}", {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/ds004889"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--out", type=Path, default=Path("data/cache"))
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--which", nargs="+", default=["train", "val"], choices=["train", "val", "test"],
                    help="splits to preprocess; 'test' only at the final evaluation step (CLAUDE.md)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sp = json.load(open(a.splits))
    ids = [s for w in a.which for s in sp[w]]
    errs, flagged = [], {}
    with cf.ProcessPoolExecutor(a.workers) as ex:
        for i, (sid, st, fl) in enumerate(ex.map(one, [(s, a.raw, a.out, a.size) for s in ids], chunksize=8)):
            if st.startswith("ERROR"):
                errs.append((sid, st))
            interesting = {k: v for k, v in fl.items() if k != "mask_max_value" or v != 1.0}
            if interesting:
                flagged[sid] = interesting
            if (i + 1) % 200 == 0:
                print(f"[{i+1}/{len(ids)}] errors={len(errs)}", flush=True)
    manifest_p = a.out / "_manifest.json"
    prev = json.load(open(manifest_p)) if manifest_p.exists() else {}
    done = sorted(set(prev.get("which", [])) | set(a.which))
    flags = {**prev.get("flags", {}), **flagged}
    counts = {}
    for f in flags.values():
        for k in f:
            counts[k] = counts.get(k, 0) + 1
    json.dump({"size": a.size, "which": done, "n": len(ids), "n_total_cached": len(list(a.out.glob("sub-*.npz"))),
               "errors": errs, "flag_counts": counts, "flags": flags}, open(manifest_p, "w"), indent=1)
    print("done, errors:", errs[:10], len(errs), "| flag counts:", counts)


if __name__ == "__main__":
    main()
