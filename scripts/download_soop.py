#!/usr/bin/env python3
"""Download the SOOP (OpenNeuro ds004889) DWI subset straight from the public S3 bucket.

Only the files this project needs are fetched:
  * sub-*/dwi/sub-*_rec-TRACE_dwi.nii.gz (+ .json)   b=1000 trace DWI
  * sub-*/dwi/sub-*_rec-ADC_dwi.nii.gz   (+ .json)   ADC map
  * derivatives/lesion_masks/**                        acute / chronic / combined lesion masks
  * participants.tsv / participants.json / README.md / dataset_description.json

Total ≈ 2.8 GB (T1w/FLAIR are skipped: ≈ 69 GB and not used here).

Usage:
    python scripts/download_soop.py --out data/raw/ds004889 [--workers 16] [--limit N]

The script is idempotent: files whose on-disk size matches the S3 listing are skipped.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import re
import sys
import time
import urllib.request
from pathlib import Path

BUCKET = "https://s3.amazonaws.com/openneuro.org"
DATASET = "ds004889"

WANTED = re.compile(
    r"^ds004889/("
    r"participants\.(tsv|json)|README\.md|CHANGES|dataset_description\.json|dwi\.b(val|vec)"
    r"|sub-\d+/dwi/sub-\d+_rec-(TRACE|ADC)_dwi\.(nii\.gz|json)"
    r"|derivatives/lesion_masks/sub-\d+/dwi/.*\.nii\.gz"
    r")$"
)


def list_keys() -> list[tuple[str, int]]:
    """Paginate the S3 ListObjects (v1) API and return (key, size) pairs."""
    keys: list[tuple[str, int]] = []
    marker = None
    while True:
        url = f"{BUCKET}/?prefix={DATASET}/&max-keys=1000" + (f"&marker={marker}" if marker else "")
        xml = urllib.request.urlopen(url, timeout=120).read().decode()
        page = re.findall(r"<Key>(.*?)</Key><LastModified>.*?</LastModified><ETag>.*?</ETag><Size>(\d+)</Size>", xml)
        keys += [(k, int(s)) for k, s in page]
        if "<IsTruncated>true" not in xml:
            return keys
        marker = page[-1][0]


def fetch(key: str, size: int, out: Path) -> tuple[str, bool, str]:
    dest = out / key[len(DATASET) + 1 :]
    if dest.exists() and dest.stat().st_size == size:
        return key, True, "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{BUCKET}/{key}", timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if tmp.stat().st_size != size:
                raise IOError(f"size mismatch {tmp.stat().st_size} != {size}")
            tmp.rename(dest)
            return key, True, "downloaded"
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(2 ** attempt)
    return key, False, err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw/ds004889"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="only first N subjects (debug)")
    args = ap.parse_args()

    keys = [(k, s) for k, s in list_keys() if WANTED.match(k)]
    if args.limit:
        subs = sorted({m.group(0) for k, _ in keys if (m := re.search(r"sub-\d+", k))},
                      key=lambda s: int(s[4:]))[: args.limit]
        keys = [(k, s) for k, s in keys if not re.search(r"sub-\d+", k) or re.search(r"sub-\d+", k).group(0) in subs]
    total = sum(s for _, s in keys)
    print(f"{len(keys)} files, {total/1e9:.2f} GB -> {args.out}", flush=True)

    done = fails = 0
    got = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch, k, s, args.out): (k, s) for k, s in keys}
        for fut in cf.as_completed(futs):
            key, ok, msg = fut.result()
            done += 1
            got += futs[fut][1]
            if not ok:
                fails += 1
                print(f"FAIL {key}: {msg}", file=sys.stderr, flush=True)
            if done % 200 == 0 or done == len(keys):
                el = time.time() - t0
                print(f"[{done}/{len(keys)}] {got/1e9:.2f} GB  {got/1e6/max(el,1e-9):.1f} MB/s  fails={fails}", flush=True)
    print("DONE" if fails == 0 else f"FINISHED WITH {fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
