#!/usr/bin/env python3
"""Patient-level stratified split -> data/splits.json. Re-running with the same seed must give the same sha256."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strokeai.data.split import eligible_for_segmentation, make_split  # noqa: E402
from strokeai.utils import seed_everything  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/index.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    seed_everything(a.seed)
    idx = pd.read_csv(a.index)
    idx["etiology_code"] = idx["etiology_code"].fillna("n/a")
    elig = eligible_for_segmentation(idx)
    split = make_split(elig, a.seed)
    split["n_eligible"] = int(len(elig))
    split["n_index"] = int(len(idx))
    with open(a.out, "w") as f:
        json.dump(split, f, indent=1)
    print({k: split[k] for k in ["seed", "n", "sha256", "n_eligible", "n_index"]})
    # stratum table
    elig = elig.assign(split="train")
    for k in ["val", "test"]:
        elig.loc[elig["participant_id"].isin(split[k]), "split"] = k
    tab = pd.crosstab([elig["etiology_code"], elig["mask_acute_voxels"].eq(0).map({True: "empty", False: "lesion"})], elig["split"])
    print(tab)
    tab.to_csv(a.out.with_name("splits_strata.csv"))


if __name__ == "__main__":
    main()
