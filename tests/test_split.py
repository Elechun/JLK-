import numpy as np
import pandas as pd

from strokeai.data.split import make_split


def _fake(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "participant_id": [f"sub-{i}" for i in range(1, n + 1)],
        "etiology_code": rng.choice(["LAA", "CE", "SVO", "Cryptogenic", "n/a"], n),
        "mask_acute_voxels": rng.integers(0, 5000, n),
        "mask_acute_ml": rng.uniform(0, 30, n),
    })


def test_no_overlap_and_coverage():
    df = _fake()
    s = make_split(df, seed=1)
    tr, va, te = set(s["train"]), set(s["val"]), set(s["test"])
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == set(df["participant_id"])


def test_deterministic_same_seed_and_order_independent():
    df = _fake()
    a = make_split(df, seed=7)
    b = make_split(df.sample(frac=1, random_state=3), seed=7)  # shuffled input rows
    assert a["sha256"] == b["sha256"]
    assert make_split(df, seed=8)["sha256"] != a["sha256"]


def test_stratification_roughly_preserved():
    df = _fake(n=1000)
    s = make_split(df, seed=2)
    frac = lambda ids: (df.set_index("participant_id").loc[ids, "etiology_code"] == "SVO").mean()  # noqa: E731
    assert abs(frac(s["train"]) - frac(s["test"])) < 0.05
