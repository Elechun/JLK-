#!/usr/bin/env python3
"""Etiology classification (LAA / CE / SVO / Others) from lesion features + clinical variables.

Two feature sources are supported:
  --mask gt     : ground-truth acute masks   (oracle: how much signal is in the lesion pattern at all)
  --mask pred   : segmentation-model masks   (deployment-like; needs runs/<run>/pred_masks/*.npz from scripts/predict_masks.py)
Two feature sets are reported side by side:
  full          : lesion + clinical
  no_size       : drops volume / component-size features (SVO is *defined* by lesion size <1.5 cm -> size leaks the label)
Cryptogenic is excluded by default (Astra review). Model selection uses stratified 5-fold CV on train+val subjects only; `--final` fits on train+val and scores test ONCE.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strokeai.features import FEATURE_NAMES, lesion_features  # noqa: E402
from strokeai.metrics import multiclass_auc  # noqa: E402
from strokeai.utils import seed_everything  # noqa: E402

CLASSES = ["LAA", "CE", "SVO", "Others"]
SIZE_FEATS = ["volume_ml", "log_volume_ml", "largest_component_ml", "n_slices_involved", "z_extent_frac"]
CLIN = ["age", "sex_male", "nihss"]


def label_of(code: str, include_cryptogenic: bool = False) -> str | None:
    """Astra review: Cryptogenic (= undetermined) is NOT an etiology; exclude it by default instead of merging into Others."""
    if code in ("LAA", "CE", "SVO"):
        return code
    if code == "OtherDet" or (include_cryptogenic and code == "Cryptogenic"):
        return "Others"
    return None


def build_table(index: pd.DataFrame, ids: list[str], cache: Path, mask_src: str, pred_dir: Path | None, include_cryptogenic: bool = False) -> pd.DataFrame:
    rows = []
    for sid in ids:
        r = index.loc[index["participant_id"] == sid].iloc[0]
        y = label_of(str(r["etiology_code"]), include_cryptogenic)
        if y is None:
            continue
        z = np.load(cache / f"{sid}.npz")
        vv = float(z["voxel_volume_mm3"])
        if mask_src == "gt":
            m = z["mask"]
        else:
            m = np.load(pred_dir / f"{sid}.npz")["mask"]
        f = lesion_features(m, vv)
        f.update(participant_id=sid, y=y, age=r["age"], sex_male=float(r["sex"] == "M") if r["sex"] in ("M", "F") else np.nan, nihss=r["nihss"])
        rows.append(f)
    return pd.DataFrame(rows)


def make_model(kind: str, seed: int):
    if kind == "logreg":
        return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, random_state=seed))
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, class_weight="balanced", random_state=seed)


def evaluate(df: pd.DataFrame, feats: list[str], kind: str, seed: int, test_df: pd.DataFrame | None = None) -> dict:
    X, y = df[feats].values.astype(float), df["y"].values
    model = make_model(kind, seed)
    if test_df is None:
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
        classes = sorted(set(y))
    else:
        model.fit(X, y)
        proba = model.predict_proba(test_df[feats].values.astype(float))
        y = test_df["y"].values
        classes = list(model.classes_)
    order = [classes.index(c) for c in CLASSES if c in classes]
    auc = multiclass_auc(y, proba[:, order], [c for c in CLASSES if c in classes])
    pred = np.array([c for c in CLASSES if c in classes])[proba[:, order].argmax(1)]
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix

    return {"n": int(len(y)), "class_counts": {c: int((y == c).sum()) for c in CLASSES}, **auc,
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "confusion": confusion_matrix(y, pred, labels=[c for c in CLASSES if c in classes]).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/index.csv"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--mask", choices=["gt", "pred"], default="gt")
    ap.add_argument("--pred-dir", type=Path, default=None)
    ap.add_argument("--model", choices=["logreg", "hgb"], default="logreg")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--final", action="store_true", help="fit train+val, evaluate on TEST once (after A6 sign-off)")
    ap.add_argument("--include-cryptogenic", action="store_true", help="merge Cryptogenic into Others (not recommended, see Astra report)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    seed_everything(a.seed)
    index = pd.read_csv(a.index)
    index["etiology_code"] = index["etiology_code"].fillna("n/a")
    sp = json.load(open(a.splits))
    dev = build_table(index, sp["train"] + sp["val"], a.cache, a.mask, a.pred_dir, a.include_cryptogenic)
    print(f"dev subjects with etiology label: {len(dev)}  classes: {dev['y'].value_counts().to_dict()}")
    sets = {"full": FEATURE_NAMES + CLIN, "no_size": [f for f in FEATURE_NAMES if f not in SIZE_FEATS] + CLIN,
            "clinical_only": CLIN, "size_only": ["log_volume_ml", "largest_component_ml"]}
    res = {"mask_source": a.mask, "model": a.model, "seed": a.seed, "final": a.final, "include_cryptogenic": a.include_cryptogenic, "cv": {}}
    for name, feats in sets.items():
        res["cv"][name] = evaluate(dev, feats, a.model, a.seed)
        print(name, json.dumps({k: res["cv"][name][k] for k in ["macro_auc_ovr", "balanced_accuracy", "per_class_auc"]}))
    if a.final:
        test = build_table(index, sp["test"], a.cache, a.mask, a.pred_dir, a.include_cryptogenic)
        res["test"] = {name: evaluate(dev, feats, a.model, a.seed, test_df=test) for name, feats in sets.items()}
        print("TEST", json.dumps({k: v["macro_auc_ovr"] for k, v in res["test"].items()}))
    out = a.out or Path("results") / f"cls_{a.mask}_{a.model}{'_final' if a.final else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
