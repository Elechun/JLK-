#!/usr/bin/env python3
"""Etiology classification (LAA / CE / SVO / Others) from lesion features + clinical variables.

Two feature sources are supported:
  --mask gt     : ground-truth acute masks   (oracle: how much signal is in the lesion pattern at all)
  --mask pred   : segmentation-model masks   (deployment-like; needs runs/<run>/pred_masks/*.npz from scripts/predict_masks.py)
Feature sets reported side by side (A6 2026-09-09, charter s."etiology"):
  volume_only     : log lesion volume, ONE feature -> the pre-registered baseline every other set must beat
  full            : lesion + clinical
  no_size         : drops the explicit volume / component-size features but KEEPS n_components and NIHSS,
                    which A2 measured at Spearman rho +0.48 / +0.47 against log volume. It is therefore
                    *not* a size-free set; kept only for continuity with the pre-A6 reports.
  scale_invariant : the charter's "size-excluded" set as redefined by A6 - dimensionless by construction
                    AND |rho vs log volume| < 0.25 measured on dev (see SCALE_INVARIANT below)
  clinical_only   : age, sex, NIHSS
  size_only       : log volume + largest component volume
Secondary criteria, all on the same CV folds:
  LAA-vs-CE AUC   : discrimination between the two large-lesion etiologies, where the size shortcut is
                    weakest; scored as p(LAA) - p(CE) on the subjects whose true label is LAA or CE
  no_svo_3class   : macro OvR AUC after dropping SVO entirely (SVO is defined by size <1.5 cm)
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
from strokeai.metrics import laa_vs_ce_auc, multiclass_auc  # noqa: E402
from strokeai.utils import seed_everything  # noqa: E402

CLASSES = ["LAA", "CE", "SVO", "Others"]
SIZE_FEATS = ["volume_ml", "log_volume_ml", "largest_component_ml", "n_slices_involved", "z_extent_frac"]
CLIN = ["age", "sex_male", "nihss"]
VOLUME_ONLY = ["log_volume_ml"]
# A6 2026-09-09.  Spearman rho against log volume, measured on the 447 dev subjects with GT masks:
#   largest_component_frac -0.101 | centroid_x -0.021 | centroid_y +0.008 | centroid_z +0.172
#   age -0.095 | sex_male -0.006                                       <- kept (|rho| < 0.25)
#   laterality_left_frac -0.415 | bilateral +0.421 | multi_territory +0.395 | n_components +0.476
#   spread_x +0.746 | spread_y +0.745 | nihss +0.471                   <- dropped, they are size proxies
# Everything kept is also dimensionless by construction (a ratio, a normalised coordinate, or a
# patient covariate), so the exclusion is not purely data-driven.
SCALE_INVARIANT = ["largest_component_frac", "centroid_x_norm", "centroid_y_norm", "centroid_z_norm", "age", "sex_male"]


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


def evaluate(df: pd.DataFrame, feats: list[str], kind: str, seed: int, test_df: pd.DataFrame | None = None,
             return_oof: bool = False):
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
    present = [c for c in CLASSES if c in classes]
    order = [classes.index(c) for c in present]
    p = proba[:, order]
    auc = multiclass_auc(y, p, present)
    pred = np.array(present)[p.argmax(1)]
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix

    res = {"n": int(len(y)), "class_counts": {c: int((y == c).sum()) for c in CLASSES}, **auc,
           "laa_vs_ce_auc": laa_vs_ce_auc(y, p, present),
           "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "confusion": confusion_matrix(y, pred, labels=present).tolist()}
    return (res, y, p, present) if return_oof else res


def bootstrap_deltas(oof: dict, pairs: list[tuple[str, str]], n_boot: int = 2000, seed: int = 2026) -> dict:
    """Paired subject bootstrap over the stored out-of-fold probabilities.

    Reports each set's macro-OvR / LAA-vs-CE AUC with a percentile CI, and the CI of the *difference*
    between two sets on the same resampled subjects (paired -> the CI of the increment is much
    tighter than the difference of the two marginal CIs).
    """
    rng = np.random.default_rng(seed)
    names = list(oof)
    y0 = oof[names[0]][0]
    n = len(y0)
    idx = [rng.integers(0, n, n) for _ in range(n_boot)]
    macro = {k: [] for k in names}
    lce = {k: [] for k in names}
    for i in idx:
        for k in names:
            y, p, present = oof[k]
            macro[k].append(multiclass_auc(y[i], p[i], present)["macro_auc_ovr"])
            lce[k].append(laa_vs_ce_auc(y[i], p[i], present))
    out = {"n_boot": n_boot, "per_set": {}, "deltas": {}}
    for k in names:
        y, p, present = oof[k]
        out["per_set"][k] = {
            "macro_auc_ovr": multiclass_auc(y, p, present)["macro_auc_ovr"],
            "macro_ci95": [float(np.nanpercentile(macro[k], 2.5)), float(np.nanpercentile(macro[k], 97.5))],
            "laa_vs_ce_auc": laa_vs_ce_auc(y, p, present),
            "laa_vs_ce_ci95": [float(np.nanpercentile(lce[k], 2.5)), float(np.nanpercentile(lce[k], 97.5))],
        }
    for a_, b_ in pairs:
        if a_ not in oof or b_ not in oof:
            continue
        dm = np.asarray(macro[a_], float) - np.asarray(macro[b_], float)
        dl = np.asarray(lce[a_], float) - np.asarray(lce[b_], float)
        out["deltas"][f"{a_}-{b_}"] = {
            "macro_delta": out["per_set"][a_]["macro_auc_ovr"] - out["per_set"][b_]["macro_auc_ovr"],
            "macro_delta_ci95": [float(np.nanpercentile(dm, 2.5)), float(np.nanpercentile(dm, 97.5))],
            "macro_delta_p_gt_0": float(np.mean(dm > 0)),
            "laa_vs_ce_delta": out["per_set"][a_]["laa_vs_ce_auc"] - out["per_set"][b_]["laa_vs_ce_auc"],
            "laa_vs_ce_delta_ci95": [float(np.nanpercentile(dl, 2.5)), float(np.nanpercentile(dl, 97.5))],
        }
    return out


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
    sets = {"volume_only": VOLUME_ONLY,
            "full": FEATURE_NAMES + CLIN,
            "no_size": [f for f in FEATURE_NAMES if f not in SIZE_FEATS] + CLIN,
            "scale_invariant": SCALE_INVARIANT,
            "clinical_only": CLIN,
            "size_only": ["log_volume_ml", "largest_component_ml"]}
    pairs = [("full", "volume_only"), ("scale_invariant", "volume_only"), ("no_size", "volume_only"),
             ("full", "clinical_only")]
    res = {"mask_source": a.mask, "model": a.model, "seed": a.seed, "final": a.final,
           "include_cryptogenic": a.include_cryptogenic, "feature_sets": sets, "cv": {}}
    oof = {}
    for name, feats in sets.items():
        res["cv"][name], y_, p_, cl_ = evaluate(dev, feats, a.model, a.seed, return_oof=True)
        oof[name] = (y_, p_, cl_)
        print(name, json.dumps({k: res["cv"][name][k] for k in ["macro_auc_ovr", "laa_vs_ce_auc", "balanced_accuracy", "per_class_auc"]}))
    res["cv_bootstrap"] = bootstrap_deltas(oof, pairs, seed=a.seed)

    # SVO is *defined* by lesion size (<1.5 cm), so a macro AUC that drops it shows what is left
    # once the label-definition shortcut is gone (A2 F3).
    dev3 = dev[dev["y"] != "SVO"].reset_index(drop=True)
    res["cv_no_svo_3class"] = {name: evaluate(dev3, feats, a.model, a.seed) for name, feats in sets.items()}
    print("no-SVO 3-class macro:", json.dumps({k: round(v["macro_auc_ovr"], 4) for k, v in res["cv_no_svo_3class"].items()}))

    if a.final:
        test = build_table(index, sp["test"], a.cache, a.mask, a.pred_dir, a.include_cryptogenic)
        res["test"] = {}
        toof = {}
        for name, feats in sets.items():
            res["test"][name], y_, p_, cl_ = evaluate(dev, feats, a.model, a.seed, test_df=test, return_oof=True)
            toof[name] = (y_, p_, cl_)
        res["test_bootstrap"] = bootstrap_deltas(toof, pairs, seed=a.seed)
        test3 = test[test["y"] != "SVO"].reset_index(drop=True)
        res["test_no_svo_3class"] = {name: evaluate(dev3, feats, a.model, a.seed, test_df=test3) for name, feats in sets.items()}
        print("TEST macro", json.dumps({k: round(v["macro_auc_ovr"], 4) for k, v in res["test"].items()}))
        print("TEST LAAvsCE", json.dumps({k: round(v["laa_vs_ce_auc"], 4) for k, v in res["test"].items()}))
    out = a.out or Path("results") / f"cls_{a.mask}_{a.model}{'_final' if a.final else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
