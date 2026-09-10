"""Discharge-mRS prognosis from acute DWI lesion features + clinical variables (A7 2026-09-10).

Task (charter, "확장 과제 — 예후 예측"):
    binary outcome  poor = mRS 3-6  vs  good = mRS 0-2   at DISCHARGE (`gs_rankin_6isdeath`).
    NOTE: the international standard endpoint is the 90-day mRS. This dataset only carries the
    discharge score, which sits closer to acute severity; results are NOT comparable to 90-day
    literature (A1b risk table).

Held-out discipline
-------------------
`load_cohort` refuses to touch a subject in `data/splits.json["test"]` unless `allow_test=True`, which
only the orchestrator's single final evaluation passes. Everything in this module - feature building,
model selection, thresholds, calibration - is defined on dev = train + val.

Inputs
------
clinical : age, sex_male, nihss, priorstroke      (0 missing among the 538 dev subjects with mRS)
lesion   : the 15 features of `strokeai.features` computed on
             * the segmentation model's PREDICTED mask (primary, deployment-like), or
             * the ground-truth mask (oracle contrast - an input reference, not a performance ceiling)
Forbidden inputs (charter): mRS itself, `acuteischaemicstroke` (constant, A2 F9), manufacturer and
field strength (protocol leakage), and anything recorded after the outcome.

Model selection
---------------
`nested_cv` estimates the performance of the *selection procedure*: the candidate configuration
(feature set x algorithm x hyper-parameter) is chosen inside each outer training fold by an inner
5-fold CV, and scored on the untouched outer fold. Picking the best of several plain 5-fold CV runs
would be optimistically biased (A5a item 8). The three baselines are FIXED configurations, refitted
on the same outer folds so that every delta is paired.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, lesion_features
from .metrics import binary_auc

MRS_COL = "gs_rankin_6isdeath"
POOR_CUTOFF = 3  # mRS >= 3 -> poor functional outcome (the standard 0-2 / 3-6 dichotomy)
CLINICAL = ["age", "sex_male", "nihss", "priorstroke"]

# A small, pre-specified imaging block: volume, burden pattern and cranio-caudal position.
# Chosen before any CV was run, from what a radiology report of a DWI actually states.
LESION_CORE = ["log_volume_ml", "n_components", "largest_component_frac", "n_slices_involved",
               "centroid_z_norm"]

FEATURE_SETS: dict[str, list[str]] = {
    # --- the three pre-registered baselines -------------------------------------------------------
    "nihss_only": ["nihss"],
    "volume_only": ["log_volume_ml"],
    "clinical_only": CLINICAL,
    # --- imaging + clinical candidates ------------------------------------------------------------
    "clinical_volume": CLINICAL + ["log_volume_ml"],
    "clinical_lesion_core": CLINICAL + LESION_CORE,
    "clinical_lesion_all": CLINICAL + FEATURE_NAMES,
    "lesion_all": FEATURE_NAMES,
}
BASELINES = ["clinical_only", "nihss_only", "volume_only"]
IMAGING_SETS = ["clinical_volume", "clinical_lesion_core", "clinical_lesion_all", "lesion_all"]


def make_model(spec: dict, seed: int = 2026):
    """spec = {"algo": "logreg", "C": float} or {"algo": "hgb", "max_depth": int, "max_iter": int}.

    No class weighting: the dev outcome is near-balanced (265 good : 273 poor) and calibration is a
    reported endpoint, so re-weighting would distort the predicted risks for no discrimination gain.
    """
    algo = spec["algo"]
    if algo == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000, C=spec["C"], random_state=seed))
    if algo == "hgb":
        return HistGradientBoostingClassifier(max_depth=spec.get("max_depth", 3),
                                              learning_rate=spec.get("learning_rate", 0.05),
                                              max_iter=spec.get("max_iter", 200),
                                              min_samples_leaf=spec.get("min_samples_leaf", 20),
                                              l2_regularization=spec.get("l2", 1.0),
                                              early_stopping=False, random_state=seed)
    raise ValueError(algo)


def candidate_grid(sets: list[str]) -> list[dict]:
    """feature set x algorithm x hyper-parameter. Kept deliberately small for n = 538."""
    grid = []
    for s in sets:
        for c in (0.03, 0.1, 0.3, 1.0):
            grid.append({"feature_set": s, "algo": "logreg", "C": c})
        for md, ms in ((2, 40), (3, 20)):
            grid.append({"feature_set": s, "algo": "hgb", "max_depth": md, "min_samples_leaf": ms})
    return grid


def spec_name(spec: dict) -> str:
    if spec["algo"] == "logreg":
        return f"{spec['feature_set']}|logreg|C={spec['C']}"
    return f"{spec['feature_set']}|hgb|d={spec.get('max_depth', 3)},leaf={spec.get('min_samples_leaf', 20)}"


# ------------------------------------------------------------------------------------------------
# cohort
# ------------------------------------------------------------------------------------------------
def load_cohort(index: pd.DataFrame, splits: dict, cache: Path, mask_src: str = "pred",
                pred_dir: Path | None = None, which: tuple[str, ...] = ("train", "val"),
                allow_test: bool = False) -> pd.DataFrame:
    """Build the prognosis table for the requested splits.

    Raises unless `allow_test=True` when "test" is requested: reading a test subject's mRS is the one
    thing the extension pre-registration forbids before the orchestrator's single final evaluation.
    """
    if "test" in which and not allow_test:
        raise PermissionError(
            "refusing to read the held-out test split (charter, extension pre-registration). "
            "Pass allow_test=True only in the single final evaluation.")
    ids = [s for w in which for s in splits[w]]
    idx = index.set_index("participant_id")
    rows = []
    for sid in ids:
        r = idx.loc[sid]
        mrs = r[MRS_COL]
        if pd.isna(mrs):
            continue
        z = np.load(cache / f"{sid}.npz")
        vv = float(z["voxel_volume_mm3"])
        m = z["mask"] if mask_src == "gt" else np.load(pred_dir / f"{sid}.npz")["mask"]
        f = lesion_features(m, vv)
        f.update(participant_id=sid, mrs=float(mrs), y=int(float(mrs) >= POOR_CUTOFF),
                 age=float(r["age"]), sex_male=float(r["sex"] == "M"),
                 nihss=float(r["nihss"]), priorstroke=float(r["priorstroke"]),
                 split="val" if sid in set(splits["val"]) else ("train" if sid in set(splits["train"]) else "test"))
        rows.append(f)
    df = pd.DataFrame(rows)
    missing = df[CLINICAL].isna().sum().to_dict()
    if any(missing.values()):
        raise ValueError(f"clinical variables are supposed to be complete on the mRS cohort: {missing}")
    return df


# ------------------------------------------------------------------------------------------------
# cross-validation
# ------------------------------------------------------------------------------------------------
def _fit_predict(df_tr: pd.DataFrame, df_te: pd.DataFrame, spec: dict, seed: int) -> np.ndarray:
    feats = FEATURE_SETS[spec["feature_set"]]
    model = make_model(spec, seed)
    model.fit(df_tr[feats].to_numpy(float), df_tr["y"].to_numpy(int))
    return model.predict_proba(df_te[feats].to_numpy(float))[:, 1]


def _inner_score(df_tr: pd.DataFrame, spec: dict, seed: int, n_inner: int) -> tuple[float, float]:
    """Mean inner-fold AUC of one candidate on the outer TRAINING data only, and its fold SE."""
    y = df_tr["y"].to_numpy(int)
    cv = StratifiedKFold(n_inner, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in cv.split(np.zeros(len(y)), y):
        p = _fit_predict(df_tr.iloc[tr], df_tr.iloc[te], spec, seed)
        aucs.append(binary_auc(y[te], p))
    a = np.asarray(aucs, float)
    return float(np.nanmean(a)), float(np.nanstd(a, ddof=1) / np.sqrt(len(a)))


# Selection rules applied to the inner-CV scores.  `complexity` orders the candidates from simplest to
# most complex so the one-SE rule has a well-defined "simplest model within one SE of the best"
# (Breiman et al. 1984; Hastie, Tibshirani & Friedman, ESL 2nd ed. section 7.10).
def complexity(spec: dict) -> tuple:
    n_feat = len(FEATURE_SETS[spec["feature_set"]])
    if spec["algo"] == "logreg":
        # a stronger penalty (smaller C) is a simpler model
        return (0, n_feat, spec["C"])
    return (1, n_feat, spec.get("max_depth", 3), -spec.get("min_samples_leaf", 20))


def apply_rule(scored: list[dict], rule: str) -> dict:
    """scored: [{"spec", "mean", "se"}]. rule = "argmax" | "one_se"."""
    best = max(scored, key=lambda d: d["mean"])
    if rule == "argmax":
        return best
    thr = best["mean"] - best["se"]
    within = [d for d in scored if d["mean"] >= thr]
    return min(within, key=lambda d: complexity(d["spec"]))


def nested_cv(df: pd.DataFrame, candidates: list[dict], baselines: list[str],
              seed: int = 2026, n_outer: int = 5, n_inner: int = 5, repeats: int = 5,
              rules: tuple[str, ...] = ("argmax", "one_se"),
              pools: dict[str, list[dict]] | None = None) -> dict:
    """Repeated nested CV.

    Every candidate in `candidates` is scored by an inner CV on the outer TRAINING fold only.  The
    selection rules and the candidate POOLS are then applied to those same inner scores, so adding a
    rule or a pool costs nothing extra: one inner CV serves all of them.  `pools` maps a pool name to
    a subset of `candidates` (default: one pool "all"); the deployed pipeline restricts the pool to
    configurations that contain lesion features, while an unrestricted pool is reported next to it so
    the reader can see whether a purely clinical configuration would ever have been selected.

    Each fixed baseline is refitted on the SAME outer folds, so every delta is paired.  Per-repeat
    out-of-fold probabilities are averaged over repeats (`oof`), which is the headline.
    """
    y = df["y"].to_numpy(int)
    n = len(df)
    pools = pools or {"all": candidates}
    base_specs = {b: {"feature_set": b, "algo": "logreg", "C": 0.3} for b in baselines}
    combos = [(pn, r) for pn in pools for r in rules]
    out = {"n": n, "n_poor": int(y.sum()), "seed": seed, "n_outer": n_outer, "n_inner": n_inner,
           "repeats": repeats, "rules": list(rules),
           "pools": {k: [spec_name(s) for s in v] for k, v in pools.items()},
           "baseline_specs": {k: spec_name(v) for k, v in base_specs.items()},
           "per_repeat": [], "selected_per_fold": []}
    oof_sel = {f"{pn}:{r}": np.zeros((repeats, n)) for pn, r in combos}
    oof_base = {b: np.zeros((repeats, n)) for b in baselines}
    for r in range(repeats):
        cv = StratifiedKFold(n_outer, shuffle=True, random_state=seed + r)
        for k, (tr, te) in enumerate(cv.split(np.zeros(n), y)):
            df_tr, df_te = df.iloc[tr], df.iloc[te]
            scored = []
            for spec in candidates:
                m, se = _inner_score(df_tr, spec, seed + 100 * r, n_inner)
                scored.append({"spec": spec, "mean": m, "se": se})
            by_name = {spec_name(d["spec"]): d for d in scored}
            rec = {"repeat": r, "fold": k}
            cache: dict[str, np.ndarray] = {}
            for pn, rule in combos:
                subset = [by_name[spec_name(s)] for s in pools[pn]]
                pick = apply_rule(subset, rule)
                name = spec_name(pick["spec"])
                rec[f"{pn}:{rule}"] = name
                rec[f"{pn}:{rule}:inner_auc"] = pick["mean"]
                if name not in cache:
                    cache[name] = _fit_predict(df_tr, df_te, pick["spec"], seed)
                oof_sel[f"{pn}:{rule}"][r, te] = cache[name]
            out["selected_per_fold"].append(rec)
            for b, s_ in base_specs.items():
                oof_base[b][r, te] = _fit_predict(df_tr, df_te, s_, seed)
        rep = {"repeat": r}
        rep.update({f"{key}_auc": binary_auc(y, oof_sel[key][r]) for key in oof_sel})
        rep.update({f"{b}_auc": binary_auc(y, oof_base[b][r]) for b in baselines})
        out["per_repeat"].append(rep)
    out["oof"] = {"y": y.tolist(), "participant_id": df["participant_id"].tolist(),
                  "mrs": df["mrs"].tolist(),
                  **{key: v.mean(0).tolist() for key, v in oof_sel.items()},
                  **{b: v.mean(0).tolist() for b, v in oof_base.items()}}
    out["selection_frequency"] = {
        f"{pn}:{rule}": {k: int(v) for k, v in
                         pd.Series([s[f"{pn}:{rule}"] for s in out["selected_per_fold"]]).value_counts().items()}
        for pn, rule in combos}
    return out


def plain_cv_oof(df: pd.DataFrame, spec: dict, seed: int = 2026, n_outer: int = 5,
                 repeats: int = 5) -> np.ndarray:
    """Repeat-averaged out-of-fold probabilities of ONE fixed configuration (same folds as nested_cv)."""
    y = df["y"].to_numpy(int)
    n = len(df)
    oof = np.zeros((repeats, n))
    for r in range(repeats):
        cv = StratifiedKFold(n_outer, shuffle=True, random_state=seed + r)
        for tr, te in cv.split(np.zeros(n), y):
            oof[r, te] = _fit_predict(df.iloc[tr], df.iloc[te], spec, seed)
    return oof.mean(0)


def ordinal_cv_pred(df: pd.DataFrame, feature_set: str, seed: int = 2026, n_outer: int = 5,
                    repeats: int = 5, alpha: float = 1.0) -> np.ndarray:
    """Auxiliary ordinal analysis: ridge regression on the raw mRS 0-6, same folds.

    A ridge on an ordinal score is not a proper ordinal model; it is reported only as a rank/MAE
    check that the binary dichotomy is not hiding a different story (charter: auxiliary, not a gate).
    """
    feats = FEATURE_SETS[feature_set]
    y = df["y"].to_numpy(int)
    m = df["mrs"].to_numpy(float)
    n = len(df)
    pred = np.zeros((repeats, n))
    for r in range(repeats):
        cv = StratifiedKFold(n_outer, shuffle=True, random_state=seed + r)
        for tr, te in cv.split(np.zeros(n), y):
            mdl = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=None))
            mdl.fit(df.iloc[tr][feats].to_numpy(float), m[tr])
            pred[r, te] = mdl.predict(df.iloc[te][feats].to_numpy(float))
    return pred.mean(0)


def fit_frozen(df: pd.DataFrame, spec: dict, seed: int = 2026):
    """Fit the frozen configuration on the whole dev set (what the final test evaluation will use)."""
    feats = FEATURE_SETS[spec["feature_set"]]
    model = make_model(spec, seed)
    model.fit(df[feats].to_numpy(float), df["y"].to_numpy(int))
    return model, feats


def select_on_dev(df: pd.DataFrame, candidates: list[dict], seed: int = 2026, n_inner: int = 5,
                  rule: str = "one_se") -> dict:
    """Run the same inner-CV selection rule once on the FULL dev set -> the configuration to freeze."""
    scored = []
    for spec in candidates:
        m, se = _inner_score(df, spec, seed, n_inner)
        scored.append({"spec": spec, "mean": m, "se": se})
    ranking = sorted(scored, key=lambda d: -d["mean"])
    picks = {r: apply_rule(scored, r) for r in ("argmax", "one_se")}
    chosen = picks[rule]
    return {"rule": rule,
            "ranking": [{"name": spec_name(d["spec"]), "inner_auc": d["mean"], "inner_se": d["se"]}
                        for d in ranking],
            "picks": {r: {"name": spec_name(p["spec"]), "inner_auc": p["mean"]} for r, p in picks.items()},
            "one_se_threshold": max(d["mean"] for d in scored) - max(scored, key=lambda d: d["mean"])["se"],
            "selected": chosen["spec"], "selected_name": spec_name(chosen["spec"]),
            "selected_inner_auc": chosen["mean"]}


def load_splits(path: Path) -> dict:
    return json.load(open(path))
