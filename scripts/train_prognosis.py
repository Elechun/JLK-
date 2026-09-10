#!/usr/bin/env python3
"""Discharge-mRS prognosis: acute DWI lesion features + clinical variables -> poor outcome (mRS 3-6).

    python scripts/train_prognosis.py --mask pred          # primary, deployment-like
    python scripts/train_prognosis.py --mask gt            # oracle-input contrast
    python scripts/train_prognosis.py --mask pred --final  # ORCHESTRATOR ONLY, single test evaluation

Everything except `--final` stays inside dev = train + val (`strokeai.prognosis.load_cohort` raises if
a test subject is requested without `allow_test`).  `--final` refits the FROZEN configuration recorded
in `results/prognosis/frozen_model.json` on the whole dev set and scores the test subjects once; it
never re-selects features, algorithm, hyper-parameters or thresholds.

What is reported (charter, extension pre-registration v1):
  * nested-CV AUC of the selection procedure vs the three fixed baselines (clinical / NIHSS / volume)
  * subject-bootstrap CI of every AUC and of every paired delta
  * calibration: Brier, Brier skill, calibration slope/intercept, quantile calibration table
  * confusion matrix at the frozen operating points (0.50 and the dev Youden point)
  * auxiliary: ordinal mRS ridge (Spearman/MAE), NIHSS strata, seg-train vs seg-val strata
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strokeai.metrics import (  # noqa: E402
    binary_auc, bootstrap_binary_auc_ci, ordinal_summary, paired_auc_delta_ci, prognosis_summary,
)
from strokeai.prognosis import (  # noqa: E402
    BASELINES, CLINICAL, FEATURE_SETS, IMAGING_SETS, candidate_grid, fit_frozen, load_cohort,
    nested_cv, ordinal_cv_pred, plain_cv_oof, select_on_dev, spec_name,
)
from strokeai.utils import seed_everything  # noqa: E402


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Operating point that maximises sensitivity + specificity - 1 on the dev out-of-fold scores.

    Frozen on dev and never re-tuned on test (charter section C discipline)."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, p)
    j = np.argmax(tpr - fpr)
    return float(thr[j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/index.csv"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--which", nargs="+", default=["train", "val"],
                    help="splits to build the dev cohort from; 'test' is never accepted here")
    ap.add_argument("--mask", choices=["gt", "pred"], default="pred")
    ap.add_argument("--pred-dir", type=Path, default=Path("runs/seg_unet2d/pred_masks"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--rule", choices=["argmax", "one_se"], default="one_se",
                    help="inner-CV selection rule used to FREEZE the primary model")
    ap.add_argument("--frozen", type=Path, default=Path("results/prognosis/frozen_model.json"))
    ap.add_argument("--final", action="store_true",
                    help="ORCHESTRATOR ONLY: refit the frozen model on dev and score TEST once")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    seed_everything(a.seed)
    if "test" in a.which:
        ap.error("--which never accepts 'test': the dev analysis is train+val only. The single final "
                 "evaluation is --final, which builds the test cohort itself from the frozen config.")

    index = pd.read_csv(a.index)
    sp = json.load(open(a.splits))
    dev = load_cohort(index, sp, a.cache, a.mask, a.pred_dir, which=tuple(a.which))
    y = dev["y"].to_numpy(int)
    print(f"dev cohort: n={len(dev)}  poor(mRS>=3)={y.sum()}  good={len(y) - y.sum()}  mask={a.mask}")

    res = {"mask_source": a.mask, "seed": a.seed, "which": a.which, "final": a.final,
           "cohort": {"n": int(len(dev)), "n_poor": int(y.sum()), "n_good": int(len(y) - y.sum()),
                      "mrs_counts": {str(int(k)): int(v) for k, v in dev["mrs"].value_counts().sort_index().items()},
                      "clinical_missing": {c: int(dev[c].isna().sum()) for c in CLINICAL}},
           "feature_sets": {k: v for k, v in FEATURE_SETS.items()}}

    # ---------------------------------------------------------------------------------------------
    # 1. nested CV.  Two candidate pools: the deployed imaging pipeline, and an unrestricted pool that
    #    is allowed to drop imaging entirely (so the reader can see whether the selector wants it).
    # ---------------------------------------------------------------------------------------------
    all_cands = candidate_grid(IMAGING_SETS + BASELINES)
    pools = {"imaging": candidate_grid(IMAGING_SETS), "unrestricted": all_cands}
    rules = ("argmax", "one_se")
    nc = nested_cv(dev, all_cands, BASELINES, seed=a.seed, repeats=a.repeats, rules=rules, pools=pools)
    oof = nc["oof"]
    keys = [f"{pn}:{r}" for pn in pools for r in rules] + BASELINES
    summary, deltas = {}, {}
    for key in keys:
        p = np.asarray(oof[key], float)
        summary[key] = prognosis_summary(y, p, threshold=0.5)
        summary[key].update(bootstrap_binary_auc_ci(y, p, n_boot=a.n_boot, seed=a.seed))
    for pn in pools:
        for r in rules:
            for base in BASELINES:
                deltas[f"{pn}:{r}-{base}"] = paired_auc_delta_ci(
                    y, np.asarray(oof[f"{pn}:{r}"], float), np.asarray(oof[base], float),
                    n_boot=a.n_boot, seed=a.seed)
    res["nested_cv"] = {**nc, "summary": summary, "deltas": deltas,
                        "repeat_auc_range": {k: [min(r_[f"{k}_auc"] for r_ in nc["per_repeat"]),
                                                 max(r_[f"{k}_auc"] for r_ in nc["per_repeat"])] for k in keys}}
    for key in keys:
        print(f"  nested {key:28s} AUC {summary[key]['auc']:.4f} "
              f"CI {np.round(summary[key]['ci95'], 4).tolist()}")
    print("selection frequency:", json.dumps(nc["selection_frequency"], indent=1))

    # ---------------------------------------------------------------------------------------------
    # 2. every fixed configuration on the same folds (so the reader can see the whole ladder, not
    #    just the winner) - reported, never used to pick the frozen model.
    # ---------------------------------------------------------------------------------------------
    res["fixed_ladder"] = {}
    for s in FEATURE_SETS:
        for spec in ({"feature_set": s, "algo": "logreg", "C": 0.3},
                     {"feature_set": s, "algo": "hgb", "max_depth": 3, "min_samples_leaf": 20}):
            p = plain_cv_oof(dev, spec, seed=a.seed, repeats=a.repeats)
            res["fixed_ladder"][spec_name(spec)] = {
                **prognosis_summary(y, p, threshold=0.5),
                **bootstrap_binary_auc_ci(y, p, n_boot=a.n_boot, seed=a.seed)}
    print("fixed ladder:", json.dumps({k: round(v["auc"], 4) for k, v in res["fixed_ladder"].items()}))

    # ---------------------------------------------------------------------------------------------
    # 3. freeze: run the SAME inner-CV selection rule once on the full dev set.
    # ---------------------------------------------------------------------------------------------
    sel = select_on_dev(dev, pools["imaging"], seed=a.seed, rule=a.rule)
    res["dev_selection"] = sel
    frozen_spec = sel["selected"]
    p_frozen = plain_cv_oof(dev, frozen_spec, seed=a.seed, repeats=a.repeats)
    p_clin = plain_cv_oof(dev, {"feature_set": "clinical_only", "algo": "logreg", "C": 0.3},
                          seed=a.seed, repeats=a.repeats)
    thr_youden = youden_threshold(y, p_frozen)
    res["frozen"] = {
        "spec": frozen_spec, "name": spec_name(frozen_spec),
        "features": FEATURE_SETS[frozen_spec["feature_set"]],
        "dev_cv": {**prognosis_summary(y, p_frozen, threshold=0.5),
                   **bootstrap_binary_auc_ci(y, p_frozen, n_boot=a.n_boot, seed=a.seed)},
        "dev_cv_at_youden": prognosis_summary(y, p_frozen, threshold=thr_youden),
        "threshold_youden_dev": thr_youden,
        "delta_vs_clinical_only": paired_auc_delta_ci(y, p_frozen, p_clin, n_boot=a.n_boot, seed=a.seed),
    }
    print("frozen:", spec_name(frozen_spec), "dev CV AUC", round(res["frozen"]["dev_cv"]["auc"], 4),
          "delta vs clinical", round(res["frozen"]["delta_vs_clinical_only"]["delta"], 4),
          res["frozen"]["delta_vs_clinical_only"]["delta_ci95"])

    # coefficients of the frozen model refitted on all of dev (interpretability, not a metric)
    model, feats = fit_frozen(dev, frozen_spec, a.seed)
    if frozen_spec["algo"] == "logreg":
        lr = model[-1]
        res["frozen"]["coefficients"] = dict(zip(feats, [float(v) for v in lr.coef_[0]]))
        res["frozen"]["intercept"] = float(lr.intercept_[0])

    # ---------------------------------------------------------------------------------------------
    # 4. auxiliary analyses (reported, never gates)
    # ---------------------------------------------------------------------------------------------
    ord_pred = ordinal_cv_pred(dev, frozen_spec["feature_set"], seed=a.seed, repeats=a.repeats)
    ord_clin = ordinal_cv_pred(dev, "clinical_only", seed=a.seed, repeats=a.repeats)
    res["ordinal"] = {"frozen_feature_set": ordinal_summary(dev["mrs"].to_numpy(float), ord_pred),
                      "clinical_only": ordinal_summary(dev["mrs"].to_numpy(float), ord_clin)}

    res["subgroups"] = {}
    nih = dev["nihss"].to_numpy(float)
    is_val = dev["split"].to_numpy() == "val"
    for name, m in [("nihss_lt_5", nih < 5), ("nihss_ge_5", nih >= 5),
                    ("seg_train", ~is_val), ("seg_val", is_val),
                    ("mrs_missing_free_all", np.ones(len(dev), bool))]:
        if m.sum() < 20 or len(set(y[m].tolist())) < 2:
            continue
        res["subgroups"][name] = {
            "n": int(m.sum()), "n_poor": int(y[m].sum()),
            "frozen_auc": binary_auc(y[m], p_frozen[m]),
            "clinical_only_auc": binary_auc(y[m], p_clin[m]),
            "delta": binary_auc(y[m], p_frozen[m]) - binary_auc(y[m], p_clin[m])}
    # mortality (mRS 6) is exploratory: n is small
    death = (dev["mrs"].to_numpy(float) >= 6).astype(int)
    res["subgroups"]["death_mrs6_exploratory"] = {
        "n_deaths": int(death.sum()),
        "frozen_auc_for_death": binary_auc(death, p_frozen),
        "clinical_only_auc_for_death": binary_auc(death, p_clin),
        "note": "the model was never trained on this label; exploratory only"}

    # ---------------------------------------------------------------------------------------------
    # 5. optional single final evaluation on the held-out test split (orchestrator).
    # ---------------------------------------------------------------------------------------------
    if a.final:
        frozen_cfg = json.load(open(a.frozen))
        spec = frozen_cfg["spec"]
        thr = frozen_cfg["threshold_youden_dev"]
        test = load_cohort(index, sp, a.cache, a.mask, a.pred_dir, which=("test",), allow_test=True)
        yt = test["y"].to_numpy(int)
        model, feats = fit_frozen(dev, spec, a.seed)
        pt = model.predict_proba(test[feats].to_numpy(float))[:, 1]
        clin_model, cfeats = fit_frozen(dev, {"feature_set": "clinical_only", "algo": "logreg", "C": 0.3}, a.seed)
        pc = clin_model.predict_proba(test[cfeats].to_numpy(float))[:, 1]
        res["test"] = {
            "n": int(len(test)), "frozen_name": frozen_cfg["name"],
            "primary": {**prognosis_summary(yt, pt, threshold=0.5),
                        **bootstrap_binary_auc_ci(yt, pt, n_boot=a.n_boot, seed=a.seed),
                        "at_frozen_youden": prognosis_summary(yt, pt, threshold=thr)},
            "clinical_only": {**prognosis_summary(yt, pc, threshold=0.5),
                              **bootstrap_binary_auc_ci(yt, pc, n_boot=a.n_boot, seed=a.seed)},
            "secondary_delta_vs_clinical": paired_auc_delta_ci(yt, pt, pc, n_boot=a.n_boot, seed=a.seed),
        }
        res["test_predictions"] = {"participant_id": test["participant_id"].tolist(),
                                   "y": yt.tolist(), "p_frozen": np.round(pt, 6).tolist(),
                                   "p_clinical_only": np.round(pc, 6).tolist()}
        print("TEST AUC", round(res["test"]["primary"]["auc"], 4), res["test"]["primary"]["ci95"])

    out = a.out or Path("results/prognosis") / f"prognosis_{a.mask}{'_final' if a.final else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1, default=float)
    print("wrote", out)

    # the frozen-model sidecar is written only by the primary (predicted-mask, dev-only) run
    if a.mask == "pred" and not a.final:
        a.frozen.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"spec": frozen_spec, "name": spec_name(frozen_spec), "features": feats,
                   "seed": a.seed, "threshold_primary": 0.5, "selection_rule": a.rule,
                   "threshold_youden_dev": thr_youden,
                   "mask_source": "pred", "pred_dir": str(a.pred_dir),
                   "dev_cv_auc": res["frozen"]["dev_cv"]["auc"],
                   "frozen_on": "2026-09-10 (dev only; test mRS not read)"},
                  open(a.frozen, "w"), indent=1)
        print("wrote", a.frozen)


if __name__ == "__main__":
    main()
