"""A7 guards for the discharge-mRS prognosis extension (2026-09-10).

Two jobs:
  1. the held-out discipline - no code path may build a prognosis feature or read an mRS value for a
     test subject without the explicit `allow_test` / `--final` opt-in;
  2. the new metrics in `strokeai.metrics` are checked against independent formulations, not against
     themselves (the A5a house rule).
"""
import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strokeai import prognosis as P
from strokeai import metrics as M

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------------
# held-out discipline
# ---------------------------------------------------------------------------------------------
def _argparse_default(script: Path, option: str):
    tree = ast.parse(script.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == option:
                for kw in node.keywords:
                    if kw.arg == "default":
                        return ast.literal_eval(kw.value)
                return None
    raise AssertionError(f"{option} not found in {script}")


def test_train_prognosis_cli_defaults_to_train_val():
    assert _argparse_default(ROOT / "scripts/train_prognosis.py", "--which") == ["train", "val"]
    assert _argparse_default(ROOT / "scripts/train_prognosis.py", "--final") is None  # store_true -> False


def test_load_cohort_refuses_the_test_split():
    splits = {"train": ["a"], "val": ["b"], "test": ["c"]}
    with pytest.raises(PermissionError, match="held-out test split"):
        P.load_cohort(pd.DataFrame({"participant_id": []}), splits, Path("."), which=("test",))
    with pytest.raises(PermissionError):
        P.load_cohort(pd.DataFrame({"participant_id": []}), splits, Path("."), which=("train", "test"))


def test_cli_rejects_which_test_even_with_final():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/train_prognosis.py"),
                        "--which", "test", "--final"], capture_output=True, text=True)
    assert r.returncode != 0
    assert "never accepts 'test'" in (r.stderr + r.stdout)


def test_no_forbidden_input_in_any_feature_set():
    """mRS itself, the constant `acuteischaemicstroke`, and the scanner/protocol variables are banned
    as model inputs (charter, extension pre-registration)."""
    banned = {"gs_rankin_6isdeath", "mrs", "y", "acuteischaemicstroke", "trace_Manufacturer",
              "trace_MagneticFieldStrength", "manufacturer", "field_strength", "etiology", "etiology_code"}
    for name, feats in P.FEATURE_SETS.items():
        assert not (set(feats) & banned), f"{name} contains a forbidden input"


def test_poor_outcome_definition_is_the_standard_dichotomy():
    assert P.POOR_CUTOFF == 3
    assert [int(v >= P.POOR_CUTOFF) for v in range(7)] == [0, 0, 0, 1, 1, 1, 1]


# ---------------------------------------------------------------------------------------------
# selection rules
# ---------------------------------------------------------------------------------------------
def test_one_se_rule_prefers_the_simpler_model_within_one_standard_error():
    simple = {"feature_set": "clinical_volume", "algo": "logreg", "C": 0.1}
    complexer = {"feature_set": "clinical_lesion_all", "algo": "hgb", "max_depth": 3, "min_samples_leaf": 20}
    scored = [{"spec": simple, "mean": 0.80, "se": 0.02},
              {"spec": complexer, "mean": 0.815, "se": 0.02}]
    assert P.apply_rule(scored, "argmax")["spec"] is complexer
    assert P.apply_rule(scored, "one_se")["spec"] is simple
    # ... but not when the gap is larger than one SE of the best
    scored[1]["mean"] = 0.90
    assert P.apply_rule(scored, "one_se")["spec"] is complexer


def test_candidate_grid_covers_every_requested_set():
    g = P.candidate_grid(["clinical_volume", "clinical_only"])
    assert {s["feature_set"] for s in g} == {"clinical_volume", "clinical_only"}
    assert {s["algo"] for s in g} == {"logreg", "hgb"}
    assert len({P.spec_name(s) for s in g}) == len(g)  # no duplicate configurations


def test_baselines_and_imaging_sets_are_disjoint_and_known():
    assert set(P.BASELINES).isdisjoint(P.IMAGING_SETS)
    assert set(P.BASELINES) | set(P.IMAGING_SETS) == set(P.FEATURE_SETS)
    for s in P.IMAGING_SETS:  # every imaging candidate really contains a lesion feature
        assert set(P.FEATURE_SETS[s]) & set(P.FEATURE_NAMES if hasattr(P, "FEATURE_NAMES") else
                                            __import__("strokeai.features", fromlist=["x"]).FEATURE_NAMES)


# ---------------------------------------------------------------------------------------------
# metrics, checked against independent formulations
# ---------------------------------------------------------------------------------------------
def _auc_by_rank_pairs(y, s):
    """Independent AUC: fraction of (positive, negative) pairs ranked correctly, ties count 1/2."""
    pos = np.asarray(s)[np.asarray(y) == 1]
    neg = np.asarray(s)[np.asarray(y) == 0]
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / (len(pos) * len(neg)))


def test_binary_auc_matches_a_rank_pair_count():
    rng = np.random.default_rng(7)
    for _ in range(20):
        y = rng.integers(0, 2, 60)
        if y.sum() in (0, 60):
            continue
        s = rng.normal(size=60).round(1)  # deliberate ties
        assert M.binary_auc(y, s) == pytest.approx(_auc_by_rank_pairs(y, s), abs=1e-12)


def test_binary_auc_is_nan_without_both_classes():
    assert np.isnan(M.binary_auc(np.ones(10, int), np.arange(10.0)))
    assert np.isnan(M.binary_auc(np.zeros(10, int), np.arange(10.0)))


def test_brier_and_skill_score():
    y = np.array([1, 1, 0, 0])
    assert M.brier_score(y, np.array([1.0, 1.0, 0.0, 0.0])) == 0.0
    assert M.brier_score(y, np.full(4, 0.5)) == pytest.approx(0.25)
    # the prevalence-only forecast has zero skill by construction
    assert M.brier_skill_score(y, np.full(4, y.mean())) == pytest.approx(0.0)
    assert M.brier_skill_score(y, np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_calibration_slope_is_one_on_self_consistent_probabilities():
    """Draw the outcome FROM the predicted probability: the refit slope must be ~1, intercept ~0."""
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    c = M.calibration_slope_intercept(y, p)
    assert c["slope"] == pytest.approx(1.0, abs=0.06)
    assert c["intercept"] == pytest.approx(0.0, abs=0.06)
    assert c["calibration_in_the_large"] == pytest.approx(0.0, abs=0.02)
    # squashing the probabilities towards 0.5 makes them too timid -> slope > 1
    squashed = 0.5 + (p - 0.5) / 3
    assert M.calibration_slope_intercept(y, squashed)["slope"] > 1.5


def test_calibration_bins_partition_every_subject():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=53)
    y = rng.integers(0, 2, 53)
    bins = M.calibration_bins(y, p, n_bins=5)
    assert sum(b["n"] for b in bins) == 53
    assert [b["mean_predicted"] for b in bins] == sorted(b["mean_predicted"] for b in bins)


def test_threshold_metrics_against_a_hand_counted_example():
    y = np.array([1, 1, 1, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.4, 0.6, 0.2, 0.1])
    r = M.binary_threshold_metrics(y, p, 0.5)
    assert r["confusion"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}
    assert r["sensitivity"] == pytest.approx(2 / 3)
    assert r["specificity"] == pytest.approx(2 / 3)
    assert r["ppv"] == pytest.approx(2 / 3)
    assert r["npv"] == pytest.approx(2 / 3)
    assert r["accuracy"] == pytest.approx(4 / 6)


def test_hanley_mcneil_se_reduces_to_the_mann_whitney_null_se_at_auc_0_5():
    """At AUC = 0.5 the Hanley-McNeil expression collapses algebraically to sqrt((n1+n2+1)/(12 n1 n2)),
    the exact null SD of the Mann-Whitney statistic. Independent identity, not a re-run of the code."""
    for n1, n2 in [(41, 41), (10, 90), (35, 47), (100, 100)]:
        expect = np.sqrt((n1 + n2 + 1) / (12 * n1 * n2))
        assert M.hanley_mcneil_se(0.5, n1, n2) == pytest.approx(expect, rel=1e-12)


def test_hanley_mcneil_se_shrinks_with_n_and_with_a_higher_auc():
    assert M.hanley_mcneil_se(0.8, 41, 41) > M.hanley_mcneil_se(0.8, 410, 410)
    assert M.hanley_mcneil_se(0.9, 41, 41) < M.hanley_mcneil_se(0.7, 41, 41)
    assert M.hanley_mcneil_se(0.8, 41, 41) == pytest.approx(0.0492, abs=5e-4)  # charter power calc


def test_bootstrap_auc_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, 200)
    p = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, 200), 0.01, 0.99)
    r = M.bootstrap_binary_auc_ci(y, p, n_boot=500, seed=5)
    assert r["ci95"][0] < r["auc"] < r["ci95"][1]
    assert r["n_boot_effective"] == 500
    assert M.bootstrap_binary_auc_ci(y, p, n_boot=500, seed=5) == r  # deterministic given the seed


def test_paired_auc_delta_is_the_difference_of_the_two_aucs():
    rng = np.random.default_rng(12)
    y = rng.integers(0, 2, 150)
    good = y + rng.normal(0, 0.5, 150)
    weak = y + rng.normal(0, 2.0, 150)
    r = M.paired_auc_delta_ci(y, good, weak, n_boot=400, seed=1)
    assert r["delta"] == pytest.approx(M.binary_auc(y, good) - M.binary_auc(y, weak))
    assert r["delta"] > 0 and r["p_gt_0"] > 0.9
    # a score compared against itself has delta exactly 0 in every resample
    same = M.paired_auc_delta_ci(y, good, good, n_boot=200, seed=1)
    assert same["delta"] == 0.0 and same["delta_ci95"] == [0.0, 0.0]


def test_ordinal_summary_on_a_known_vector():
    t = np.array([0, 1, 2, 3, 4, 5, 6], float)
    r = M.ordinal_summary(t, t)
    assert r["spearman_rho"] == pytest.approx(1.0)
    assert r["mae"] == 0.0 and r["within_1_rate"] == 1.0
    r2 = M.ordinal_summary(t, t + 1)
    assert r2["mae"] == pytest.approx(1.0) and r2["within_1_rate"] == 1.0
    assert r2["spearman_rho"] == pytest.approx(1.0)


def test_prognosis_summary_reports_every_required_field():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 120)
    p = np.clip(0.5 + 0.25 * (y - 0.5) + rng.normal(0, 0.2, 120), 0.01, 0.99)
    s = M.prognosis_summary(y, p, threshold=0.5)
    for k in ("n", "n_poor", "n_good", "auc", "brier", "brier_skill", "calibration",
              "calibration_bins", "operating_point"):
        assert k in s
    assert s["n_poor"] + s["n_good"] == s["n"] == 120
    assert set(s["operating_point"]["confusion"]) == {"tp", "fp", "fn", "tn"}
    assert sum(s["operating_point"]["confusion"].values()) == 120


# ---------------------------------------------------------------------------------------------
# end-to-end on a tiny synthetic cohort (no real data needed)
# ---------------------------------------------------------------------------------------------
def _tiny_cohort(n=80, seed=0):
    rng = np.random.default_rng(seed)
    from strokeai.features import FEATURE_NAMES

    df = pd.DataFrame({f: rng.normal(size=n) for f in FEATURE_NAMES})
    df["age"] = rng.uniform(40, 90, n)
    df["sex_male"] = rng.integers(0, 2, n).astype(float)
    df["nihss"] = rng.uniform(0, 30, n)
    df["priorstroke"] = rng.integers(0, 2, n).astype(float)
    lin = 0.12 * (df["nihss"] - 10) + 0.8 * df["log_volume_ml"]
    df["y"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-lin))).astype(int)
    df["mrs"] = df["y"] * 3 + rng.integers(0, 3, n)
    df["participant_id"] = [f"sub-{i}" for i in range(n)]
    df["split"] = "train"
    return df


def test_nested_cv_runs_and_reports_one_selection_per_outer_fold():
    df = _tiny_cohort()
    cands = P.candidate_grid(["clinical_volume", "clinical_only"])
    out = P.nested_cv(df, cands, ["clinical_only"], seed=1, n_outer=3, n_inner=3, repeats=2,
                      rules=("argmax", "one_se"),
                      pools={"imaging": [c for c in cands if c["feature_set"] == "clinical_volume"],
                             "unrestricted": cands})
    assert len(out["selected_per_fold"]) == 3 * 2
    for key in ("imaging:argmax", "imaging:one_se", "unrestricted:argmax", "unrestricted:one_se",
                "clinical_only"):
        assert len(out["oof"][key]) == len(df)
        assert np.all((np.asarray(out["oof"][key]) >= 0) & (np.asarray(out["oof"][key]) <= 1))
    # the imaging pool can only ever select an imaging configuration
    assert all(k.startswith("clinical_volume") for k in out["selection_frequency"]["imaging:argmax"])


def test_select_on_dev_returns_both_rules_and_a_valid_spec():
    df = _tiny_cohort(seed=4)
    cands = P.candidate_grid(["clinical_volume"])
    sel = P.select_on_dev(df, cands, seed=1, n_inner=3, rule="one_se")
    assert sel["selected"] in cands
    assert set(sel["picks"]) == {"argmax", "one_se"}
    assert sel["ranking"][0]["inner_auc"] >= sel["ranking"][-1]["inner_auc"]
    model, feats = P.fit_frozen(df, sel["selected"], seed=1)
    assert feats == P.FEATURE_SETS[sel["selected"]["feature_set"]]
    assert model.predict_proba(df[feats].to_numpy(float)).shape == (len(df), 2)


@pytest.mark.skipif(not (ROOT / "results/prognosis/frozen_model.json").exists(),
                    reason="the frozen model sidecar has not been produced in this checkout")
def test_frozen_model_sidecar_is_self_consistent():
    f = json.load(open(ROOT / "results/prognosis/frozen_model.json"))
    assert f["features"] == P.FEATURE_SETS[f["spec"]["feature_set"]]
    assert f["spec"]["feature_set"] in P.IMAGING_SETS, "the frozen primary model must use imaging"
    assert f["mask_source"] == "pred", "the primary model is the deployment-like predicted-mask one"
    assert 0.0 < f["threshold_youden_dev"] < 1.0
