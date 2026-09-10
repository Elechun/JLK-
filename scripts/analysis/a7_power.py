#!/usr/bin/env python3
"""A7 - what can the 82 test subjects with a discharge mRS actually decide?

Sizes the pre-registered thresholds of the prognosis extension BEFORE the test labels are read.

  * primary   : absolute AUC of the frozen model.  Analytic SE from Hanley & McNeil (1982), then a
                simulation of the ACTUAL decision rule (point estimate >= threshold AND the lower end
                of the 95 % subject-bootstrap CI > 0.50), because E1 of the etiology task showed that
                a CI-based clause can turn a 93 %-power point gate into a 54 %-PASS rule (A5b).
  * secondary : the imaging increment (delta AUC vs the clinical-only model).  The dev paired-bootstrap
                SD is rescaled by sqrt(n_dev / n_test) to show that n = 82 has no power for it - which
                is why it is reported, not gated.

The test outcome balance is ESTIMATED from the dev proportion (the split is stratified on etiology and
lesion-size band, not on mRS, so this is an assumption, not a measurement); a sensitivity band over
plausible splits is printed next to it.  No test mRS value is read anywhere in this script.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from strokeai.metrics import binary_auc, hanley_mcneil_se  # noqa: E402


def norm_cdf(z: float) -> float:
    return 0.5 * (1 + erf(z / sqrt(2)))


def norm_ppf(p: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(p))


def _auc_rows(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Vectorised ROC-AUC for a stack of (score, label) rows.

    Mann-Whitney form with mid-ranks for ties:  AUC = (sum of positive ranks - n1(n1+1)/2) / (n1 n2).
    This is an OPTIMISATION of `strokeai.metrics.binary_auc` for the bootstrap inner loop, not a second
    definition of the metric: `_check_fast_auc` asserts the two agree to 1e-12 before it is used.
    """
    from scipy.stats import rankdata

    r = rankdata(scores, axis=1)
    n1 = labels.sum(1).astype(float)
    n2 = labels.shape[1] - n1
    pos_rank = (r * labels).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (pos_rank - n1 * (n1 + 1) / 2) / (n1 * n2)


def _check_fast_auc(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(20):
        y = rng.integers(0, 2, 82)
        if y.sum() in (0, 82):
            continue
        s = rng.normal(size=82).round(1)  # deliberate ties
        fast = _auc_rows(s[None, :], y[None, :].astype(float))[0]
        assert abs(fast - binary_auc(y, s)) < 1e-12, (fast, binary_auc(y, s))


def simulate_rule(true_auc: float, n1: int, n2: int, thr: float, n_sim: int, n_boot: int,
                  seed: int) -> dict:
    """Binormal simulation of the pre-registered decision rule.

    Positives ~ N(d, 1), negatives ~ N(0, 1) with d = sqrt(2) * Phi^-1(AUC) reproduces `true_auc` in
    expectation.  For each simulated test set the empirical AUC and a percentile bootstrap CI are
    computed and the rule (point estimate >= thr AND CI lower > 0.5) is applied.  Bootstrap resamples
    that lose a class are dropped, matching `metrics.bootstrap_binary_auc_ci`.
    """
    rng = np.random.default_rng(seed)
    d = sqrt(2) * norm_ppf(true_auc) if 0 < true_auc < 1 else 0.0
    y = np.r_[np.ones(n1, float), np.zeros(n2, float)]
    n = n1 + n2
    counts = {"PASS": 0, "INCONCLUSIVE": 0, "FAIL": 0}
    aucs, lows = [], []
    for _ in range(n_sim):
        s = np.r_[rng.normal(d, 1.0, n1), rng.normal(0.0, 1.0, n2)]
        a = float(_auc_rows(s[None, :], y[None, :])[0])
        idx = rng.integers(0, n, (n_boot, n))
        ys, ss = y[idx], s[idx]
        keep = (ys.sum(1) > 0) & (ys.sum(1) < n)
        b = _auc_rows(ss[keep], ys[keep])
        lo = float(np.percentile(b, 2.5))
        aucs.append(a)
        lows.append(lo)
        if a >= thr and lo > 0.5:
            counts["PASS"] += 1
        elif a >= thr:
            counts["INCONCLUSIVE"] += 1
        else:
            counts["FAIL"] += 1
    return {"true_auc": true_auc, "n1": n1, "n2": n2, "threshold": thr, "n_sim": n_sim,
            "n_boot": n_boot, "rates": {k: v / n_sim for k, v in counts.items()},
            "mean_auc_hat": float(np.mean(aucs)), "sd_auc_hat": float(np.std(aucs, ddof=1)),
            "mean_ci_lower": float(np.mean(lows))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=Path, default=Path("results/prognosis/prognosis_pred.json"))
    ap.add_argument("--n-test", type=int, default=82, help="test subjects WITH a discharge mRS (metadata count)")
    ap.add_argument("--n-dev", type=int, default=538)
    ap.add_argument("--dev-poor-frac", type=float, default=273 / 538)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.60, 0.65, 0.68, 0.70, 0.72, 0.75])
    ap.add_argument("--n-sim", type=int, default=2000)
    ap.add_argument("--sim-boot", type=int, default=1000)
    ap.add_argument("--sim-thr", type=float, default=0.70)
    ap.add_argument("--nested-key", default="imaging:argmax",
                    help="which nested-CV selection procedure is the frozen one")
    ap.add_argument("--generalisation-band", type=float, default=0.10,
                    help="pre-registered |AUC_test - AUC_devCV| band (criterion P1b)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=Path("results/prognosis/power.json"))
    a = ap.parse_args()

    _check_fast_auc()
    dev = json.load(open(a.dev))
    nested = dev["nested_cv"]["summary"]
    key = a.nested_key if a.nested_key in nested else list(nested)[0]
    auc_dev = float(dev["frozen"]["dev_cv"]["auc"])
    auc_nested = float(nested[key]["auc"])
    auc_clin = float(nested["clinical_only"]["auc"])

    n1 = int(round(a.n_test * a.dev_poor_frac))  # poor outcomes (positives)
    n2 = a.n_test - n1
    out = {"note": "test outcome balance is ESTIMATED from the dev proportion; no test mRS value was read",
           "n_test_with_mrs": a.n_test, "assumed_n_poor": n1, "assumed_n_good": n2,
           "dev_estimates": {"nested_cv_auc_" + key: auc_nested,
                             "frozen_plain_cv_auc": auc_dev,
                             "clinical_only_cv_auc": auc_clin,
                             "which_is_used_as_the_true_value": "nested CV (the plain CV of the frozen "
                                                                "configuration is optimistic - A5a item 8)"},
           "analytic": {}, "balance_sensitivity": {}, "simulation": {}, "secondary_delta": {}}

    # ---- analytic: point-estimate gate ---------------------------------------------------------
    se_true = hanley_mcneil_se(auc_nested, n1, n2)
    se_chance = hanley_mcneil_se(0.5, n1, n2)
    out["analytic"]["se_at_dev_auc"] = se_true
    out["analytic"]["se_at_chance"] = se_chance
    out["analytic"]["expected_ci95_halfwidth"] = 1.96 * se_true
    for thr in a.thresholds:
        out["analytic"][f"auc >= {thr}"] = {
            "power_if_true_equals_nested_dev": 1 - norm_cdf((thr - auc_nested) / se_true),
            "power_if_true_is_0.75": 1 - norm_cdf((thr - 0.75) / hanley_mcneil_se(0.75, n1, n2)),
            "power_if_true_is_clinical_only_dev": 1 - norm_cdf((thr - auc_clin) / hanley_mcneil_se(auc_clin, n1, n2)),
            "false_pass_if_true_is_chance": 1 - norm_cdf((thr - 0.5) / se_chance),
        }

    # ---- how much does the assumed balance matter? ----------------------------------------------
    for frac in (0.40, 0.45, 0.50, 0.5074, 0.55, 0.60):
        k1 = int(round(a.n_test * frac))
        k2 = a.n_test - k1
        out["balance_sensitivity"][f"poor_frac={frac}"] = {
            "n_poor": k1, "n_good": k2, "se": hanley_mcneil_se(auc_nested, k1, k2),
            "power_at_0.70": 1 - norm_cdf((0.70 - auc_nested) / hanley_mcneil_se(auc_nested, k1, k2))}

    # ---- simulation of the actual PASS / INCONCLUSIVE / FAIL rule --------------------------------
    for true_auc in (auc_nested, 0.75, auc_clin, 0.65, 0.50):
        r = simulate_rule(true_auc, n1, n2, a.sim_thr, a.n_sim, a.sim_boot, a.seed)
        out["simulation"][f"true_auc={round(true_auc, 4)}"] = r
        print(f"true AUC {true_auc:.4f} -> {r['rates']}  (sd of AUC-hat {r['sd_auc_hat']:.4f})")

    # ---- P1b: the generalisation band |AUC_test - AUC_devCV| <= band ----------------------------
    # Modelled on charter criterion S1b: the absolute-AUC gate has a large margin and therefore little
    # discriminating power, so a second criterion bounds how far the test value may drift from dev.
    sd_dev = (nested[key]["ci95"][1] - nested[key]["ci95"][0]) / (2 * 1.96)
    sd_diff = sqrt(se_true**2 + sd_dev**2)
    out["generalisation_P1b"] = {
        "dev_cv_auc": auc_nested, "dev_bootstrap_sd": sd_dev, "test_se_est": se_true,
        "sd_of_the_difference": sd_diff, "band": a.generalisation_band,
        "band_in_sd": a.generalisation_band / sd_diff,
        "false_alarm_if_true_difference_is_zero": 2 * (1 - norm_cdf(a.generalisation_band / sd_diff)),
        # If the TRUE test-time AUC is d, how often does the band still pass (i.e. how often do we
        # fail to notice that the dev estimate did not transport)?  power = 1 - pass_rate.
        "pass_rate_if_true_auc_is": {
            str(d): 1 - norm_cdf((auc_nested - a.generalisation_band - d) / se_true)
            for d in (0.70, 0.72, 0.75, 0.767)},
        "power_to_detect_true_auc_of": {
            str(d): norm_cdf((auc_nested - a.generalisation_band - d) / se_true)
            for d in (0.70, 0.72, 0.75, 0.767)},
        "comment": "one-sided reading: a test AUC below dev - band means the dev estimate did not transport",
    }
    print("P1b:", json.dumps(out["generalisation_P1b"], indent=1))

    # ---- secondary: the imaging increment has no power at n = 82 ---------------------------------
    dkey = f"{key}-clinical_only"
    delta = dev["nested_cv"]["deltas"][dkey]
    sd_dev = float(delta["delta_boot_sd"])
    sd_test = sd_dev * sqrt(a.n_dev / a.n_test)
    out["secondary_delta"] = {
        "dev_delta": delta["delta"], "dev_delta_ci95": delta["delta_ci95"],
        "dev_delta_boot_sd": sd_dev, "n_dev": a.n_dev,
        "projected_test_delta_sd": sd_test,
        "projected_test_ci95_halfwidth": 1.96 * sd_test,
        "power_for_ci_lower_gt_0_at_dev_delta": 1 - norm_cdf((1.96 * sd_test - delta["delta"]) / sd_test),
        "n_needed_for_80pct_power": int(np.ceil(((1.96 + 0.84) * sd_dev * sqrt(a.n_dev) / delta["delta"]) ** 2))
        if delta["delta"] > 0 else None,
        "verdict": "reported with a CI, NEVER gated: a CI-lower-bound > 0 rule on n = 82 would fail "
                   "most of the time even if the dev increment is exactly right",
    }
    print("secondary delta:", json.dumps(out["secondary_delta"], indent=1))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
