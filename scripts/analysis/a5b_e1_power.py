#!/usr/bin/env python3
"""A5b - power of the charter's ACTUAL E1 PASS rule, by simulation (no test data involved).

A6's `a6_power.py` computed P(AUC_hat > 0.55) under a normal approximation.  The charter's PASS rule is
stricter: point estimate > 0.55 AND percentile-bootstrap 95 % CI lower bound > 0.50 (otherwise
"inconclusive"; point <= 0.55 = FAIL).  Here the whole rule - including the 2,000-resample percentile
bootstrap - is simulated under a binormal score model for the assumed test cohort (LAA 35 / CE 22).

    PYTHONPATH=src .venv/bin/python scripts/analysis/a5b_e1_power.py
"""
from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.stats import norm


def auc_rank(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC with tie = 1/2 (independent of sklearn)."""
    from scipy.stats import rankdata

    s = np.concatenate([pos, neg])
    r = rankdata(s)
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def simulate(true_auc: float, n_pos: int, n_neg: int, n_sim: int, n_boot: int, seed: int, thr: float = 0.55) -> dict:
    rng = np.random.default_rng(seed)
    d = sqrt(2.0) * norm.ppf(true_auc)  # binormal: AUC = Phi(d / sqrt 2)
    verdict = {"PASS": 0, "INCONCLUSIVE": 0, "FAIL": 0}
    point_gt = 0
    for _ in range(n_sim):
        pos = rng.normal(d, 1.0, n_pos)
        neg = rng.normal(0.0, 1.0, n_neg)
        a = auc_rank(pos, neg)
        # percentile bootstrap over subjects (positives and negatives resampled jointly, as train_cls does:
        # i.i.d. subject resample, resamples lacking a class dropped)
        scores = np.concatenate([pos, neg])
        labels = np.concatenate([np.ones(n_pos, bool), np.zeros(n_neg, bool)])
        boots = []
        while len(boots) < n_boot:
            i = rng.integers(0, n_pos + n_neg, n_pos + n_neg)
            lb = labels[i]
            if lb.all() or not lb.any():
                continue
            boots.append(auc_rank(scores[i][lb], scores[i][~lb]))
        lo = float(np.percentile(boots, 2.5))
        point_gt += a > thr
        if a > thr and lo > 0.5:
            verdict["PASS"] += 1
        elif a > thr:
            verdict["INCONCLUSIVE"] += 1
        else:
            verdict["FAIL"] += 1
    return {"true_auc": true_auc, "n_pos": n_pos, "n_neg": n_neg, "n_sim": n_sim, "n_boot": n_boot,
            "p_point_gt_thr": point_gt / n_sim, **{k: v / n_sim for k, v in verdict.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pos", type=int, default=35)
    ap.add_argument("--n-neg", type=int, default=22)
    ap.add_argument("--n-sim", type=int, default=400)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("results/a5b/e1_power.json"))
    a = ap.parse_args()
    out = {"note": "cohort sizes are the dev-proportion ESTIMATE (A6); test labels were not read",
           "rule": "PASS = point > 0.55 and bootstrap CI lower > 0.50; INCONCLUSIVE = point > 0.55 but CI lower <= 0.50; FAIL = point <= 0.55",
           "scenarios": {}}
    for name, auc in [("dev_estimate_0.655", 0.655), ("null_0.5", 0.5), ("0.60", 0.60), ("0.70", 0.70)]:
        out["scenarios"][name] = simulate(auc, a.n_pos, a.n_neg, a.n_sim, a.n_boot, seed=2026)
        print(name, json.dumps(out["scenarios"][name]))
    # normal-approximation cross-check for the point gate only (what a6_power.py computed)
    se = sqrt((0.655 * (1 - 0.655) + (35 - 1) * (0.655 / (2 - 0.655) - 0.655**2) + (22 - 1) * (2 * 0.655**2 / (1 + 0.655) - 0.655**2)) / (35 * 22))
    out["normal_approx_point_gate"] = {"se_hanley_mcneil": se, "p_point_gt_0.55_at_0.655": float(1 - norm.cdf((0.55 - 0.655) / se)),
                                       "p_pass_if_ci_is_pm_1.96se": float(1 - norm.cdf((0.5 + 1.96 * se - 0.655) / se))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["normal_approx_point_gate"]))


if __name__ == "__main__":
    main()
