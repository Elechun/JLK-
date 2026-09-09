#!/usr/bin/env python3
"""A4b - aggregate the small-lesion experiments and apply the PRE-REGISTERED decision rule.

Decision rule, fixed before any variant had finished training (2026-09-09, see
docs/agents/A4b_small_lesion_report.md s.2).  The primary quantity is the val Dice of the
[0, 2) mL band (n = 60 subjects); every variant is compared with the shipped config at the SAME three
seeds {2026, 7, 77}:

  sd_small = SD of the baseline small-lesion Dice over its seeds (>= 3 seeds; 5 are trained)
  effect   = mean(variant) - mean(baseline)

  "효과 있음"  (net gain)  : effect >= 2 * sd_small
                            AND mean overall val Dice >= baseline - 0.007   (A4's seed noise)
                            AND mean val ICC          >= baseline - 2 * sd_icc
  "소병변만 개선"          : effect >= 2 * sd_small but a guard rail above is violated
  "효과 없음"              : effect <  2 * sd_small

A 2*SD bar on a 3-seed mean is deliberately conservative (SE of the mean = SD/sqrt(3), so the bar is
~3.5 SE): with n = 60 subjects in the band and a seed spread of 0.54-0.57 measured across the existing
runs, anything smaller cannot be told from noise.

    PYTHONPATH=src .venv/bin/python scripts/analysis/small_lesion_report.py

Reads results/small_lesion/runs/*.json, writes results/small_lesion/summary.json and prints the table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = [2026, 7, 77]
DICE_NOISE = 0.007  # A4: seed-to-seed spread of the overall val Dice
BAND_HI = 2.0       # the small-lesion band, mL


def _band_dice(run: Path, hi: float = BAND_HI) -> dict[str, float]:
    """{subject id -> Dice} for the GT-positive subjects below `hi` mL, from a run's val_best.json."""
    per = json.load(open(run / "val_best.json"))["per_subject"]
    return {r["sid"]: r["dice"] for r in per if r["gt_pos"] and r["gt_ml"] < hi}


def paired_subject_bootstrap(var_runs: list[dict], base_runs: list[dict], n_boot: int = 2000,
                             seed: int = 0) -> dict:
    """Seed-matched, subject-level bootstrap CI of the band-Dice difference.

    The seed SD tells us how much a re-run moves the number; this tells us how much the *val split*
    does.  For every seed present in both arms we take the per-subject Dice difference, average the
    differences over seeds, and resample the 60 subjects.  A CI that excludes 0 means the difference
    is not an artefact of which 60 subjects happen to be in val.
    """
    seeds = sorted({r["seed"] for r in var_runs} & {r["seed"] for r in base_runs})
    if not seeds:
        return {}
    vr = {r["seed"]: _band_dice(Path(r["run"])) for r in var_runs if r["seed"] in seeds}
    br = {r["seed"]: _band_dice(Path(r["run"])) for r in base_runs if r["seed"] in seeds}
    sids = sorted(set.intersection(*[set(d) for d in list(vr.values()) + list(br.values())]))
    diff = np.array([np.mean([vr[s][sid] - br[s][sid] for s in seeds]) for sid in sids])
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    return {"n_seeds_paired": len(seeds), "seeds": seeds, "n_subjects": len(sids),
            "mean_diff": float(diff.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "frac_bootstrap_positive": float((boots > 0).mean()),
            "n_subjects_improved": int((diff > 0).sum()), "n_subjects_worse": int((diff < 0).sum())}


def agg(rows: list[dict], key: str) -> tuple[float, float]:
    v = np.array([r[key] for r in rows], float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("results/small_lesion/runs"))
    ap.add_argument("--out", type=Path, default=Path("results/small_lesion/summary.json"))
    ap.add_argument("--markdown", action="store_true", help="also print the report table as markdown")
    a = ap.parse_args()

    runs = [json.load(open(f)) for f in sorted(a.dir.glob("*.json"))]
    by_variant: dict[str, list[dict]] = {}
    for r in runs:
        by_variant.setdefault(r["variant"], []).append(r)
    for v in by_variant.values():
        v.sort(key=lambda r: r["seed"])

    base_all = by_variant.get("base", []) + by_variant.get("base_extra", [])
    if not base_all:
        raise SystemExit("no baseline runs in " + str(a.dir))
    base_3 = [r for r in by_variant.get("base", []) if r["seed"] in SEEDS]
    sd_small = float(np.std([r["small_dice"] for r in base_all], ddof=1))
    sd_icc = float(np.std([r["val_icc"] for r in base_all], ddof=1))
    sd_dice = float(np.std([r["val_dice"] for r in base_all], ddof=1))
    b_small, b_dice, b_icc = (np.mean([r[k] for r in base_3]) for k in ("small_dice", "val_dice", "val_icc"))

    out = {"decision_rule": {"primary": "val Dice, GT volume band [0,2) mL (n=60)",
                             "seeds": SEEDS, "bar": "effect >= 2 * sd_small",
                             "guard_overall_dice": f">= baseline - {DICE_NOISE}",
                             "guard_icc": ">= baseline - 2 * sd_icc"},
           "baseline_seed_noise": {"n_runs": len(base_all), "seeds": sorted(r["seed"] for r in base_all),
                                   "small_dice_sd": sd_small, "small_dice_mean": float(np.mean([r["small_dice"] for r in base_all])),
                                   "small_dice_values": {str(r["seed"]): r["small_dice"] for r in sorted(base_all, key=lambda r: r["seed"])},
                                   "val_dice_sd": sd_dice, "val_icc_sd": sd_icc,
                                   "bar_2sd": 2 * sd_small},
           "baseline_3seed": {"small_dice": float(b_small), "val_dice": float(b_dice), "val_icc": float(b_icc)},
           "variants": {}}

    for name, rows in sorted(by_variant.items()):
        rows3 = [r for r in rows if r["seed"] in SEEDS]
        if name in ("base", "base_extra") or len(rows3) == 0:
            continue
        m_small, s_small = agg(rows3, "small_dice")
        m_dice, _ = agg(rows3, "val_dice")
        m_icc, _ = agg(rows3, "val_icc")
        m_det, _ = agg(rows3, "small_detect")
        m_sens, _ = agg(rows3, "val_sens")
        eff = m_small - b_small
        paired = [r["small_dice"] - next(b["small_dice"] for b in base_3 if b["seed"] == r["seed"]) for r in rows3]
        ok_dice = m_dice >= b_dice - DICE_NOISE
        ok_icc = m_icc >= b_icc - 2 * sd_icc
        if eff >= 2 * sd_small:
            verdict = "효과 있음(순이득)" if (ok_dice and ok_icc) else "소병변만 개선(순이득 아님)"
        else:
            verdict = "효과 없음"
        out["variants"][name] = {
            "n_seeds": len(rows3), "seeds": [r["seed"] for r in rows3],
            "small_dice_by_seed": {str(r["seed"]): r["small_dice"] for r in rows3},
            "small_dice_mean": m_small, "small_dice_sd": s_small, "effect_vs_base": eff,
            "effect_in_sd": eff / sd_small if sd_small else float("nan"),
            "paired_delta_by_seed": paired, "paired_delta_mean": float(np.mean(paired)),
            "small_detect_mean": m_det, "small_median_abs_pct_err_mean": agg(rows3, "small_median_abs_pct_err")[0],
            "val_dice_mean": m_dice, "val_dice_delta": m_dice - b_dice, "guard_overall_dice_ok": bool(ok_dice),
            "val_icc_mean": m_icc, "val_icc_delta": m_icc - b_icc, "guard_icc_ok": bool(ok_icc),
            "val_sens_mean": m_sens,
            "bands_dice_mean": {band: float(np.mean([r["bands"][band]["dice"] for r in rows3]))
                                for band in rows3[0]["bands"]},
            "verdict": verdict,
        }
        # SUPPLEMENTARY (not the pre-registered test): every seed that exists for this variant against
        # every baseline seed.  Reported separately so the registered 3-seed judgement stays intact.
        if len(rows) > len(rows3):
            m_all = float(np.mean([r["small_dice"] for r in rows]))
            out["variants"][name]["supplementary_all_seeds"] = {
                "n_seeds": len(rows), "seeds": [r["seed"] for r in rows],
                "small_dice_by_seed": {str(r["seed"]): r["small_dice"] for r in rows},
                "small_dice_mean": m_all,
                "effect_vs_base_all_seeds": m_all - float(np.mean([r["small_dice"] for r in base_all])),
                "effect_in_sd": (m_all - float(np.mean([r["small_dice"] for r in base_all]))) / sd_small,
                "val_dice_mean": float(np.mean([r["val_dice"] for r in rows])),
                "val_dice_delta": float(np.mean([r["val_dice"] for r in rows])
                                        - np.mean([r["val_dice"] for r in base_all])),
                "val_icc_mean": float(np.mean([r["val_icc"] for r in rows])),
                "val_icc_delta": float(np.mean([r["val_icc"] for r in rows])
                                       - np.mean([r["val_icc"] for r in base_all])),
                "small_detect_mean": float(np.mean([r["small_detect"] for r in rows])),
                "n_seeds_above_every_baseline_seed": int(sum(
                    r["small_dice"] > max(b["small_dice"] for b in base_all) for r in rows)),
                "paired_subject_bootstrap": paired_subject_bootstrap(rows, base_all),
            }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)

    print(f"baseline seed noise (n={len(base_all)}): small Dice SD = {sd_small:.4f}  -> bar = {2 * sd_small:+.4f}"
          f"   (overall Dice SD {sd_dice:.4f}, ICC SD {sd_icc:.4f})")
    print(f"baseline 3-seed mean: small {b_small:.4f}  overall {b_dice:.4f}  ICC {b_icc:.4f}\n")
    hdr = f"{'variant':13s} {'small(seedwise)':26s} {'mean':>7} {'Δ':>8} {'Δ/SD':>6} {'det':>6} {'dice':>7} {'Δdice':>7} {'ICC':>7} {'verdict'}"
    print(hdr)
    for name, v in out["variants"].items():
        sw = "/".join(f"{x:.3f}" for x in v["small_dice_by_seed"].values())
        print(f"{name:13s} {sw:26s} {v['small_dice_mean']:7.4f} {v['effect_vs_base']:+8.4f} "
              f"{v['effect_in_sd']:6.2f} {v['small_detect_mean']:6.3f} {v['val_dice_mean']:7.4f} "
              f"{v['val_dice_delta']:+7.4f} {v['val_icc_mean']:7.4f} {v['verdict']}")

    if a.markdown:
        print("\n| variant | 소병변 Dice (2026/7/77) | 평균 | Δ vs 기준선 | Δ/SD | 소병변 검출 | 전체 Dice | ICC | 판정 |")
        print("|---|---|---|---|---|---|---|---|---|")
        bs = "/".join(f"{r['small_dice']:.4f}" for r in base_3)
        print(f"| **기준선** | {bs} | **{b_small:.4f}** | — | — | "
              f"{np.mean([r['small_detect'] for r in base_3]):.3f} | {b_dice:.4f} | {b_icc:.4f} | — |")
        for name, v in out["variants"].items():
            sw = "/".join(f"{x:.4f}" for x in v["small_dice_by_seed"].values())
            print(f"| `{name}` | {sw} | {v['small_dice_mean']:.4f} | {v['effect_vs_base']:+.4f} | "
                  f"{v['effect_in_sd']:+.2f} | {v['small_detect_mean']:.3f} | "
                  f"{v['val_dice_mean']:.4f} ({v['val_dice_delta']:+.4f}) | "
                  f"{v['val_icc_mean']:.4f} ({v['val_icc_delta']:+.4f}) | {v['verdict']} |")


if __name__ == "__main__":
    main()
