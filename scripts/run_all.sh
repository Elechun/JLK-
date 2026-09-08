#!/usr/bin/env bash
# End-to-end pipeline for the server. Each step is idempotent; re-run safely.
# Usage: bash scripts/run_all.sh [--gpu]      (GPU is auto-detected by torch; flag is informational)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

echo "== 1. download SOOP DWI subset (~2.8 GB, skips cached) =="
python scripts/download_soop.py --out data/raw/ds004889 --workers 16

echo "== 2. subject index =="
python scripts/build_index.py

echo "== 3. patient-level split (seed 2026) =="
python scripts/make_splits.py

echo "== 4. preprocess train+val slice cache; size/clip come from configs/seg_unet2d.yaml (test is NOT read until the final step) =="
python scripts/preprocess.py --config configs/seg_unet2d.yaml --workers "$(nproc)" --which train val

echo "== 5. unit tests =="
python -m pytest -q

echo "== 6. train 2-D U-Net (val-based model selection) =="
python scripts/train_seg.py --config configs/seg_unet2d.yaml --run runs/seg_unet2d

echo "== 7. diagnostics on val (A6) =="
python scripts/diagnose.py --run runs/seg_unet2d

echo "== 8. predicted masks for train+val -> etiology classifier (dev CV only) =="
python scripts/predict_masks.py --run runs/seg_unet2d --which train val
python scripts/train_cls.py --mask gt --model logreg
python scripts/train_cls.py --mask pred --pred-dir runs/seg_unet2d/pred_masks --model logreg

cat <<'MSG'
== DONE (dev).

Test-set evaluation is a separate, ONE-TIME step. It is frozen by the charter (section
"C. 최종 test 평가에 사용할 설정") and signed off in docs/agents/A6_report.md. Run it exactly as
written -- no threshold change, no checkpoint change, no re-training, and report the numbers
whether or not they meet the pre-registered expectations.

  export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0
  # 0. regression tests + proof that the cache still holds no test subject
  python -m pytest -q
  # 1. FIRST time any test image is read
  python scripts/preprocess.py --config configs/seg_unet2d.yaml --workers "$(nproc)" --which test
  # 2. test-cohort integrity / distribution (A3 hand-off 5) -- separate out dir, do not overwrite dev
  python scripts/analysis/a3_checks.py --which test --workers 32 --out results/a3_test
  python scripts/analysis/a3_stats.py  --which test --workers 32 --out results/a3_test
  # 3. segmentation, once
  python scripts/eval_seg.py --run runs/seg_unet2d --split test
  # 4. diagnostics on the test predictions
  python scripts/diagnose.py --run runs/seg_unet2d --eval runs/seg_unet2d/eval_test.json \
         --flags results/a3_test/acute_chronic_flags.csv --out results/diag_seg_unet2d_test
  python scripts/analysis/a6_failures.py --run runs/seg_unet2d --split test
  # 5. etiology classifier, once (deployment-like first, oracle second)
  python scripts/predict_masks.py --run runs/seg_unet2d --which test
  python scripts/train_cls.py --mask pred --pred-dir runs/seg_unet2d/pred_masks --final
  python scripts/train_cls.py --mask gt --final
MSG
