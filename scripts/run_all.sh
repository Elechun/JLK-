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

echo "== 4. preprocess train+val to 128x128 slice cache (test is NOT read until the final step) =="
python scripts/preprocess.py --size 128 --workers "$(nproc)" --which train val

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

echo "== DONE (dev). Test-set evaluation is a separate, one-time step after A6 sign-off:"
echo "   python scripts/preprocess.py --size 128 --workers \$(nproc) --which test"
echo "   python scripts/eval_seg.py --run runs/seg_unet2d --split test"
echo "   python scripts/predict_masks.py --run runs/seg_unet2d --which test && python scripts/train_cls.py --mask pred --pred-dir runs/seg_unet2d/pred_masks --final"
