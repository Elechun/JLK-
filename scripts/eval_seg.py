#!/usr/bin/env python3
"""Evaluate a trained run on a split. `--split test` is the FINAL evaluation: run it once, after A6 sign-off."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strokeai.data.dataset import SubjectCache  # noqa: E402
from strokeai.models import UNet2D  # noqa: E402
from strokeai.train import evaluate_subjects  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--min-voxels", type=int, default=0)
    a = ap.parse_args()
    cfg = json.load(open(a.run / "config.json"))
    torch.set_num_threads(cfg.get("threads", 4))
    model = UNet2D(in_ch=2, base=cfg["base_channels"], depth=cfg["depth"])
    model.load_state_dict(torch.load(a.run / a.ckpt, map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    ids = json.load(open(a.splits))[a.split]
    cache = SubjectCache(a.cache, ids)
    summ, per = evaluate_subjects(model, cache, threshold=a.threshold or cfg["threshold"], device=device, tta_flip=a.tta, min_voxels=a.min_voxels)
    out = a.run / f"eval_{a.split}{'_tta' if a.tta else ''}.json"
    json.dump({"summary": summ, "per_subject": per, "threshold": a.threshold or cfg["threshold"], "tta": a.tta, "min_voxels": a.min_voxels},
              open(out, "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print("wrote", out)
