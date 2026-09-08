#!/usr/bin/env python3
"""Save binary predicted masks for a split (input to train_cls.py --mask pred)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strokeai.data.dataset import SubjectCache  # noqa: E402
from strokeai.models import UNet2D  # noqa: E402
from strokeai.train import predict_subject  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--which", nargs="+", default=["train", "val"])
    ap.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args()
    cfg = json.load(open(a.run / "config.json"))
    torch.set_num_threads(cfg.get("threads", 4))
    model = UNet2D(in_ch=2, base=cfg["base_channels"], depth=cfg["depth"])
    model.load_state_dict(torch.load(a.run / "best.pt", map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    out = a.run / "pred_masks"
    out.mkdir(exist_ok=True)
    sp = json.load(open(a.splits))
    ids = sum([sp[w] for w in a.which], [])
    cache = SubjectCache(a.cache, ids)
    thr = a.threshold or cfg["threshold"]
    for sid in ids:
        prob = predict_subject(model, cache.img[sid], device=device)
        np.savez_compressed(out / f"{sid}.npz", mask=(prob >= thr).astype(np.uint8), prob_max=prob.max())
    print("wrote", len(ids), "masks to", out)
