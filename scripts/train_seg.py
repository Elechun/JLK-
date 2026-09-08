#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strokeai.train import train  # noqa: E402
from strokeai.utils import load_yaml  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/seg_unet2d.yaml"))
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--max-steps", type=int, default=None, help="debug: stop after N optimizer steps")
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    ap.add_argument("--device", default=None, help="cpu | cuda (default: cuda if available)")
    a = ap.parse_args()
    import torch

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    cfg = load_yaml(a.config)
    if a.epochs:
        cfg["epochs"] = a.epochs
    run = a.run or Path("runs") / a.config.stem
    splits = json.load(open(a.splits))
    print(train(cfg, run, a.cache, splits, device=device, max_steps=a.max_steps))
