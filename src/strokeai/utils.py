from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    """Fix every RNG we use. Call once at process start *and* pass `seed` to np.random.default_rng for
    anything that must be reproducible independently of call order."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # torch is optional for the data-only scripts
        pass


def sha256_of(obj: Any) -> str:
    """Stable hash of a JSON-serialisable object (sorted keys)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


class JsonlLogger:
    """Append-only JSON Lines logger: one dict per line, wall-clock stamped."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()

    def log(self, **record: Any) -> None:
        record.setdefault("t", round(time.time() - self._t0, 3))
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=float) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_yaml(path: Path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
