"""Reproducibility and atomic cache I/O."""
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np

SEED = 2026

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    import cv2
    import torch
    cv2.setRNGSeed(seed)
    cv2.setNumThreads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    os.replace(tmp, path)

def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)

def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def implementation_id(*filenames):
    """Invalidate features when student implementation files change."""
    package = Path(__file__).resolve().parent
    return fingerprint({name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                        for name in filenames})
