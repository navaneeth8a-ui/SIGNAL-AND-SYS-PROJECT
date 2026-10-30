"""Dataset loading, position-aware train/validation split, and global scaling."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def load_filter_dataset(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. From the repo root, generate W.npy with "
            "the Part 1 workflow (see the existing README), then run Part 2 with --data ../W.npy. "
            "For a code-only smoke test, use --demo-data."
        )
    W = np.load(path)
    if W.ndim != 2:
        raise ValueError(f"Expected a 2-D W array, got shape {W.shape}")
    if W.shape[1] == 512:
        pass
    elif W.shape[0] == 512:
        W = W.T
    else:
        raise ValueError(f"Expected shape (N,512) or (512,N), got {W.shape}")
    W = np.asarray(W, dtype=np.float32)
    if not np.isfinite(W).all():
        raise ValueError("W contains NaN or infinite values")
    if W.shape[0] < 16:
        raise ValueError("Need at least 16 filters to split into train/validation sets")
    return W


def position_split(W: np.ndarray, val_stride: int = 8):
    """Hold out every val_stride-th position to avoid random-neighbour leakage."""
    if val_stride < 2:
        raise ValueError("val_stride must be at least 2")
    idx = np.arange(W.shape[0])
    val_mask = (idx % val_stride) == 0
    train_idx, val_idx = idx[~val_mask], idx[val_mask]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Split produced an empty train or validation set")
    return train_idx, val_idx


def global_scale(W_train: np.ndarray) -> float:
    """One scalar for all filters; per-filter normalization is intentionally avoided."""
    scale = float(np.max(np.abs(W_train)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Global scaling constant must be finite and positive")
    return scale
