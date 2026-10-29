"""Training sequences and evaluation windows for model3."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def load(npz_path: str) -> dict:
    z = np.load(npz_path, allow_pickle=False)
    return {k: z[k] for k in z.files}


class SequenceDataset(Dataset):
    """Fully valid sequences of `seq_len` days inside one split, one ticker each (no ticker identity kept)."""

    def __init__(self, z: dict, split: int, seq_len: int, stride: int, offset: int = 0):
        ok = (z["split"][:, None] == split) & z["valid"] & z["m_next"]
        D, N = ok.shape
        self.z, self.L = z, seq_len
        self.items = [(s, n) for n in range(N) for s in range(offset % stride, D - seq_len + 1, stride)
                      if ok[s:s + seq_len, n].all()]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        s, n = self.items[i]
        sl = slice(s, s + self.L)
        z = self.z
        return {"x": torch.from_numpy(z["X"][sl, n]), "r": torch.from_numpy(z["r_next"][sl, n]),
                "sigma": torch.from_numpy(z["sigma"][sl, n]), "y_trend": torch.from_numpy(z["y_trend"][sl, n]),
                "m_trend": torch.from_numpy(z["m_trend"][sl, n])}


def eval_windows(z: dict, split: int, seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windows ending at every (day, ticker) of a split. Returns (X [M, L, F], day_idx [M], ticker_idx [M]).
    Steps before a ticker's first usable day are zero-filled (the LSTM sees only the past)."""
    days = np.where(z["split"] == split)[0]
    d_idx, t_idx = np.where(z["valid"][days] & z["m_next"][days])
    d_idx = days[d_idx]
    X, V = z["X"], z["valid"]
    F = X.shape[-1]
    out = np.zeros((len(d_idx), seq_len, F), dtype=np.float32)
    for k, (d, n) in enumerate(zip(d_idx, t_idx)):
        lo = max(0, d - seq_len + 1)
        w = X[lo:d + 1, n] * V[lo:d + 1, n, None]
        out[k, seq_len - len(w):] = w
    return out, d_idx, t_idx
