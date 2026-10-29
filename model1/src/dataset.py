"""PyTorch dataset: one sample = one trading day = all tickers' windows for that day."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class DayDataset(Dataset):
    def __init__(self, npz_path: str, split: int, window: int, min_history: int,
                 train: bool = False, aug: dict | None = None, seed: int = 0):
        z = np.load(npz_path, allow_pickle=False)
        self.X, self.valid = z["X"], z["valid"]
        self.y1, self.y5, self.yvol = z["y1"], z["y5"], z["yvol"]
        self.m1, self.m5, self.ymkt = z["ymask1"], z["ymask5"], z["ymkt"]
        self.dates, self.tickers = z["dates"], z["tickers"]
        self.L, self.min_hist = window, min_history
        self.train, self.aug = train, aug or {}
        self.rng = np.random.default_rng(seed)
        D = len(self.dates)
        self.days = [d for d in np.where(z["split"] == split)[0]
                     if d >= window - 1 and self._eligible(d).sum() >= 2]
        assert D == self.X.shape[0]

    def _eligible(self, d: int) -> np.ndarray:
        hist = self.valid[d - self.L + 1:d + 1].sum(axis=0)
        return self.valid[d] & self.m1[d] & (hist >= self.min_hist)

    def __len__(self) -> int:
        return len(self.days)

    def __getitem__(self, i: int) -> dict:
        d = self.days[i]
        tick = np.where(self._eligible(d))[0]
        if self.train and "ticker_subsample" in self.aug:
            lo, hi = self.aug["ticker_subsample"]
            k = int(self.rng.integers(min(lo, len(tick)), min(hi, len(tick)) + 1))
            tick = self.rng.choice(tick, size=max(2, k), replace=False)
        if self.train:
            # Random slot order every sample: a ticker never sits in a fixed position, so the model
            # cannot associate a slot (and therefore a symbol) with its behaviour.
            tick = self.rng.permutation(tick)
        x = self.X[d - self.L + 1:d + 1, tick].transpose(1, 0, 2).copy()      # [n, L, F]
        tmask = self.valid[d - self.L + 1:d + 1, tick].T.copy()               # [n, L] True = real day
        if self.train:
            p = self.aug.get("history_dropout", 0.0)
            for j in range(len(tick)):
                if self.rng.random() < p:
                    keep = int(self.rng.integers(self.min_hist, self.L + 1))
                    tmask[j, :self.L - keep] = False
            x += self.rng.normal(0, self.aug.get("feature_noise", 0.0), x.shape).astype(np.float32)
            fd = self.aug.get("feature_dropout", 0.0)
            if fd > 0:
                x *= (self.rng.random(x.shape[-1]) >= fd).astype(np.float32)
        x[~tmask] = 0.0
        return {
            "x": torch.from_numpy(x.astype(np.float32)),
            "tmask": torch.from_numpy(tmask),
            "y1": torch.from_numpy(self.y1[d, tick]),
            "y5": torch.from_numpy(self.y5[d, tick]),
            "yvol": torch.from_numpy(self.yvol[d, tick]),
            "m5": torch.from_numpy(self.m5[d, tick]),
            "ymkt": torch.tensor(self.ymkt[d], dtype=torch.float32),
            "day": d,
        }


def collate_days(batch: list[dict]) -> dict:
    """Pad a list of days to the max ticker count; `nmask` marks real tickers."""
    B = len(batch)
    n = max(b["x"].shape[0] for b in batch)
    L, F = batch[0]["x"].shape[1:]
    out = {
        "x": torch.zeros(B, n, L, F), "tmask": torch.zeros(B, n, L, dtype=torch.bool),
        "nmask": torch.zeros(B, n, dtype=torch.bool),
        "y1": torch.zeros(B, n), "y5": torch.zeros(B, n), "yvol": torch.zeros(B, n),
        "m5": torch.zeros(B, n, dtype=torch.bool),
        "ymkt": torch.stack([b["ymkt"] for b in batch]),
        "day": torch.tensor([b["day"] for b in batch]),
    }
    for i, b in enumerate(batch):
        k = b["x"].shape[0]
        out["x"][i, :k], out["tmask"][i, :k], out["nmask"][i, :k] = b["x"], b["tmask"], True
        for key in ("y1", "y5", "yvol", "m5"):
            out[key][i, :k] = b[key]
    return out
