"""One sample = one ticker-day: up to K news items (tokenised) + day-D price context + labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from prepare import CTX_FEATURES


class TickerDayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_items: int, max_len: int, y_std: float,
                 train: bool = False, seed: int = 0):
        self.groups = [g for _, g in df.groupby(["date", "ticker"], sort=True)]
        self.tok, self.K, self.T = tokenizer, max_items, max_len
        self.y_std, self.train = y_std, train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i: int) -> dict:
        g = self.groups[i].sort_values("relevant", ascending=False, kind="stable")
        if self.train and len(g) > self.K:      # keep relevant items, sample the rest
            rel, rest = g[g["relevant"] == 1], g[g["relevant"] == 0]
            take = max(self.K - len(rel), 0)
            rest = rest.iloc[self.rng.permutation(len(rest))[:take]]
            g = pd.concat([rel, rest])
        g = g.iloc[:self.K]
        enc = self.tok(g["text"].tolist(), truncation=True, max_length=self.T, padding="max_length",
                       return_tensors="pt")
        k = len(g)
        ids = torch.zeros(self.K, self.T, dtype=torch.long)
        att = torch.zeros(self.K, self.T, dtype=torch.long)
        ids[:k], att[:k] = enc["input_ids"], enc["attention_mask"]
        item_mask = torch.zeros(self.K, dtype=torch.bool)
        item_mask[:k] = True
        def pad(col, dtype=torch.float32):
            v = torch.zeros(self.K, dtype=dtype)
            v[:k] = torch.tensor(g[col].to_numpy(), dtype=dtype)
            return v
        teacher = torch.zeros(self.K, 3)
        teacher[:k] = torch.tensor(g[["t_pos", "t_neg", "t_neu"]].to_numpy(), dtype=torch.float32)
        first = g.iloc[0]
        return {
            "ids": ids, "att": att, "item_mask": item_mask,
            "relevant": pad("relevant"), "teacher": teacher,
            "has_teacher": torch.tensor(bool(first["has_teacher"])),
            "ctx": torch.tensor(first[CTX_FEATURES].to_numpy(dtype=np.float32)),
            "y_ret": torch.tensor(float(first["y_ret"]) / self.y_std, dtype=torch.float32),
            "y_raw": torch.tensor(float(first["y_ret"]), dtype=torch.float32),
            "y_abs": torch.tensor(float(np.log(first["y_abs"] + 1e-4)), dtype=torch.float32),
            "idx": torch.tensor(i),
        }

    def keys(self, i: int) -> tuple[str, str]:
        g = self.groups[i]
        return g["date"].iloc[0], g["ticker"].iloc[0]
