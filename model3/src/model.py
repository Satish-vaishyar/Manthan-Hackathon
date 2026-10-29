"""model3 — Deep Momentum Network (LSTM).

  x [B, L, F] momentum features -> LSTM(hidden) -> Dropout -> Linear -> tanh  = position signal per step
                                                          +-> Linear          = 21-day trend logit (auxiliary)

Trained end-to-end on the net Sharpe ratio of the volatility-targeted positions it outputs (costs included),
so it learns *how much* trend to follow, not only the direction. One shared network for every ticker; no ticker
identity anywhere.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DMN(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, layers: int = 1, dropout: float = 0.3,
                 position: str = "tanh", **_):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.pos_head = nn.Linear(hidden, 1)
        self.trend_head = nn.Linear(hidden, 1)
        self.act = torch.tanh if position == "tanh" else torch.sigmoid

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.lstm(x)
        h = self.drop(h)
        return self.act(self.pos_head(h)).squeeze(-1), self.trend_head(h).squeeze(-1)
