"""model1 — Cross-Sectional Temporal Transformer (CSTT).

  per ticker:  [L, F] --Linear+PosEmb--> Temporal Transformer --attention pool--> h_i      (d)
  per day:     [MKT] + {h_i}  --Cross-sectional Transformer (set, no position)--> z_i, z_mkt
  (optional)   news_i --MLP--> gated add to z_i
  heads:       z_i -> ret(h1,h5), direction logit, vol ;  z_mkt -> market regime logit

The model has no ticker embedding on purpose: live tickers differ from training tickers.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AttnPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # h [M,L,d], mask [M,L]
        s = self.score(h).squeeze(-1).masked_fill(~mask, float("-inf"))
        w = torch.softmax(s, dim=-1).nan_to_num(0.0)
        return (w.unsqueeze(-1) * h).sum(1)


class CSTT(nn.Module):
    def __init__(self, n_features: int, window: int, d_model: int = 64, temporal_layers: int = 2,
                 cross_layers: int = 2, n_heads: int = 4, ff_mult: int = 2, dropout: float = 0.2,
                 use_news: bool = False, news_dim: int = 8, **_):
        super().__init__()
        d = d_model
        self.inp = nn.Sequential(nn.Linear(n_features, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(1, window, d))
        nn.init.normal_(self.pos, std=0.02)

        def enc(layers: int) -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(d, n_heads, d * ff_mult, dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

        self.temporal = enc(temporal_layers)
        self.pool = AttnPool(d)
        self.last_proj = nn.Linear(2 * d, d)      # concat(attention pool, last real day)
        self.mkt_token = nn.Parameter(torch.zeros(1, 1, d))
        self.cross = enc(cross_layers)
        self.norm = nn.LayerNorm(d)
        self.use_news = use_news
        if use_news:
            self.news = nn.Sequential(nn.Linear(news_dim, d), nn.GELU(), nn.Linear(d, d))
            self.news_gate = nn.Linear(2 * d, d)
        self.ret_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 2))
        self.dir_head = nn.Linear(d, 1)
        self.vol_head = nn.Linear(d, 1)
        self.mkt_head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor, tmask: torch.Tensor, nmask: torch.Tensor,
                news: torch.Tensor | None = None, news_mask: torch.Tensor | None = None) -> dict:
        """x [B,N,L,F]; tmask [B,N,L] True=real day; nmask [B,N] True=real ticker."""
        B, N, L, F = x.shape
        xf, tm = x.reshape(B * N, L, F), tmask.reshape(B * N, L)
        # padded tickers: keep one slot so attention is defined (out-of-place for ONNX export)
        tm = torch.cat([tm[:, :-1], (tm[:, -1] | ~tm.any(-1)).unsqueeze(-1)], dim=1)
        h = self.inp(xf) + self.pos[:, -L:]
        h = self.temporal(h, src_key_padding_mask=~tm)
        last = h[:, -1]                           # day D is always the last slot
        h = self.last_proj(torch.cat([self.pool(h, tm), last], -1)).reshape(B, N, -1)

        tokens = torch.cat([self.mkt_token.expand(B, -1, -1), h], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~nmask], 1)
        z = self.norm(self.cross(tokens, src_key_padding_mask=pad))
        z_mkt, z = z[:, 0], z[:, 1:]
        if self.use_news and news is not None and news_mask is not None:
            n = self.news(news)
            g = torch.sigmoid(self.news_gate(torch.cat([z, n], -1)))
            z = z + g * n * news_mask.unsqueeze(-1).float()
        ret = self.ret_head(z)
        return {
            "ret1": ret[..., 0], "ret5": ret[..., 1],
            "dir": self.dir_head(z).squeeze(-1),
            "vol": self.vol_head(z).squeeze(-1),
            "mkt": self.mkt_head(z_mkt).squeeze(-1),
        }
