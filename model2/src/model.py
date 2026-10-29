"""model2 — News Relevance & Impact Model (NRIM).

  each news item:  company-masked text --FinBERT (top layers fine-tuned)--> [CLS] --proj--> e_k (128)
                   e_k -> relevance logit (weak labels: does it mention the company?)
                   e_k -> 3-way sentiment (distilled from frozen FinBERT)
  per ticker-day:  relevance-gated attention pooling over items -> v
                   [v ‖ ctx-MLP(day-D candle) ‖ log(1 + expected #relevant items)] -> MLP
                   -> ret (next-day excess return, standardised), direction logit, log |move|
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class NRIM(nn.Module):
    def __init__(self, encoder: str, n_ctx: int = 5, d_item: int = 128, unfreeze_layers: int = 4,
                 dropout: float = 0.2, **_):
        super().__init__()
        self.enc = AutoModel.from_pretrained(encoder)
        for p in self.enc.parameters():
            p.requires_grad = False
        layers = self.enc.encoder.layer
        for layer in layers[len(layers) - unfreeze_layers:] if unfreeze_layers > 0 else []:
            for p in layer.parameters():
                p.requires_grad = True
        h = self.enc.config.hidden_size
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, d_item), nn.GELU())
        self.rel_head = nn.Linear(d_item, 1)
        self.sent_head = nn.Linear(d_item, 3)
        self.att = nn.Linear(d_item, 1)
        self.ctx = nn.Sequential(nn.Linear(n_ctx, 32), nn.GELU())
        self.day = nn.Sequential(nn.Linear(d_item + 32 + 1, d_item), nn.GELU(), nn.Dropout(dropout))
        self.ret_head = nn.Linear(d_item, 1)
        self.dir_head = nn.Linear(d_item, 1)
        self.abs_head = nn.Linear(d_item, 1)

    def encoder_params(self):
        return [p for p in self.enc.parameters() if p.requires_grad]

    def head_params(self):
        enc = {id(p) for p in self.enc.parameters()}
        return [p for p in self.parameters() if id(p) not in enc]

    def forward(self, ids, att, item_mask, ctx) -> dict:
        """ids/att [B,K,T]; item_mask [B,K]; ctx [B,C]."""
        B, K, T = ids.shape
        flat = item_mask.reshape(-1)
        cls = torch.zeros(B * K, self.enc.config.hidden_size, device=ids.device)
        if flat.any():   # encode only real items
            out = self.enc(input_ids=ids.reshape(-1, T)[flat], attention_mask=att.reshape(-1, T)[flat])
            cls[flat] = out.last_hidden_state[:, 0].to(cls.dtype)
        e = self.proj(cls).reshape(B, K, -1)
        rel = self.rel_head(e).squeeze(-1)
        sent = self.sent_head(e)
        score = self.att(e).squeeze(-1) + F.logsigmoid(rel)          # relevance-gated attention
        score = score.masked_fill(~item_mask, float("-inf"))
        w = torch.softmax(score, -1).nan_to_num(0.0)
        v = (w.unsqueeze(-1) * e).sum(1)
        n_rel = (torch.sigmoid(rel) * item_mask).sum(-1, keepdim=True)
        z = self.day(torch.cat([v, self.ctx(ctx), torch.log1p(n_rel)], -1))
        return {"ret": self.ret_head(z).squeeze(-1), "dir": self.dir_head(z).squeeze(-1),
                "abs": self.abs_head(z).squeeze(-1), "rel": rel, "sent": sent, "w": w}
