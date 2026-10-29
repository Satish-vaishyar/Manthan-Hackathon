"""Live inference for the bot: news of one simulated day -> 8 features per ticker.

    scorer = NewsScorer("model2/artifacts")                      # loads every seed in model2_meta.json
    feats = scorer.score_day(news_items, candles, tickers, aliases)
    # feats[ticker] = [news_score, p_up, abs_move, rel_max, log1p_n_relevant, sent_pos, sent_neg, has_news]

These 8 values are exactly model1's optional news input (news_dim = 8) and the strategy's news signal.
Tickers with no news today get zeros with has_news = 0. The bot adds its own decayed memory on top.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from model import NRIM
from text import clean, compile_aliases, mask

FEATURES = ["news_score", "p_up", "abs_move", "rel_max", "log1p_n_relevant", "sent_pos", "sent_neg", "has_news"]


class NewsScorer:
    def __init__(self, artifacts: str | Path, device: str = "cpu", quantize: bool = True):
        art = Path(artifacts)
        self.meta = json.load(open(art / "model2_meta.json"))
        self.dev = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(self.meta["encoder"])
        self.members = []
        for mem in self.meta["members"]:
            st = torch.load(art / mem["pt"], map_location="cpu", weights_only=True)
            m = NRIM(**st["model_cfg"])
            m.load_state_dict(st["state_dict"], strict=False)
            m = m.to(self.dev).eval()
            if quantize and self.dev.type == "cpu":   # int8 Linear layers: ~2-3x faster on CPU
                m = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
            self.members.append(m)
        self.K, self.T, self.desc = self.meta["max_items"], self.meta["max_len"], self.meta["desc_chars"]

    @staticmethod
    def candle_ctx(c: dict | None) -> list[float]:
        """c = {open, high, low, close, prev_close (or None)} for day D."""
        if not c:
            return [0.0] * 5
        o, h, l, cl, pc = c["open"], c["high"], c["low"], c["close"], c.get("prev_close")
        rng = max(h - l, 1e-9)
        return [float(np.log(cl / pc)) if pc else 0.0, float(np.log(o / pc)) if pc else 0.0,
                float(np.log(cl / o)), float(np.log(h / l)), (2 * cl - h - l) / rng]

    @torch.no_grad()
    def score_day(self, news_items: list[dict], candles: dict[str, dict], tickers: list[str],
                  aliases: dict[str, list[str]]) -> dict[str, list[float]]:
        pats = compile_aliases(aliases)
        out = {t: [0.0] * len(FEATURES) for t in tickers}
        by_t: dict[str, list[str]] = {}
        for it in news_items:
            t = it.get("ticker")
            if t not in out:
                continue
            raw = clean(it.get("headline", "")) + ". " + clean(it.get("description", ""))[:self.desc]
            by_t.setdefault(t, []).append(mask(raw, t, tickers, pats))
        if not by_t:
            return out
        names = list(by_t)
        B = len(names)
        texts = [by_t[t][:self.K] for t in names]
        # Pad only to the longest item of the day (headlines are short), not to max_len.
        enc = self.tok([x for txt in texts for x in txt], truncation=True, max_length=self.T, padding=True,
                       return_tensors="pt")
        T = enc["input_ids"].shape[1]
        ids = torch.zeros(B, self.K, T, dtype=torch.long)
        att = torch.zeros_like(ids)
        im = torch.zeros(B, self.K, dtype=torch.bool)
        j = 0
        for i, txt in enumerate(texts):
            n = len(txt)
            ids[i, :n], att[i, :n], im[i, :n] = enc["input_ids"][j:j + n], enc["attention_mask"][j:j + n], True
            j += n
        ctx = torch.tensor([self.candle_ctx(candles.get(t)) for t in names], dtype=torch.float32)
        outs = [m(ids.to(self.dev), att.to(self.dev), im.to(self.dev), ctx.to(self.dev)) for m in self.members]
        def avg(key, fn=lambda x: x):
            return torch.stack([fn(o[key].float()) for o in outs]).mean(0).cpu()
        ret, p_up, ab = avg("ret"), avg("dir", torch.sigmoid), avg("abs")
        rel = avg("rel", torch.sigmoid) * im
        sent = torch.stack([torch.softmax(o["sent"].float(), -1) for o in outs]).mean(0).cpu()
        w = rel / rel.sum(-1, keepdim=True).clamp(min=1e-6)
        for i, t in enumerate(names):
            out[t] = [float(torch.tanh(ret[i])), float(p_up[i]), float(np.exp(ab[i])), float(rel[i].max()),
                      float(np.log1p(rel[i].sum())), float((w[i] * sent[i, :, 0]).sum()),
                      float((w[i] * sent[i, :, 1]).sum()), 1.0]
        return out
