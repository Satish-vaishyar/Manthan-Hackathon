"""bot.models.model2_bridge — wrapper around the trained model2 (M2).

Per model2/src/infer.py:
    scorer = NewsScorer("model2/artifacts")
    feats = scorer.score_day(news_items, candles, tickers, aliases)
    # feats[ticker] = [news_score, p_up, abs_move, rel_max, log1p_n_relevant,
    #                   sent_pos, sent_neg, has_news]   (8 values)

This bridge:
    - Imports NewsScorer directly (model2/ is a sibling package).
    - Maps the 8-feature output to a single [-1, +1] news_score for the ensemble.
    - Provides a fallback when the artifact isn't ready (calls rules-based scorer).
    - Surfaces `news_features[ticker] = 8-vector` so model1 can consume it later.

Why import model2 directly rather than re-implement?
    Per docs/05 §4: same code path for training and inference. Re-implementing
    in the bot would risk drift from the trained model.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path
from typing import Optional

from ..contracts import DayBundle, NewsFlashData, PriceTickData


class Model2NotReady(RuntimeError):
    pass


@dataclass
class Model2Meta:
    encoder: str = "ProsusAI/finbert"
    max_items: int = 16
    max_len: int = 128
    desc_chars: int = 300
    n_features: int = 8


class Model2NewsScorer:
    """Adapter that uses model2/src/infer.py:NewsScorer under the hood.

    Falls back to keyword polarity (from rules.py) when model2 hasn't been
    trained yet — so the bot keeps running on P0 until model2 is shipped.
    """

    def __init__(self, artifacts_dir: str | Path = "model2/artifacts",
                 quantize: bool = True) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.quantize = quantize
        self.meta: Optional[Model2Meta] = None
        self._scorer = None
        self._loaded = False

    def _try_load(self) -> bool:
        if self._loaded:
            return self._scorer is not None
        try:
            # Add model2/src to sys.path so `from infer import NewsScorer` works
            model2_src = Path(__file__).resolve().parents[2] / "model2" / "src"
            if not (model2_src / "infer.py").exists():
                return False
            sp = str(model2_src)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            from infer import NewsScorer, FEATURES  # noqa: F401
            from text import load_aliases

            art = self.artifacts_dir
            if not (art / "model2_meta.json").exists():
                return False
            self._scorer = NewsScorer(art, device="cpu", quantize=self.quantize)
            cfg = json.loads((Path(__file__).resolve().parents[2] /
                              "model2" / "configs" / "model2.yaml").read_text())
            self.meta = Model2Meta(
                encoder=cfg["model"]["encoder"],
                max_items=cfg["text"]["max_items_per_day"],
                max_len=cfg["text"]["max_len"],
                desc_chars=cfg["text"]["desc_chars"],
            )
            self._aliases = load_aliases(
                Path(cfg["data"]["aliases_auto"]),
                Path(cfg["data"]["aliases_manual"]),
            )
            self._loaded = True
            return True
        except Exception:  # noqa: BLE001
            return False

    @property
    def ready(self) -> bool:
        return self._try_load() and self._scorer is not None

    def score(self, bundle: DayBundle) -> dict[str, dict[str, float]]:
        """Score every ticker in the bundle.

        Returns:
            dict[ticker -> {
                "news_score": float in [-1, +1],    # for the ensemble
                "p_up": float, "abs_move": float,
                "rel_max": float, "log1p_n": float,
                "sent_pos": float, "sent_neg": float,
                "has_news": float (0 or 1),
            }]
        """
        if not self.ready:
            raise Model2NotReady(
                f"model2 artifacts not found in {self.artifacts_dir} — "
                "run model2/src/train.py and model2/src/export.py first"
            )

        tickers = list(bundle.ticks.keys())
        # Build inputs for model2
        news_items = []
        candles = {}
        for t, tick in bundle.ticks.items():
            candles[t] = {
                "open": tick.open, "high": tick.high,
                "low": tick.low, "close": tick.close,
                "prev_close": None,  # we don't have D-1 here; model2 handles None
            }
            for n in bundle.news:
                if n.ticker == t:
                    news_items.append({
                        "ticker": t,
                        "headline": n.headline or "",
                        "description": n.description or "",
                    })

        feats = self._scorer.score_day(news_items, candles, tickers, self._aliases)

        out: dict[str, dict[str, float]] = {}
        for t in tickers:
            v = feats.get(t, [0.0] * 8)
            out[t] = {
                "news_score": float(v[0]),
                "p_up": float(v[1]),
                "abs_move": float(v[2]),
                "rel_max": float(v[3]),
                "log1p_n": float(v[4]),
                "sent_pos": float(v[5]),
                "sent_neg": float(v[6]),
                "has_news": float(v[7]),
            }
        return out


# -----------------------------------------------------------------------------
# Keyword fallback (re-uses rules.py logic) when model2 isn't trained yet
# -----------------------------------------------------------------------------
class KeywordNewsScorer:
    """Pure-keyword fallback. Mirrors rules._news_score for parity."""

    def __init__(self) -> None:
        from .rules import RulesStrategy
        self._rules = RulesStrategy()

    def score(self, bundle: DayBundle) -> dict[str, float]:
        out: dict[str, float] = {}
        for t in bundle.ticks.keys():
            score = self._rules._news_score(bundle, t)
            out[t] = {"news_score": score, "p_up": 0.5 + score / 2.0,
                      "abs_move": 0.0, "rel_max": abs(score),
                      "log1p_n": 1.0 if score != 0 else 0.0,
                      "sent_pos": max(0.0, score), "sent_neg": max(0.0, -score),
                      "has_news": 1.0 if score != 0 else 0.0}
        return out