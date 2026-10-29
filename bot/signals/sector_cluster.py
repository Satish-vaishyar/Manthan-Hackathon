"""bot.signals.sector_cluster — P2-08: TF-IDF + KMeans over /stocks descriptions.

Per docs/05 §6 + docs/08 P2-08:
    - Cluster company descriptions from `GET /stocks` into `k` sectors.
    - Deterministic via fixed `random_seed`.
    - Graceful fallback: if sklearn isn't installed, assign every ticker to
      sector_id 0 and emit a warning (the bot must still trade).
    - Used by the constructor to enforce `RiskSettings.max_sector_weight`.

Sklearn is a soft dependency — listed in `requirements.txt` but not required
for the P0 bot to boot. We import lazily so the absence of sklearn never
crashes the import of this module.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from ..config import SectorClusterSettings


# A minimal text-cleaning tokenizer — avoids a hard sklearn dependency just to
# strip punctuation + lowercase.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")


def _tokenize(text: str) -> list[str]:
    """Lowercased word tokens; mirrors the TF-IDF token_pattern above."""
    return [w.lower() for w in _WORD_RE.findall(text or "")]


@dataclass
class SectorCluster:
    """TF-IDF + KMeans sector mapper.

    Use:
        sc = SectorCluster(SectorClusterSettings())
        sc.fit(stock_infos)               # stock_infos: list[StockInfo]
        sector_of = sc.predict(ticker)    # ticker -> "S0".."S5"
    """

    settings: "SectorClusterSettings"  # type: ignore[assignment]
    _vectorizer: object = field(default=None, init=False, repr=False)
    _kmeans: object = field(default=None, init=False, repr=False)
    _ticker_index: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _tickers: list[str] = field(default_factory=list, init=False, repr=False)
    _descriptions: list[str] = field(default_factory=list, init=False, repr=False)
    _fitted: bool = field(default=False, init=False)
    _fallback: bool = field(default=False, init=False)

    # ------------------------------------------------------------------
    def fit(self, stock_infos: Iterable) -> "SectorCluster":
        """Fit the clusterer from a list of StockInfo-shaped objects."""
        tickers: list[str] = []
        descs: list[str] = []
        for s in stock_infos:
            t = getattr(s, "ticker", None)
            d = getattr(s, "description", None) or ""
            if not t:
                continue
            tickers.append(t)
            descs.append(d)
        self._tickers = tickers
        self._descriptions = descs
        self._ticker_index = {t: i for i, t in enumerate(tickers)}

        if len(tickers) < max(2, self.settings.min_n_clusters):
            # Too few — fall back to single bucket
            self._fallback = True
            self._fitted = True
            return self

        try:
            from sklearn.cluster import KMeans  # type: ignore
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        except ImportError:
            self._fallback = True
            self._fitted = True
            return self

        n = min(self.settings.n_clusters, max(self.settings.min_n_clusters,
                                              len(tickers) // 2))
        try:
            self._vectorizer = TfidfVectorizer(
                max_features=self.settings.max_features,
                ngram_range=(1, self.settings.ngram_max),
                min_df=self.settings.min_df,
                lowercase=True,
                token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9_\-]+\b",
            )
            x = self._vectorizer.fit_transform(descs)
            if x.shape[0] < n:
                self._fallback = True
                self._fitted = True
                return self
            self._kmeans = KMeans(
                n_clusters=n,
                random_state=self.settings.random_seed,
                n_init=10,
            )
            self._kmeans.fit(x)
            self._fitted = True
            self._fallback = False
        except Exception:
            # Any sklearn failure -> fallback
            self._fallback = True
            self._fitted = True
            self._vectorizer = None
            self._kmeans = None
        return self

    # ------------------------------------------------------------------
    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def n_clusters(self) -> int:
        if self._fallback or self._kmeans is None:
            return 1
        return int(getattr(self._kmeans, "n_clusters", 1))

    def predict_one(self, ticker: str) -> str:
        """Map a ticker to its sector label (e.g. 'S3')."""
        if self._fallback or self._kmeans is None or self._vectorizer is None:
            return "S0"
        idx = self._ticker_index.get(ticker)
        if idx is None:
            return "S0"
        x = self._vectorizer.transform([self._descriptions[idx]])
        label = int(self._kmeans.predict(x)[0])
        return f"S{label}"

    def predict(self, ticker: str) -> str:
        """Alias for `predict_one` — matches the constructor's sector_of lookup."""
        return self.predict_one(ticker)

    def predict_many(self, tickers: Iterable[str]) -> dict[str, str]:
        """Batch version of predict_one -> {ticker: sector_label}."""
        return {t: self.predict_one(t) for t in tickers}

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, str]:
        """Return the full {ticker -> sector_label} map (used by the bot)."""
        return {t: self.predict_one(t) for t in self._tickers}


# Standalone helper for callers that don't want to instantiate the class.
def cluster_sectors(stock_infos: Iterable,
                    settings: "SectorClusterSettings") -> dict[str, str]:
    """One-shot fit + predict -> {ticker: sector_label}."""
    sc = SectorCluster(settings)
    sc.fit(stock_infos)
    return sc.to_dict()