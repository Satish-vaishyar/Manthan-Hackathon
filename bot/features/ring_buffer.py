"""bot.features.ring_buffer — per-ticker rolling OHLCV history (P1-01).

Maintains the last `window` daily candles per ticker. The bot's DayAssembler
calls `push(ticker, tick)` once per simulated day; downstream model loaders
call `frame(ticker)` to get a pandas DataFrame shaped exactly like the
training-time input to `model1/src/features.py::ticker_features` and
`model3/src/features.py::ticker_features`.

Design:
  - Per-ticker deque of PriceTickData, maxlen=window. Memory cost: ~280 bytes
    per ticker-day × 280 days × 40 tickers ≈ 3.5 MB total.
  - `frame(ticker, n)` returns the last n days as a DataFrame sorted oldest
    -> newest, with the column names model1/src/features.py and
    model3/src/features.py expect (date, ticker, open, high, low, close,
    volume).
  - Cross-sectional aggregates (`mkt_r1`, `excess_r1`, `cs_rank_r20`) are
    computed once the bot has pushed a full day's bundle.

Why not pandas throughout? Per docs/05 §3: features must be incremental.
Each push appends one row to one deque; cross-sectional pass over a 40-ticker
universe runs once per simulated day.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..contracts import PriceTickData


@dataclass
class Candle:
    """Lightweight row for the ring buffer (no need for full PriceTickData)."""
    date: object         # datetime.date
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @classmethod
    def from_tick(cls, tick: PriceTickData, sim_date) -> "Candle":
        """Convert a Kafka PriceTickData into a Candle (missing volume -> 0)."""
        return cls(
            date=sim_date,
            open=tick.open,
            high=tick.high,
            low=tick.low,
            close=tick.close,
            volume=getattr(tick, "volume", 0.0) or 0.0,
        )


class RingBuffer:
    """Per-ticker rolling history of daily candles.

    Args:
        window: max candles kept per ticker (default 280 — covers model1's
            longest lookback of 252 days with a small slack).
    """

    def __init__(self, window: int = 280) -> None:
        self.window = window
        self._buf: dict[str, deque[Candle]] = {}

    # ------------------------------------------------------------------
    # Push / extend
    # ------------------------------------------------------------------
    def push(self, ticker: str, tick: PriceTickData, sim_date) -> None:
        """Add one candle for `ticker`. Idempotent on the same date (replaces)."""
        buf = self._buf.get(ticker)
        if buf is None:
            buf = deque(maxlen=self.window)
            self._buf[ticker] = buf
        candle = Candle.from_tick(tick, sim_date)
        # Replace if the last entry is for the same date (DayAssembler
        # may push twice on overlap days).
        if buf and buf[-1].date == sim_date:
            buf[-1] = candle
        else:
            buf.append(candle)

    def push_candle(self, ticker: str, candle: Candle) -> None:
        """Direct push of a Candle (for tests + backfill)."""
        buf = self._buf.get(ticker)
        if buf is None:
            buf = deque(maxlen=self.window)
            self._buf[ticker] = buf
        if buf and buf[-1].date == candle.date:
            buf[-1] = candle
        else:
            buf.append(candle)

    # ------------------------------------------------------------------
    # Read-out
    # ------------------------------------------------------------------
    def frame(self, ticker: str, n: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Return the last `n` candles as a sorted DataFrame.

        Returns None if the ticker has fewer than 2 candles (can't compute
        returns).
        """
        buf = self._buf.get(ticker)
        if not buf or len(buf) < 2:
            return None
        rows = list(buf) if n is None else list(buf)[-n:]
        df = pd.DataFrame([{
            "date": c.date, "ticker": ticker,
            "open": c.open, "high": c.high, "low": c.low,
            "close": c.close, "volume": c.volume,
        } for c in rows])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def frame_all(self, tickers: list[str], n: Optional[int] = None) -> pd.DataFrame:
        """Concatenate per-ticker frames into one long frame."""
        parts = []
        for t in tickers:
            f = self.frame(t, n=n)
            if f is not None:
                parts.append(f)
        if not parts:
            return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])
        return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)

    def length(self, ticker: str) -> int:
        return len(self._buf.get(ticker, ()))

    def tickers(self) -> list[str]:
        return list(self._buf.keys())

    def __len__(self) -> int:
        return len(self._buf)