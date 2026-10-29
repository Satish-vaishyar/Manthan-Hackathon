"""bot.portfolio.constructor — scores -> target weights -> share deltas.

Per docs/04 §4:
    - Inputs: per-ticker score (composite of ranker + news + momentum)
    - Inverse-vol sizing, max weight per name
    - Rebalance threshold (only trade if weight moves > threshold)
    - Min trade notional (₹50,000)
    - Max orders/day (caller passes budget remaining)
    - Generates trade deltas to apply to current holdings
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date
from typing import Optional

from ..config import StrategySettings
from .ledger import Ledger


@dataclass
class TickerScore:
    ticker: str
    score: float            # composite ensemble score (higher = more attractive)
    vol: float = 0.0        # recent vol (for inverse-vol sizing)
    sector: Optional[str] = None
    price: float = 0.0      # latest close for sizing math


@dataclass
class TradeDelta:
    ticker: str
    action: str             # BUY | SELL | SHORT | COVER | HOLD
    qty: int
    price: float            # reference price for notional checks
    target_weight: float
    current_weight: float
    reason: str

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass
class PortfolioTarget:
    """Output of one constructor pass: trades to execute this rebalance."""
    trades: list[TradeDelta] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (ticker, reason)


def inverse_vol_weights(
    scores: list[TickerScore],
    *,
    top_k: int,
    max_weight: float,
    equity: float,
) -> dict[str, float]:
    """Pick top_k tickers by score, weight = score / vol (clipped at max_weight)."""
    if not scores or equity <= 0:
        return {}
    eligible = [s for s in scores if s.score > 0 and s.vol > 0]
    if not eligible:
        return {}
    eligible.sort(key=lambda s: s.score, reverse=True)
    chosen = eligible[:top_k]
    raw: dict[str, float] = {}
    for s in chosen:
        raw[s.ticker] = max(0.0, s.score) / max(1e-9, s.vol)
    total = sum(raw.values())
    if total <= 0:
        return {}
    weights = {t: w / total for t, w in raw.items()}
    # Cap per name; redistribute excess to non-capped
    for _ in range(5):
        excess = 0.0
        capped = set()
        for t, w in weights.items():
            if w > max_weight:
                excess += w - max_weight
                weights[t] = max_weight
                capped.add(t)
        if excess <= 0:
            break
        non_capped = [t for t in weights if t not in capped]
        if not non_capped:
            break
        share = excess / len(non_capped)
        for t in non_capped:
            weights[t] += share
    # Renormalise after capping
    total = sum(weights.values())
    if total > 0:
        weights = {t: w / total for t, w in weights.items()}
    return weights


def build_target_weights(
    scores: list[TickerScore],
    settings: StrategySettings,
    equity: float,
    *,
    regime: str = "neutral",
    sector_caps: Optional[dict[str, float]] = None,
    sector_of: Optional[dict[str, str]] = None,
) -> dict[str, float]:
    """Top-K inverse-vol weights, capped, regime-scaled."""
    weights = inverse_vol_weights(
        scores,
        top_k=settings.top_k,
        max_weight=settings.max_weight_name,
        equity=equity,
    )
    # Scale by regime's gross long target
    gross = settings.weights  # not used here; use risk.gross_long_target
    # (regime scaling is applied in caller via settings.risk.gross_long_target)
    # Sector caps: zero out any ticker whose sector exceeds cap
    if sector_caps and sector_of:
        sector_exp: dict[str, float] = {}
        for t, w in weights.items():
            sec = sector_of.get(t)
            if sec:
                sector_exp[sec] = sector_exp.get(sec, 0.0) + w
        for t in list(weights.keys()):
            sec = sector_of.get(t)
            if not sec:
                continue
            cap = sector_caps.get(sec)
            if cap and sector_exp[sec] > cap:
                weights[t] = 0.0
        # Renormalise after sector zeros
        total = sum(weights.values())
        if total > 0:
            weights = {t: w / total for t, w in weights.items()}
    return weights


def build_trade_deltas(
    current_holdings: dict[str, int],
    current_shorts: dict[str, int],
    last_close: dict[str, float],
    target_weights: dict[str, float],
    equity: float,
    *,
    settings: StrategySettings,
    max_orders_remaining: int,
    short_allowed: bool = True,
) -> PortfolioTarget:
    """Convert target weights into trade deltas.

    - Skip if |target - current| < settings.rebalance_threshold
    - Skip if notional < settings.min_trade_notional
    - Cap total trades by max_orders_remaining
    - LONG-only by default; emits SHORT when short_allowed and target < 0
    """
    if equity <= 0:
        return PortfolioTarget()
    out = PortfolioTarget()

    # Long side
    for ticker, tgt_w in target_weights.items():
        price = last_close.get(ticker, 0)
        if price <= 0:
            out.skipped.append((ticker, "no_price"))
            continue
        cur_qty = current_holdings.get(ticker, 0)
        cur_w = (cur_qty * price) / equity
        if abs(tgt_w - cur_w) < settings.rebalance_threshold:
            out.skipped.append((ticker, "below_threshold"))
            continue
        tgt_qty = int((tgt_w * equity) // price)
        delta = tgt_qty - cur_qty
        if delta == 0:
            out.skipped.append((ticker, "rounded_to_zero"))
            continue
        notional = abs(delta) * price
        if notional < settings.min_trade_notional:
            out.skipped.append((ticker, f"min_notional_{notional:.0f}"))
            continue
        action = "BUY" if delta > 0 else "SELL"
        out.trades.append(TradeDelta(
            ticker=ticker, action=action, qty=abs(delta), price=price,
            target_weight=tgt_w, current_weight=cur_w,
            reason=f"weight {cur_w:.2%}->{tgt_w:.2%}",
        ))

    # Exit positions not in target
    for ticker, cur_qty in current_holdings.items():
        if ticker in target_weights or cur_qty <= 0:
            continue
        price = last_close.get(ticker, 0)
        if price <= 0:
            continue
        notional = cur_qty * price
        if notional < settings.min_trade_notional:
            continue
        out.trades.append(TradeDelta(
            ticker=ticker, action="SELL", qty=cur_qty, price=price,
            target_weight=0.0, current_weight=(cur_qty * price) / equity,
            reason="exit_target",
        ))

    # Shorts: only cover existing shorts whose target=0, only SHORT when short_allowed
    for ticker, short_qty in current_shorts.items():
        if ticker in target_weights or short_qty <= 0:
            continue
        price = last_close.get(ticker, 0)
        if price <= 0:
            continue
        out.trades.append(TradeDelta(
            ticker=ticker, action="COVER", qty=short_qty, price=price,
            target_weight=0.0, current_weight=-(short_qty * price) / equity,
            reason="cover_exit",
        ))

    # Cap by max orders remaining
    if len(out.trades) > max_orders_remaining:
        # Keep the largest-notional trades
        out.trades.sort(key=lambda t: t.notional, reverse=True)
        kept = out.trades[:max_orders_remaining]
        dropped = out.trades[max_orders_remaining:]
        out.trades = kept
        for t in dropped:
            out.skipped.append((t.ticker, f"order_budget_exceeded"))

    return out