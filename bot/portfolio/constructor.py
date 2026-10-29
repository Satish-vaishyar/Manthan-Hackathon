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
    # Final cap sweep — anything still above max_weight gets clipped. The
    # gross_long_target applied by build_target_weights then scales the
    # total to regime target (0.92 / 0.75 / 0.40) so the per-name cap
    # invariant is what protects us from single-name concentration.
    for t, w in list(weights.items()):
        if w > max_weight:
            weights[t] = max_weight
    return weights


def build_target_weights(
    scores: list[TickerScore],
    settings: StrategySettings,
    equity: float,
    *,
    regime: str = "neutral",
    sector_caps: Optional[dict[str, float]] = None,
    sector_of: Optional[dict[str, str]] = None,
    gross_long_target: Optional[dict[str, float]] = None,
    current_holdings: Optional[dict[str, int]] = None,
    current_shorts: Optional[dict[str, int]] = None,
    last_close: Optional[dict[str, float]] = None,
    target_cash_utilization: float = 0.0,
) -> dict[str, float]:
    """Top-K inverse-vol weights, capped, regime-scaled.

    Args:
        regime: 'risk_on' / 'neutral' / 'risk_off'. Used to look up gross long
            target if `gross_long_target` is provided.
        gross_long_target: dict[regime -> multiplier in [0, 1]]. If supplied,
            the final weights are scaled by this multiplier so total gross
            long exposure matches the regime target. If None, no scaling is
            applied (caller is responsible for it).
    """
    weights = inverse_vol_weights(
        scores,
        top_k=settings.top_k,
        max_weight=settings.max_weight_name,
        equity=equity,
    )
    # P3-13: drop excluded tickers before sector caps / regime scaling.
    for t in getattr(settings, "exclude_tickers", []) or []:
        weights.pop(t, None)
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
    # Apply gross long target (M5)
    if gross_long_target is not None:
        mult = gross_long_target.get(regime, 1.0)
        if mult < 1.0:
            weights = {t: w * mult for t, w in weights.items()}
    # P3-14: cash-utilization targeting. If the current long exposure is
    # below the target (e.g. 85% of equity), scale up the weights so a
    # rebalance actually invests the idle cash. Conversely if exposure is
    # above target, deflate so we don't over-buy. This keeps 80-90% of the
    # wallet invested most of the time instead of letting cash pile up.
    # IMPORTANT: only apply when caller explicitly opts in via
    # target_cash_utilization > 0 OR the strategy carries the flag. The
    # default fallback (0.85) is intentionally NOT applied when callers
    # don't pass the param, so existing tests / older call sites aren't
    # silently altered.
    target_util = float(target_cash_utilization or 0.0)
    if target_util <= 0:
        target_util = float(getattr(settings, "_target_cash_utilization", 0.0) or 0.0)
    # Only proceed if the caller explicitly opted in.
    if target_util <= 0:
        return weights
    target_util = max(0.0, min(1.0, float(target_util)))
    if current_holdings is not None and last_close is not None:
        long_exposure = sum((last_close.get(t, 0) * q) for t, q in current_holdings.items())
    else:
        long_exposure = 0.0
    if current_shorts is not None and last_close is not None:
        short_exposure = sum((last_close.get(t, 0) * q) for t, q in current_shorts.items())
    else:
        short_exposure = 0.0
    current_util = (long_exposure - short_exposure) / equity if equity > 0 else 0.0
    total_weight = sum(weights.values())
    if total_weight > 0 and current_util > 0:
        scale = target_util / max(current_util, 1e-9)
        scale = max(0.5, min(1.5, scale))
        weights = {t: w * scale for t, w in weights.items()}
    elif total_weight > 0 and current_util <= 0 and target_util > 0:
        weights = {t: w * (target_util / max(total_weight, 1e-9))
                   for t, w in weights.items()}
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

    # Exit positions not in target. P3-19: exits (including legacy holdings
    # from previous runs) use the LOWER exit_min_notional floor — selling a
    # Rs 20K stale position is worth the Rs 20 brokerage; entries stay at
    # min_trade_notional for fee efficiency.
    exit_floor = float(getattr(settings, "exit_min_notional", 0.0) or 0.0)
    if exit_floor <= 0:
        exit_floor = settings.min_trade_notional
    for ticker, cur_qty in current_holdings.items():
        if ticker in target_weights or cur_qty <= 0:
            continue
        price = last_close.get(ticker, 0)
        if price <= 0:
            continue
        notional = cur_qty * price
        if notional < exit_floor:
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

    # P3-16: drop marginal trades UNCONDITIONALLY — any trade whose notional
    # is below 50x the round-trip brokerage pays more in fees than it moves
    # the needle. (min_trade_notional already covers this at 75k, but keep
    # the explicit guard for configs that lower min_trade_notional.)
    min_worthwhile = max(
        float(getattr(settings, "min_trade_notional", 50_000.0) or 50_000.0) * 0.5,
        50.0 * 20.0,  # 50x brokerage_per_order (Rs 20)
    )
    kept_trades = []
    for t in out.trades:
        # P3-19: risk-reducing exits (exit_target / cover_exit) are exempt —
        # getting out of an unwanted position is worth the fee at any size
        # above exit_min_notional.
        if t.reason in ("exit_target", "cover_exit") or t.notional >= min_worthwhile:
            kept_trades.append(t)
        else:
            out.skipped.append((t.ticker, f"marginal_notional_{t.notional:.0f}"))
    out.trades = kept_trades

    # Cap by max orders remaining
    if len(out.trades) > max_orders_remaining:
        # Keep the largest-notional trades (concentrate capital, fewer
        # brokerage events).
        out.trades.sort(key=lambda t: t.notional, reverse=True)
        kept = out.trades[:max_orders_remaining]
        dropped = out.trades[max_orders_remaining:]
        out.trades = kept
        for t in dropped:
            out.skipped.append((t.ticker, f"order_budget_exceeded"))

    return out