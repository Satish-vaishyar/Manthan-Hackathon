"""bot.portfolio.short_engine — Short Alpha Engine (P4-01).

A SEPARATE bearish signal pipeline — not a reversal of the long strategy:

    S_short(t) = w_R * (-R) + w_N * (-N) + w_M * (-M)

where R/N/M are the per-ticker rank / news / momentum sub-scores from the
SignalsEnsemble (each in [-1, +1]; missing axes get weight 0 and the rest
renormalise). A LARGER S_short means stronger evidence of a future decline.
The engine has its own weights (default equal thirds — the long weights
25.7/33.7/40.6 were optimised for longs and are NOT assumed valid here),
its own entry threshold (default 0.85 — stricter than the long 0.75), and
its own exit rule (bearish-score decay below `exit_threshold`).

Modes (config strategy.short_engine.mode):
    off          — no short candidates, ever (Configuration A control).
    hedge        — candidates only in risk_off regimes (Configuration B:
                   small bearish hedge when the market is falling).
    independent  — candidates every day (Configuration C: full long-short).

Design rules (per spec):
  * Never short a stock merely because the long side rejected it — a
    candidate must pass its OWN entry threshold.
  * The idle-cash deployer never touches shorts (long-only by design).
  * If nothing qualifies, the short allocation stays unused.
  * Sizing: inverse-vol among qualifiers, HARD per-name cap
    (min(strategy.max_weight_name, max_weight_short)), gross short exposure
    capped at risk.max_short_gross. Shorts share the daily order budget
    with longs (constructor consumes the same `max_orders_remaining`).
"""
from __future__ import annotations

from dataclasses import dataclass

import structlog

_log = structlog.get_logger("portfolio.short_engine")


@dataclass
class ShortSignal:
    ticker: str
    bearish_score: float       # S_short, higher = stronger decline evidence
    vol: float
    rank_score: float
    news_score: float
    momentum_score: float


class ShortAlphaEngine:
    """Evaluates negative-return opportunities independently of the longs."""

    def __init__(self, settings) -> None:
        # settings: config.ShortEngineSettings
        self.settings = settings

    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(getattr(self.settings, "enabled", False)) and \
            getattr(self.settings, "mode", "off") != "off"

    # ------------------------------------------------------------------
    def bearish_score(self, rank: float, news: float, mom: float,
                      has_rank: bool, has_news: bool, has_mom: bool) -> float:
        """Weighted sum of the NEGATED sub-scores, renormalised over the
        axes that actually produced a value (mirrors the long ensemble's
        graceful-degradation rule)."""
        w = getattr(self.settings, "weights", None) or {}
        wr = float(w.get("rank", 1.0 / 3.0)) if has_rank else 0.0
        wn = float(w.get("news", 1.0 / 3.0)) if has_news else 0.0
        wm = float(w.get("mom", 1.0 / 3.0)) if has_mom else 0.0
        tot = wr + wn + wm
        if tot <= 0:
            return 0.0
        return (wr * (-rank) + wn * (-news) + wm * (-mom)) / tot

    # ------------------------------------------------------------------
    def evaluate(self, evals, *, regime: str, vol_fn,
                 held_shorts: dict[str, int] | None = None,
                 has_rank: bool = True, has_news: bool = True,
                 has_mom: bool = True) -> dict[str, ShortSignal]:
        """Return bearish candidates: ticker -> ShortSignal.

        A ticker qualifies when:
          * mode allows shorts in this regime (hedge => risk_off only), AND
          * S_short >= entry_threshold (new entries), OR
          * the ticker is already short AND S_short >= exit_threshold
            (kept so the constructor maintains/resize it; below exit the
            target drops to 0 and the constructor COVERs).
        """
        if not self.enabled:
            return {}
        mode = getattr(self.settings, "mode", "off")
        if mode == "hedge" and regime != "risk_off":
            return {}
        entry_thr = float(getattr(self.settings, "entry_threshold", 0.85) or 0.85)
        exit_thr = float(getattr(self.settings, "exit_threshold", 0.40) or 0.40)
        max_names = int(getattr(self.settings, "max_names", 3) or 3)
        held = held_shorts or {}

        scored: list[ShortSignal] = []
        for e in evals:
            bs = self.bearish_score(e.rank_score, e.news_score, e.momentum_score,
                                    has_rank, has_news, has_mom)
            try:
                vol = float(vol_fn(e.ticker) or 0.0)
            except Exception:  # noqa: BLE001
                vol = 0.0
            if vol <= 0:
                continue
            is_entry = bs >= entry_thr
            is_hold = int(held.get(e.ticker, 0)) > 0 and bs >= exit_thr
            if not (is_entry or is_hold):
                continue
            scored.append(ShortSignal(
                ticker=e.ticker, bearish_score=bs, vol=vol,
                rank_score=e.rank_score, news_score=e.news_score,
                momentum_score=e.momentum_score,
            ))
        # Rank by bearish conviction; entries capped at max_names, held
        # shorts always kept (never strand a position without a target).
        scored.sort(key=lambda s: s.bearish_score, reverse=True)
        out: dict[str, ShortSignal] = {}
        n_entries = 0
        for s in scored:
            if int(held.get(s.ticker, 0)) > 0:
                out[s.ticker] = s
            elif n_entries < max_names and s.bearish_score >= entry_thr:
                out[s.ticker] = s
                n_entries += 1
        if out:
            _log.info("short_alpha_candidates",
                      n=len(out), regime=regime, mode=mode,
                      tickers={t: round(s.bearish_score, 3) for t, s in out.items()},
                      component="short_engine")
        return out

    # ------------------------------------------------------------------
    def target_weights(self, signals: dict[str, ShortSignal], *,
                       equity: float, max_short_gross: float,
                       max_weight_name: float) -> dict[str, float]:
        """Inverse-vol NEGATIVE weights for the constructor.

        Gross short exposure <= max_short_gross; per-name weight <=
        min(max_weight_name, per-name short cap). Empty dict when no
        candidates — the short allocation simply stays unused.
        """
        if not signals or equity <= 0:
            return {}
        gross = max(0.0, float(max_short_gross))
        cap = min(float(getattr(self.settings, "max_weight_short", 0.10) or 0.10),
                  float(max_weight_name))
        if gross <= 0 or cap <= 0:
            return {}
        inv = {t: 1.0 / max(s.vol, 1e-9) for t, s in signals.items()}
        tot = sum(inv.values())
        if tot <= 0:
            return {}
        weights = {t: -min(cap, gross * (iv / tot)) for t, iv in inv.items()}
        return weights
