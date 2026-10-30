"""bot.main — entrypoint: `python -m bot.main --env sandbox|live [--dry-run]`

Wires every component per docs/02 §3:

    Kafka -> Queue -> DayAssembler -> DayBundle
                                      |
                                      v
                                StrategyStack  --> TradeDeltas
                                      |
                                      v
                                OrderManager -> Ledger
                                                   |
                                                   v
                                              Journal (JSONL)

Also runs:
    - Heartbeat (acks preflight_tick + watchdog)
    - StageWatcher (drives Lifecycle transitions)
    - Daily reconciliation (GET /portfolio every N days)

Usage:
    BOT_ACCESS_TOKEN=... SERVER_IP=192.168.65.157 TEAM_USERNAME=MN2632 \\
        python -m bot.main --env sandbox
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import time
from datetime import date as _date, datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

import numpy as np

from .config import Settings, fail_fast, load_settings
from .contracts import (
    DayBundle,
    OrderRequest,
    PortfolioResponse,
    RejectCode,
    StageInfo,
)
from .execution.api_client import APIClient
from .execution.order_manager import OrderManager
from .execution.rate_limiter import RateLimiter
from .execution.validator import OrderValidator
from .ingest.day_assembler import DayAssembler
from .ingest.heartbeat import Heartbeat
from .ingest.kafka_consumer import KafkaIngest
from .ingest.stage_watcher import StageWatcher
from .journal.journal import DecisionJournal
from .lifecycle import BotState, Lifecycle
from .models.ensemble import Ensemble
from .models.regime import RegimeDetector
from .models.vol import VolEstimator
from .portfolio.constructor import (
    PortfolioTarget,
    TickerScore,
    build_target_weights,
    build_trade_deltas,
)
from .portfolio.ledger import Ledger
from .portfolio.risk import RiskManager
from .portfolio.rl_tuner import RLTuner, build_tuner_from_settings, build_bandit_from_settings
from .portfolio.settlement import SettlementCalendar
from .portfolio.tax_engine import TaxEngineConfig
from .features import RingBuffer, build_model1_features, build_model3_features
from .signals import SignalsBundle
from .signals.ensemble import SignalsEnsemble
from .signals.event_tagger import EventTagger
from .signals.rules_fallback import RulesFallback
from .signals.volatility import VolatilityModule

# ---------------------------------------------------------------------------
# Logging setup — JSON to stdout (per docs/10 §3)
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
    cache_logger_on_first_use=True,
)
_log = structlog.get_logger("main")


# ---------------------------------------------------------------------------
class TradingBot:
    """Top-level orchestrator. Owns every component's lifecycle."""

    def __init__(self, settings: Settings) -> None:
        """Construct every component in dependency order (no I/O here).

        Wiring order mirrors the pipeline: lifecycle/gate -> ledger/journal ->
        risk -> API/rate-limiter/validator -> signals (M1a/M4/M5/rules +
        ensemble) -> optional ML models -> RL tuner -> Kafka/assembler ->
        heartbeat/stage-watcher -> order manager. Network starts in start().
        """
        self.settings = settings
        self.lifecycle = Lifecycle(settings)
        self.ledger = Ledger(
            cash=settings.tax.brokerage_per_order,  # placeholder; overwritten on reconcile
            initial_cash=settings.tax.brokerage_per_order,
        )
        self.journal = DecisionJournal()
        self.calendar = SettlementCalendar(
            interval_trading_days=settings.settlement.interval_trading_days,
        )
        self.risk = RiskManager(settings.risk, self.calendar)
        self.api = APIClient(settings)
        self.rate_limiter = RateLimiter(
            rate_per_min=settings.api.rate_per_min,
            capacity=min(10, max(2, settings.api.rate_per_min // 12)),
        )
        self.validator = OrderValidator(settings, self.risk)
        self.ensemble = Ensemble(settings.strategy)
        # ------------------------------------------------------------------
        # Signals package — M1a / M4 / R / ensemble (docs/05 §3, §5, §6).
        # M5 regime detection lives in bot.models.regime — wired here.
        # ------------------------------------------------------------------
        self.event_tagger = EventTagger(
            magnitude_scale=settings.signals.event_tagger.magnitude_scale,
        )
        self.volatility = VolatilityModule(
            vol_window=20,
            fallback_vol=0.02,
            atr_window=14,
        )
        # Back-compat alias for any callers still referencing `self.vol`.
        self.vol = self.volatility.vol_estimator
        # M5: market regime detector — gates gross long exposure.
        self.regime = RegimeDetector()
        self.rules_fallback = RulesFallback(tagger=self.event_tagger)
        # P1-01: per-ticker rolling OHLCV history — feeds M1 + M3 feature
        # windows each simulated day. 280-day window covers M1's longest
        # 252-day lookback plus slack.
        self.ring_buffer = RingBuffer(window=280)
        # Optional model1 (ranker) + model2 (news scorer) loaders. Failures
        # are caught -> SignalsEnsemble treats them as missing and falls back
        # to the rules sub-scorers (P0 path).
        rank_fn = self._try_build_model1_fn()
        news_fn = self._try_build_model2_fn()
        mom_fn = self._try_build_model3_fn()
        # --- Strategy-improvement overlays (all opt-in, fail-soft) ---------
        from .signals.overlays import (
            make_horizon_blend_mom_fn,
            make_mean_reversion_mom_fn,
        )
        strat = settings.strategy
        # Momentum horizon blend: 5d/10d/20d weighted z-scores replace the
        # single-horizon mom sub-score (model3 scorer itself is untouched).
        if getattr(strat, "mom_horizon_weights", None):
            mom_fn = make_horizon_blend_mom_fn(
                mom_fn, self.ring_buffer, strat.mom_horizon_weights)
        # Mean reversion in risk_off: replaces the mom sub-score with the
        # negative 10d return z-score. Reads self._current_regime, which
        # _on_day sets BEFORE evaluate() each day.
        if getattr(strat, "mean_reversion_in_risk_off", False):
            mom_fn = make_mean_reversion_mom_fn(
                mom_fn, self.ring_buffer,
                lambda: getattr(self, "_current_regime", "neutral"))
        self.signals_ensemble = SignalsEnsemble(
            settings.signals,
            rank_fn=rank_fn,
            news_fn=news_fn,
            mom_fn=mom_fn,  # momentum now has its own model3 scorer
            rules=self.rules_fallback,
            vol_fn=self.volatility.vol,  # used iff signals.vol_scale_scores
        )
        # Rolling per-ticker ensemble score history (entry-timing gate) and
        # yesterday's top-3 set (fresh-entry gate). Seeded empty — gates
        # no-op until enough history accrues.
        self._score_hist: dict[str, list[float]] = {}
        self._prev_top3: Optional[set[str]] = None
        # P3-18: statistically-optimized weights (Monte Carlo + Markowitz-on-IC,
        # see scripts/optimize_weights.py). When weights_mode == "optimized"
        # and the JSON exists, it overrides the static YAML weights AND is
        # injected as the RL tuner's arm 0 so online learning starts from the
        # data-estimated optimum rather than a hand-picked guess.
        optimized_w = self._load_optimized_weights(settings)
        if optimized_w is not None:
            self.signals_ensemble.settings.weights = dict(optimized_w)

        # P3-15: optional RL tuner — UCB1 bandit over ensemble-weight
        # triples. When enabled, we override signals.weights before each
        # evaluate() call and feed back realized P&L after fills.
        self.rl_tuner = build_tuner_from_settings(settings.strategy)
        if self.rl_tuner is not None:
            if optimized_w is not None:
                from .portfolio.rl_tuner import Arm
                self.rl_tuner.arms[0] = Arm(weights=dict(optimized_w))
            # Force the first selection so initial weights are recorded.
            self.rl_tuner.force_select()
            _log.info("rl_tuner_ready", n_arms=len(self.rl_tuner.arms),
                      current=self.rl_tuner.current_weights(), component="main")

        # P3-22: per-ticker RL bandit — penalises chronic losers / mildly
        # rewards consistent winners via a bounded score multiplier, fed by
        # realised P&L on every SELL/COVER fill (see order_manager).
        self.ticker_bandit = build_bandit_from_settings(settings.strategy)
        if self.ticker_bandit is not None:
            _log.info("ticker_bandit_ready", component="main")

        self.signals = SignalsBundle(
            tagger=self.event_tagger,
            volatility=self.volatility,
            rules=self.rules_fallback,
            ensemble=self.signals_ensemble,
        )
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=20_000)
        self.kafka = KafkaIngest(settings, self.queue)
        self.assembler = DayAssembler(
            settings=settings,
            queue=self.queue,
            on_day=self._on_day,
            expected_tickers_provider=self._expected_tickers,
        )
        self.heartbeat = Heartbeat(settings, self._api_post)
        self.stage_watcher = StageWatcher(settings, self._api_get, self.lifecycle.on_stage_change)
        self.order_manager = OrderManager(
            settings=settings,
            api=self.api,
            rate_limiter=self.rate_limiter,
            validator=self.validator,
            ledger=self.ledger,
            journal=self.journal,
            on_state_divergence=self._on_state_divergence,
            ticker_bandit=self.ticker_bandit,
        )
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._universe: set[str] = set()
        self._sector_of: dict[str, str] = {}
        # P3-24: peak close per long ticker since entry (trailing SL). Seeded
        # lazily at first sight each day; entries dropped when flat.
        self._trail_peak: dict[str, float] = {}
        # P3-25: per-ticker partial-TP rung stage (next rung index to fire;
        # reset when the position goes flat).
        self._ptp_stage: dict[str, int] = {}
        # P4-01: Short Alpha Engine — separate bearish pipeline (off unless
        # strategy.short_engine.enabled in YAML).
        from .portfolio.short_engine import ShortAlphaEngine
        _se_cfg = getattr(self.settings.strategy, "short_engine", None)
        self.short_engine = ShortAlphaEngine(_se_cfg) if _se_cfg is not None else None
        # P4-02: last sim date each ticker was SOLD (re-entry cooldown).
        self._last_sell_date: dict = {}
        # P4-03: per-ticker short profit-rung stage (reset when flat).
        self._sptp_stage: dict[str, int] = {}
        # P4-04: tickers whose giveback-guard stop is armed (saw +arm_pct).
        self._gg_armed: set = set()
        # P4-06: first sim date each ticker's lossy resize sell was deferred
        # (underwater resize guard; reset when the lot predates the entry).
        self._resize_defer: dict = {}

    @staticmethod
    def _load_optimized_weights(settings) -> Optional[dict[str, float]]:
        """P3-18: read configs/optimized_weights.json if weights_mode=optimized."""
        if getattr(settings.strategy, "weights_mode", "static") != "optimized":
            return None
        import json as _json
        import os as _os
        path = getattr(settings.strategy, "optimized_weights_path",
                       "configs/optimized_weights.json")
        if not _os.path.exists(path):
            _log.warning("optimized_weights_missing", path=path,
                         fallback="static yaml weights", component="main")
            return None
        try:
            with open(path, "r", encoding="utf-8-sig") as fh:
                payload = _json.load(fh)
            w = {str(k): float(v) for k, v in (payload.get("weights") or {}).items()}
            if not w or abs(sum(w.values()) - 1.0) > 0.05:
                _log.warning("optimized_weights_invalid", path=path, weights=w,
                             component="main")
                return None
            _log.info("optimized_weights_loaded", weights=w,
                      n_days=payload.get("n_days"),
                      ic_ir=payload.get("ic_ir"), component="main")
            return w
        except Exception as e:  # noqa: BLE001
            _log.warning("optimized_weights_load_failed", err=repr(e),
                         path=path, component="main")
            return None

    # ------------------------------------------------------------------
    # Optional model loaders — fail-soft; if the artifact isn't exported yet
    # we just return None and SignalsEnsemble falls back to the rules path.
    # ------------------------------------------------------------------
    def _try_build_model1_fn(self):
        """Build a closure that scores tickers with the trained M1 CSTT ranker.

        The closure uses `self.ring_buffer` to build [L=64, F=28] feature
        windows on each call. If the ring buffer doesn't have enough history
        (≥64 days) for a ticker, that ticker is omitted from the score dict;
        SignalsEnsemble treats it as a missing rank sub-score and renormalizes
        over the remaining signals (graceful degradation per docs/04 §4).
        """
        try:
            from .models.model1_onnx import Model1Ranker, RulesRanker
            ranker = Model1Ranker()
            rules_ranker = RulesRanker()
            if not ranker.ready:
                _log.info("model1_onnx_not_ready_using_rules_ranker",
                          component="main")
                return rules_ranker
            _log.info("model1_onnx_ready",
                      component="main",
                      members=ranker.n_members if hasattr(ranker, "n_members") else None)

            def _rank_fn(bundle: DayBundle) -> dict[str, float]:
                try:
                    tickers = list(bundle.ticks.keys())
                    if not tickers:
                        return {}
                    windows, ready = build_model1_features(
                        self.ring_buffer, tickers,
                    )
                    if not ready:
                        return {}
                    feats = {t: windows[t] for t in ready}
                    scores = ranker.score(feats, tickers=tickers)
                    if scores:
                        _log.info(
                            "m1_score_computed",
                            n_tickers=len(scores),
                            mean=round(sum(scores.values()) / max(1, len(scores)), 6),
                            component="main",
                        )
                    return scores
                except Exception as e:  # noqa: BLE001
                    _log.warning("model1_score_failed",
                                 err=repr(e), component="main")
                    return {}

            return _rank_fn
        except Exception as e:  # noqa: BLE001
            _log.warning("model1_load_failed", err=repr(e), component="main")
            return None

    def _try_build_model2_fn(self):
        """Build the M2 news scorer closure (FinBERT-based NRIM).

        Falls back per-call to the keyword scorer if model2 errors mid-run,
        and returns None (rules path) if the artifact never loads.
        """
        try:
            from .models.model2_bridge import Model2NewsScorer, KeywordNewsScorer
            scorer = Model2NewsScorer()
            keyword = KeywordNewsScorer()
            if scorer.ready:
                _log.info("model2_bridge_ready",
                          n_members=len(scorer._scorer.members), component="main")
            else:
                _log.warning("model2_bridge_not_ready_using_keyword_fallback",
                             err=scorer._last_err, component="main")

            def _news_fn(bundle: DayBundle) -> dict[str, float]:
                try:
                    if scorer.ready:
                        feats = scorer.score(bundle)
                        out = {t: feats[t]["news_score"] for t in feats}
                        n_nonzero = sum(1 for v in out.values() if v != 0.0)
                        _log.info("m2_score_computed", n_tickers=len(out),
                                  n_nonzero=n_nonzero, component="main")
                        return out
                except Exception as e:  # noqa: BLE001
                    _log.warning("model2_score_failed", err=repr(e), component="main")
                # Fallback path: keyword scorer
                kw = keyword.score(bundle)
                return {t: kw[t]["news_score"] for t in kw}

            # News decay (opt-in): exponential decay of per-ticker sentiment
            # with half-life `news_half_life_days` sim days. 0 = raw
            # same-day score (no wrapper).
            half_life = float(getattr(self.settings.strategy,
                                      "news_half_life_days", 0.0) or 0.0)
            if half_life > 0:
                from .signals.overlays import make_decayed_news_fn
                return make_decayed_news_fn(_news_fn, half_life)
            return _news_fn
        except Exception as e:  # noqa: BLE001
            _log.warning("model2_load_failed", err=repr(e), component="main")
            return None

    def _try_build_model3_fn(self):
        """Build a closure that scores tickers with the trained M3 DMN momentum LSTM.

        Uses `self.ring_buffer` to build [L=63, F=16] feature windows each
        call. Tickers without enough history (≥63 days) are omitted; the
        ensemble renormalises over remaining signals.
        """
        try:
            from .models.model3_runtime import Model3Scorer
            scorer = Model3Scorer()
            if not scorer.ready:
                _log.info("model3_runtime_not_ready_using_rules_mom",
                          component="main")
                # Build a rules-based momentum fallback that uses the ring
                # buffer directly — same shape as Model3Scorer.score.
                rb = self.ring_buffer

                def _rules_mom_fn(bundle: DayBundle) -> dict[str, float]:
                    out: dict[str, float] = {}
                    for t in bundle.ticks.keys():
                        f = rb.frame(t, n=20)
                        if f is None or len(f) < 20:
                            continue
                        closes = f["close"].to_numpy(dtype=np.float64)
                        log_c = np.log(closes)
                        r20 = log_c[-1] - log_c[0]
                        vol = float(np.std(np.diff(log_c))) + 1e-9
                        z = float(r20 / (vol * np.sqrt(20.0)))
                        out[t] = float(max(-1.0, min(1.0, z / 2.0)))
                    return out

                return _rules_mom_fn

            _log.info("model3_runtime_ready",
                      component="main",
                      members=scorer.n_members,
                      seq_len=scorer.meta.seq_len if scorer.meta else None,
                      n_features=scorer.meta.n_features if scorer.meta else None)

            def _mom_fn(bundle: DayBundle) -> dict[str, float]:
                try:
                    tickers = list(bundle.ticks.keys())
                    if not tickers:
                        return {}
                    windows, ready = build_model3_features(
                        self.ring_buffer, tickers,
                    )
                    if not ready:
                        return {}
                    feats = {t: windows[t] for t in ready}
                    scores = scorer.score(feats)
                    if scores:
                        _log.info(
                            "m3_score_computed",
                            n_tickers=len(scores),
                            mean=round(sum(scores.values()) / max(1, len(scores)), 6),
                            component="main",
                        )
                    return scores
                except Exception as e:  # noqa: BLE001
                    _log.warning("model3_score_failed",
                                 err=repr(e), component="main")
                    return {}

            return _mom_fn
        except Exception as e:  # noqa: BLE001
            _log.warning("model3_load_failed", err=repr(e), component="main")
            return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Boot sequence: API session -> DISCOVER -> universe -> ring-buffer
        backfill -> first reconcile (cash truth) -> Kafka + assembler +
        heartbeat + stage-watcher -> background loops."""
        _log.info("bot_starting",
                  env=self.settings.env,
                  team=self.settings.team_username,
                  broker=self.settings.broker,
                  group=self.settings.group_id,
                  dry_run=self.settings.dry_run,
                  component="main")
        await self.api.start()
        await self.lifecycle.transition(BotState.DISCOVER, reason="bot.start")
        # Try to populate universe + reconcile cash once at startup
        await self._discover_universe()
        # Warm M1 + M3 ring buffer with ~280 days of OHLCV per ticker so
        # the trained models can produce scores from day 1.
        await self._backfill_ring_buffer()
        await self._reconcile()
        # Seed the ticker bandit from the server orderbook so chronic-loser
        # vetoes survive restarts (bandit state is in-memory otherwise).
        await self._seed_bandit_from_orderbook()
        await self.kafka.start()
        await self.assembler.start()
        await self.heartbeat.start()
        await self.stage_watcher.start()

        self._tasks = [
            asyncio.create_task(self._preflight_router(), name="preflight-router"),
            asyncio.create_task(self._health_loop(), name="health"),
            asyncio.create_task(self._settle_loop(), name="settle"),
        ]

    async def stop(self) -> None:
        """Graceful shutdown: stop ingest first, cancel loops, close journal,
        then transition to SHUTDOWN."""
        _log.info("bot_stopping", component="main")
        self._stop.set()
        await self.assembler.stop()
        await self.kafka.stop()
        await self.stage_watcher.stop()
        await self.heartbeat.stop()
        await self.api.stop()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self.journal.close()
        await self.lifecycle.transition(BotState.SHUTDOWN, reason="bot.stop")

    async def wait_until_stopped(self) -> None:
        """Block until _stop is set (used by the CLI runner)."""
        await self._stop.wait()

    # ------------------------------------------------------------------
    # API helpers used by heartbeat / stage_watcher
    # ------------------------------------------------------------------
    async def _api_get(self, path: str) -> dict:
        return await self.api.get(path)

    async def _api_post(self, path: str, body: dict) -> dict:
        return await self.api.post(path, body)

    # ------------------------------------------------------------------
    # Universe discovery
    # ------------------------------------------------------------------
    async def _discover_universe(self, retries: int = 12, delay_s: float = 5.0) -> None:
        """Fetch /api/v1/stocks. Retries: the endpoint can 504/timeout while the
        platform warms up (observed live), and without a universe the bot has no
        tickers to trade and no backfill for M1/M3."""
        for attempt in range(1, retries + 1):
            try:
                raw = await self.api.get("/api/v1/stocks")
                items = raw.get("stocks", raw) if isinstance(raw, dict) else raw
                if isinstance(items, list) and items:
                    self._universe = {it.get("ticker") for it in items if isinstance(it, dict) and it.get("ticker")}
                    self._sector_of = {it.get("ticker"): (it.get("sector") or "")
                                        for it in items if isinstance(it, dict) and it.get("ticker")}
                    _log.info("universe_discovered", n=len(self._universe),
                              attempt=attempt, component="main")
                    return
                _log.warning("universe_discovery_failed", shape=type(items).__name__,
                             attempt=attempt, component="main")
            except Exception as e:  # noqa: BLE001
                _log.warning("universe_discovery_error", err=repr(e),
                             attempt=attempt, component="main")
            await asyncio.sleep(delay_s)
        _log.error("universe_discovery_exhausted", retries=retries, component="main")

    def _expected_tickers(self) -> set[str]:
        """Provider hook for DayAssembler's complete-by-count check."""
        return self._universe

    # ------------------------------------------------------------------
    # Historical backfill (P1-01) — warm M1+M3 ring buffer before trading
    # ------------------------------------------------------------------
    async def _backfill_ring_buffer(self) -> None:
        """Pull ~280 days of OHLCV per ticker from /stocks/:t/history and
        seed `self.ring_buffer`. Without this, M1 (needs 64 days) and M3
        (needs 63 days) cannot produce scores on day 1.
        """
        from .features import Candle
        from datetime import date as _date
        from datetime import datetime as _dt

        if not self._universe:
            _log.warning("backfill_skipped_no_universe", component="main")
            return

        target_days = 280
        ok_tickers = 0
        total_candles = 0
        # Sequential, rate-limited at ~2 req/s to stay under 100/min
        for ticker in sorted(self._universe):
            try:
                raw = await self.api.get(
                    f"/api/v1/stocks/{ticker}/history",
                    params={"days": target_days},
                )
            except Exception as e:  # noqa: BLE001
                _log.warning("backfill_history_failed",
                             ticker=ticker, err=repr(e), component="main")
                continue

            # Try common shapes: {prices: [...]}, {history: [...]}, [...]
            rows = None
            if isinstance(raw, list):
                rows = raw
            elif isinstance(raw, dict):
                for k in ("prices", "history", "data", "candles"):
                    if k in raw and isinstance(raw[k], list):
                        rows = raw[k]
                        break
                if rows is None and "ticker" in raw:
                    # Single-day shape — treat the dict itself as the row
                    rows = [raw]
            if not rows:
                _log.warning("backfill_no_rows", ticker=ticker, component="main")
                continue

            n_pushed = 0
            for r in rows:
                if not isinstance(r, dict):
                    continue
                try:
                    d_str = r.get("date") or r.get("simulated_date")
                    if d_str is None:
                        continue
                    if isinstance(d_str, str):
                        # Handle ISO "2024-01-01" or "2024-01-01T00:00:00"
                        d = _dt.fromisoformat(d_str.replace("Z", "+00:00")).date() \
                            if "T" in d_str else _date.fromisoformat(d_str[:10])
                    else:
                        d = d_str
                    o = float(r.get("open", 0))
                    h = float(r.get("high", 0))
                    lo = float(r.get("low", 0))
                    c = float(r.get("close", 0))
                    v = float(r.get("volume", 0))
                    if c <= 0:
                        continue
                    self.ring_buffer.push_candle(ticker, Candle(
                        date=d, open=o, high=h, low=lo, close=c, volume=v,
                    ))
                    n_pushed += 1
                except Exception:  # noqa: BLE001
                    continue
            if n_pushed > 0:
                ok_tickers += 1
                total_candles += n_pushed
            _log.info("backfill_ticker",
                      ticker=ticker, n_candles=n_pushed, component="main")
            # Rate-limit: ~2 req/s
            await asyncio.sleep(0.5)

        _log.info("backfill_complete",
                  ok_tickers=ok_tickers,
                  total_candles=total_candles,
                  component="main")

    # ------------------------------------------------------------------
    # Bandit boot-seeding from the server orderbook
    # ------------------------------------------------------------------
    @staticmethod
    def _fifo_realised(orders: list[dict]) -> list[tuple[str, float, object]]:
        """FIFO-match FILLED BUY->SELL per ticker; return (ticker, gross_pnl,
        sell_sim_date) per closed lot. Pure — unit-tested. Used to seed the
        ticker bandit from the server orderbook so chronic losers are vetoed
        from the first trade instead of re-learning after every restart."""
        from collections import defaultdict as _dd
        lots: dict[str, list[list]] = _dd(list)
        out: list[tuple[str, float, object]] = []
        for o in sorted(orders, key=lambda x: (x.get("simulated_date") or "",
                                               x.get("ticker") or "")):
            if o.get("status") != "FILLED":
                continue
            t = o.get("ticker")
            act = (o.get("action") or "").upper()
            q = float(o.get("quantity") or 0)
            p = float(o.get("execution_price") or 0)
            if not t or q <= 0 or p <= 0:
                continue
            if act == "BUY":
                lots[t].append([q, p])
            elif act == "SELL":
                rem = q
                while rem > 1e-9 and lots[t]:
                    lot = lots[t][0]
                    take = min(lot[0], rem)
                    out.append((t, take * (p - lot[1]), o.get("simulated_date")))
                    lot[0] -= take
                    rem -= take
                    if lot[0] <= 1e-9:
                        lots[t].pop(0)
        return out

    async def _seed_bandit_from_orderbook(self) -> None:
        """Replay the account's order history into the ticker bandit at boot.

        The bandit's edge estimates are in-memory; without seeding, every
        restart resets them and chronic losers (e.g. 0/10 winners on one
        name) keep getting bought until the veto re-accumulates. Server
        orderbook is the ground truth; on a fresh evaluation account this
        is simply a no-op (empty history)."""
        if self.ticker_bandit is None:
            return
        try:
            raw = await self.api.get("/api/v1/orders?limit=10000")
            orders = (raw.get("orders") if isinstance(raw, dict) else raw) or []
            realised = self._fifo_realised(orders)
            for ticker, pnl, day in realised:
                self.ticker_bandit.observe(ticker, pnl, day=day)
            vetoed = [t for t in {t for t, _, _ in realised}
                      if self.ticker_bandit.is_vetoed(t)]
            _log.info("ticker_bandit_seeded", trades=len(realised),
                      tickers=len({t for t, _, _ in realised}),
                      vetoed=sorted(vetoed), component="main")
        except Exception as e:  # noqa: BLE001
            _log.warning("ticker_bandit_seed_failed", err=repr(e), component="main")

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------
    async def _reconcile(self) -> None:
        """Re-sync local state from GET /api/v1/portfolio (server is truth).

        Side effects: overwrites ledger cash/lots/shorts, refreshes the
        _server_holdings/_server_shorts/_server_cash clamp caches (P3-20/21),
        seeds the DD breaker peak (P3-14), feeds equity deltas to the RL
        tuner (P3-15), triggers idle-cash deploy (P3-17), and updates the
        consecutive-loss-day streak (P3-13). Never raises — failures are
        logged so a transient blip doesn't kill the health loop.
        """
        try:
            raw = await self.api.get("/api/v1/portfolio")
            # Portal returns `holdings` (longs list). Coerce into the
            # `positions` field expected by PortfolioResponse before
            # validation. Also accept `short_positions` / `open_shorts`
            # for shorts (variants seen across portal versions).
            if isinstance(raw, dict):
                coerced = dict(raw)
                if "positions" not in coerced and "holdings" in coerced:
                    coerced["positions"] = coerced["holdings"]
                if "shorts" not in coerced:
                    for key in ("short_positions", "open_shorts", "shorts_list"):
                        if key in coerced:
                            coerced["shorts"] = coerced[key]
                            break
                raw = coerced
            p = PortfolioResponse.model_validate(raw)
            self.ledger.cash = p.cash
            self.ledger.initial_cash = p.cash
            # P3-21: server cash cache for BUY clamps between reconciles.
            self._server_cash = float(p.cash)
            # Reset positions from server snapshot (drift correction)
            from .portfolio.ledger import Lot, ShortPos
            from collections import defaultdict, deque
            self.ledger.lots = defaultdict(deque)
            self.ledger.shorts = {}
            # P3-20: cache SERVER holdings/shorts verbatim. The server can
            # report NEGATIVE holdings (a short carried inside `holdings`,
            # e.g. -1166). Every SELL/COVER is clamped against this cache
            # before submission so we never send an order the server would
            # reject with INSUFFICIENT_MARGIN / INSUFFICIENT_HOLDINGS.
            self._server_holdings = {}
            self._server_shorts = {}
            for pos in p.positions:
                self._server_holdings[pos.ticker] = int(pos.quantity)
                if pos.quantity < 0:
                    # Server-side short carried as a negative holding.
                    self._server_shorts[pos.ticker] = int(-pos.quantity)
                    self.ledger.shorts[pos.ticker] = ShortPos(
                        ticker=pos.ticker, qty=int(-pos.quantity),
                        entry_price=pos.avg_price,
                        open_date=p.timestamp.date() if hasattr(p.timestamp, "date") else _date.today(),
                        margin_locked=0.0,
                    )
                    continue
                self.ledger.lots[pos.ticker].append(Lot(
                    ticker=pos.ticker, qty=pos.quantity, price=pos.avg_price,
                    buy_date=p.timestamp.date() if hasattr(p.timestamp, "date") else _date.today(),
                    fees=0.0,
                ))
                if pos.current_price:
                    self.ledger.last_close[pos.ticker] = pos.current_price
            for sp in p.shorts:
                self._server_shorts[sp.ticker] = int(sp.quantity)
                self.ledger.shorts[sp.ticker] = ShortPos(
                    ticker=sp.ticker, qty=sp.quantity, entry_price=sp.entry_price,
                    open_date=sp.open_date, margin_locked=sp.margin_locked,
                )
                self.ledger.last_close[sp.ticker] = sp.entry_price
            _log.info("reconciled", cash=p.cash, equity=p.equity,
                      longs=len(p.positions), shorts=len(p.shorts), component="main")
            # P3-19: always-visible holdings snapshot (legacy positions
            # included) with unrealised P&L per name.
            _snap = []
            for pos in p.positions:
                px = pos.current_price or self.ledger.last_close.get(pos.ticker, 0.0)
                upnl = (px - pos.avg_price) * pos.quantity if px > 0 else 0.0
                _snap.append({"ticker": pos.ticker, "qty": pos.quantity,
                              "avg": round(pos.avg_price, 2), "last": round(px, 2),
                              "unrealized_pnl": round(upnl, 0)})
            for sp in p.shorts:
                px = self.ledger.last_close.get(sp.ticker, sp.entry_price)
                _snap.append({"ticker": sp.ticker, "qty": -sp.quantity,
                              "avg": round(sp.entry_price, 2), "last": round(px, 2),
                              "unrealized_pnl": round((sp.entry_price - px) * sp.quantity, 0)})
            _log.info("holdings_snapshot", positions=_snap,
                      n=len(_snap), component="main")
            if not getattr(self, "_startup_holdings_logged", False):
                self._startup_holdings_logged = True
                _log.info("startup_holdings_loaded", positions=_snap,
                          cash=p.cash, equity=p.equity, component="main")
            # P3-18: server equity is the source of truth for the DD breaker.
            self._last_server_equity = float(p.equity)
            # P3-14: seed risk peak_equity on first successful reconcile so
            # the DD breaker doesn't false-positive on the zero-equity boot
            # state.
            self.risk.seed_equity(p.equity)
            # P3-14: maintain trailing peak for wealth-tracking (the risk
            # manager already tracks this, but expose it on the bot for the
            # RL tuner / observability).
            self._peak_equity = max(getattr(self, "_peak_equity", 0.0), p.equity)
            # P3-15: feed realised P&L delta back to the RL tuner. We track
            # the equity delta since last reconcile; that's the cleanest
            # signal the tuner gets (per-day reward).
            if self.rl_tuner is not None:
                prev = getattr(self, "_prev_equity_for_rl", None)
                if prev is not None:
                    delta = float(p.equity) - float(prev)
                    self.rl_tuner.observe(delta)
                self._prev_equity_for_rl = float(p.equity)
            # P3-17: realized profits land in server cash after fills. Deploy
            # idle cash immediately (between day-bundles) so utilisation
            # stays at the target and freed capital keeps compounding.
            await self._maybe_deploy_idle_cash(p)
            # P3-13 cool-down: count consecutive losing days; reset on a green day.
            # A "day" is one reconcile-to-reconcile delta. We track the last seen
            # realized_pnl from the portfolio payload (when available) and the
            # day we last saw it; on a new day, compare against the prior and
            # increment the streak if it dropped. Stops opening new trades when
            # the streak hits `risk.max_consecutive_loss_pause`.
            try:
                from datetime import date as _d
                today = _date.today()
                eq = float(p.equity)
                prev_eq = getattr(self, "_last_reconcile_equity", None)
                if prev_eq is not None and today != getattr(self, "_last_reconcile_day", today):
                    if eq < prev_eq:
                        self._consec_loss_days = int(getattr(self, "_consec_loss_days", 0)) + 1
                    elif eq > prev_eq:
                        self._consec_loss_days = 0
                    self._last_reconcile_day = today
                self._last_reconcile_equity = eq
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            _log.warning("reconcile_failed", err=repr(e), component="main")

    # ------------------------------------------------------------------
    # P3-21: optimistic server-cache update on confirmed fills
    # ------------------------------------------------------------------
    def _apply_fill_to_server_cache(self, resp) -> None:
        """Update _server_holdings / _server_shorts / _server_cash after a
        FILLED order so subsequent clamps in the same batch use fresh truth
        (the 15s reconcile would otherwise serve stale numbers)."""
        try:
            fee = float(self.settings.tax.brokerage_per_order)
            notional = float(resp.quantity) * float(resp.price or 0.0)
            if not hasattr(self, "_server_holdings"):
                return
            if resp.action == "BUY":
                self._server_holdings[resp.ticker] = int(
                    self._server_holdings.get(resp.ticker, 0)) + int(resp.quantity)
                if hasattr(self, "_server_cash"):
                    self._server_cash = max(0.0, self._server_cash - notional - fee)
            elif resp.action == "SELL":
                self._server_holdings[resp.ticker] = int(
                    self._server_holdings.get(resp.ticker, 0)) - int(resp.quantity)
                if hasattr(self, "_server_cash"):
                    self._server_cash = self._server_cash + notional - fee
            elif resp.action == "SHORT":
                self._server_shorts[resp.ticker] = int(
                    self._server_shorts.get(resp.ticker, 0)) + int(resp.quantity)
            elif resp.action == "COVER":
                self._server_shorts[resp.ticker] = max(0, int(
                    self._server_shorts.get(resp.ticker, 0)) - int(resp.quantity))
        except Exception as e:  # noqa: BLE001
            _log.warning("server_cache_update_failed", err=repr(e), component="main")

    # ------------------------------------------------------------------
    # P3-20: state-divergence resync
    # ------------------------------------------------------------------
    async def _on_state_divergence(self, reason: str = "", ticker: str = "") -> None:
        """OrderManager reports a possible ledger/server divergence
        (unparseable order response or 4xx rejection). Immediately
        re-reconcile so the ledger + server-holdings cache resync from the
        server's portfolio before the next decision."""
        _log.info("state_divergence_resync", reason=reason, ticker=ticker,
                  component="main")
        try:
            await self._reconcile()
        except Exception as e:  # noqa: BLE001
            _log.warning("divergence_resync_failed", err=repr(e), component="main")

    # ------------------------------------------------------------------
    # P3-17: idle-cash deployer
    # ------------------------------------------------------------------
    async def _maybe_deploy_idle_cash(self, p: PortfolioResponse) -> None:
        """Deploy cash freed by realised profits between day-bundles.

        Called right after every successful reconcile (i.e. every
        reconcile_interval_s, default 15s, so available server cash is
        re-checked on a fixed cadence). If the current cash utilisation
        (1 - cash/equity) is more than `deploy_epsilon` below
        `risk.target_cash_utilization`, walk the conviction ranking and
        place BUYs sized to each name's remaining per-name cap room until
        the gap is closed or the daily order budget is exhausted. All
        existing guards still apply: daily order budget, min notional,
        per-name cap, validator (incl. cash reservation).

        Disabled via `strategy.idle_cash_deploy.enabled: false`.
        """
        try:
            if self.lifecycle.state != BotState.TRADING:
                return
            if self.settings.dry_run:
                return
            cfg = getattr(self.settings.strategy, "idle_cash_deploy", None)
            if cfg is not None and not getattr(cfg, "enabled", True):
                return
            scores = getattr(self, "_last_scores", None)
            if not scores:
                return  # no evaluation yet this session
            equity = float(p.equity or 0.0)
            cash = float(p.cash or 0.0)
            if equity <= 0 or cash <= 0:
                return
            target_util = float(getattr(self.settings.risk, "target_cash_utilization", 0.85) or 0.85)
            # P4-02: cap the deployer's ambition by the regime's gross-long
            # target — otherwise it buys back to 95% every 15s while the
            # day pipeline is trying to hold 40% in risk_off, which is the
            # whipsaw churn that cost -3.0L in <=2d round trips.
            if bool(getattr(self.settings.strategy, "deployer_regime_cap", True)):
                glt = getattr(self.settings.risk, "gross_long_target", None) or {}
                regime_now = getattr(self, "_last_regime", "neutral")
                if regime_now in glt:
                    target_util = min(target_util, float(glt[regime_now]))
            epsilon = float(getattr(cfg, "epsilon", 0.05) if cfg is not None else 0.05)
            # Daily order budget still applies.
            max_orders = self.settings.strategy.max_orders_per_day
            remaining = await self.lifecycle.gate.orders_remaining(max_orders)
            if remaining <= 0:
                return
            excluded = set(self.settings.strategy.exclude_tickers or [])
            # Walk the conviction ranking (not just the top name) so the
            # deployer still works when the best ticker is already at its
            # per-name cap. Up to `remaining` buys per reconcile, each
            # closing part of the utilisation gap — this is what keeps
            # utilisation >80% sustained instead of oscillating 30-60%.
            _today = getattr(self, "_last_sim_date", None)
            # P4-02: re-entry cooldown also binds the deployer — it was the
            # prime re-buyer of just-sold names.
            cd_days = int(getattr(self.settings.strategy,
                                  "reentry_cooldown_days", 0) or 0)
            cd_thr = float(getattr(self.settings.strategy,
                                   "reentry_score_override", 0.90) or 0.90)
            from .signals.overlays import reentry_blocked as _reentry_blocked

            def _cooled_out(tk: str, sc: float) -> bool:
                last = self._last_sell_date.get(tk)
                if last is None or _today is None:
                    return False
                return _reentry_blocked((_today - last).days, cd_days, sc, cd_thr)

            candidates = [
                s for s in sorted(scores, key=lambda s: s.score, reverse=True)
                if s.ticker not in excluded and s.score > 0
                # P3-22b: don't deploy idle cash into bandit-vetoed names.
                and not (self.ticker_bandit is not None
                         and self.ticker_bandit.is_vetoed(s.ticker, today=_today))
                and not _cooled_out(s.ticker, s.score)
            ]
            placed = 0
            pending_notional = 0.0
            for best in candidates:
                if placed >= remaining:
                    break
                equity = float(p.equity or 0.0)
                cash = float(p.cash or 0.0) - pending_notional
                if equity <= 0 or cash <= 0:
                    break
                current_util = max(0.0, min(1.0, 1.0 - cash / equity))
                gap = target_util - current_util
                if gap <= epsilon:
                    break  # close enough to target — done
                price = self.ledger.last_close.get(best.ticker, 0.0) or float(best.price or 0.0)
                if price <= 0:
                    continue
                # Size: utilisation gap in rupees, capped by max_cash_per_trade
                # of available cash, minus brokerage and a small price-drift
                # buffer (1%) so the server never sees an over-cash order.
                gap_cash = gap * equity
                max_frac = float(getattr(self.settings.risk, "max_cash_per_trade", 0.90) or 0.90)
                budget = min(gap_cash, cash * max_frac) * 0.99
                budget -= self.settings.tax.brokerage_per_order
                # Per-name cap: size down to the remaining room for this
                # ticker (instead of giving up entirely when it's full).
                cur_mv = self.ledger.holding_qty(best.ticker) * price
                room = self.settings.risk.max_name_weight * equity - cur_mv
                if room < self.settings.strategy.min_trade_notional:
                    continue  # name already at cap — try the next one
                budget = min(budget, room)
                qty = int(budget / price)
                if qty <= 0:
                    continue
                notional = qty * price
                if notional < self.settings.strategy.min_trade_notional:
                    continue  # gap too small to be worth a trade
                req = OrderRequest(
                    ticker=best.ticker, action="BUY", quantity=qty,
                    date=getattr(self, "_last_sim_date", None) or _date.today(),
                    reason=f"idle_cash_deploy util={current_util:.2%}->{target_util:.2%}",
                )
                _log.info("idle_cash_deploy", ticker=best.ticker, qty=qty,
                          notional=round(notional, 0), util=round(current_util, 4),
                          target=target_util, cash=round(cash, 0), component="main")
                await self.order_manager.submit(
                    req,
                    scores={"ensemble": best.score},
                    reasons=[f"regime={getattr(self, '_last_regime', 'neutral')}",
                             "idle_cash_deploy"],
                )
                await self.lifecycle.gate.record_order()
                pending_notional += notional
                placed += 1
        except Exception as e:  # noqa: BLE001
            _log.warning("idle_cash_deploy_failed", err=repr(e), component="main")

    # ------------------------------------------------------------------
    # Preflight router — drain preflight_ticks out of the queue
    # ------------------------------------------------------------------
    async def _preflight_router(self) -> None:
        """Skim preflight ticks/news off the shared queue and ACK them via
        the heartbeat; anything else is put back for the DayAssembler.

        Necessary because preflight events ride the same Kafka topics as
        market data, and ACK latency matters during the readiness stage.
        """
        while not self._stop.is_set():
            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return
            if event.event_type in ("preflight_tick", "preflight_news"):
                from .contracts import PreflightTickEvent
                if isinstance(event, PreflightTickEvent):
                    await self.heartbeat.on_preflight(event)
            else:
                # Put it back — the assembler will handle it
                try:
                    await self.queue.put(event)
                except asyncio.QueueFull:
                    pass
                await asyncio.sleep(0.01)

    # ------------------------------------------------------------------
    # Health loop — DD breaker, cash buffer check
    # ------------------------------------------------------------------
    async def _health_loop(self) -> None:
        """Every reconcile_interval_s: run the drawdown breaker against
        server-reconciled equity (SAFE_MODE on hard breach) and re-reconcile
        while TRADING."""
        while not self._stop.is_set():
            try:
                # P3-18: use the server-reconciled equity for the DD breaker.
                # self.ledger.equity is a *local mirror* and lags/mismatches
                # the server right after boot (partial fill replay), which
                # caused a false RISK_DD_HARD (drawdown 20%) -> SAFE_MODE
                # bounce on the 2026-10-29 evaluation run. Fall back to the
                # ledger only if no reconcile has completed yet.
                equity = float(getattr(self, "_last_server_equity", 0.0) or 0.0)
                if equity <= 0:
                    equity = self.ledger.equity
                v = self.risk.update_equity(equity)
                if not v.ok:
                    _log.warning("dd_breaker", code=v.code, msg=v.message, component="health")
                    if v.code == "RISK_DD_HARD":
                        await self.lifecycle.enter_safe_mode(reason=v.message or v.code)
                        break
                # P3-14: reconcile at the configured cadence (default 15s)
                # during trading. Was hard-coded 30s.
                if self.lifecycle.state == BotState.TRADING:
                    await self._reconcile()
            except Exception as e:  # noqa: BLE001
                _log.warning("health_loop_err", err=repr(e), component="health")
            try:
                await asyncio.wait_for(self._stop.wait(),
                                       timeout=self.settings.timing.reconcile_interval_s)
                return
            except asyncio.TimeoutError:
                continue

    # ------------------------------------------------------------------
    # Settlement guard loop
    # ------------------------------------------------------------------
    async def _settle_loop(self) -> None:
        """Every 10s: if we're within force_close_before_days of a settlement,
        COVER every open short and close the gate for the day."""
        while not self._stop.is_set():
            try:
                today = self.lifecycle.gate.sim_date
                if today is not None and self.calendar.should_force_close_short(
                    today, lead_days=self.settings.settlement.force_close_before_days,
                ):
                    for ticker in list(self.ledger.shorts.keys()):
                        sp = self.ledger.shorts[ticker]
                        price = self.ledger.last_close.get(ticker, sp.entry_price)
                        req = OrderRequest(
                            ticker=ticker, action="COVER", quantity=sp.qty,
                            date=today, reason="settlement_guard",
                        )
                        await self.order_manager.submit(req)
                    await self.lifecycle.gate.close_for_day()
            except Exception as e:  # noqa: BLE001
                _log.warning("settle_loop_err", err=repr(e), component="settle")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=10.0)
                return
            except asyncio.TimeoutError:
                continue

    # ------------------------------------------------------------------
    # Daily pipeline — invoked by DayAssembler
    # ------------------------------------------------------------------
    async def _on_day(self, bundle: DayBundle) -> None:
        """The daily pipeline, invoked once per assembled DayBundle.

        Steps: (1) open the trading gate (+ cool-down check), (2) push ticks
        into the ring buffer + vol state, (3) M5 regime + ensemble scores
        (RL-tuner weights if enabled), (4) target weights + trade deltas,
        (5) server-truth clamps (P3-20/21) then submit each order,
        (6) journal HOLD records for untouched tickers.
        """
        # 1) Mark trading day on calendar + gate
        self.calendar.observe_trading_day(bundle.simulated_date)
        max_orders = self.settings.strategy.max_orders_per_day
        short_allowed = self.lifecycle.state == BotState.TRADING  # tighten if needed
        await self.lifecycle.gate.open_for_day(bundle.simulated_date, max_orders, short_allowed)

        # P3-13: cool-down gate — pause new entries after N consecutive losing
        # reconcile days. Existing positions are unaffected (we still call
        # _on_day downstream, so risk/validator can act on stops). The gate
        # only suppresses new trade generation by zeroing max_orders.
        pause = int(getattr(self.settings.risk, "max_consecutive_loss_pause", 0) or 0)
        if pause > 0:
            streak = int(getattr(self, "_consec_loss_days", 0))
            if streak >= pause:
                _log.info("cooldown_active", streak=streak, pause=pause,
                          sim_date=bundle.simulated_date.isoformat(), component="main")
                max_orders = 0

        # Wait for TRADING state if not yet (preflight still happening)
        if self.lifecycle.state != BotState.TRADING:
            await self.lifecycle.wait_for(BotState.TRADING, timeout=2.0)

        # 2) Feed the signals bundle's per-ticker vol state. M5 regime runs
        # below against the day's bundle directly.
        # P1-01: push today's ticks into the rolling history before scoring
        # so M1 + M3 have fresh features for the next day.
        try:
            for ticker, tick in bundle.ticks.items():
                self.ring_buffer.push(ticker, tick, bundle.simulated_date)
                # Populate last_close for the validator — it reads
                # `ledger.last_close[ticker]` to compute notional.
                self.ledger.last_close[ticker] = tick.close
        except Exception as e:  # noqa: BLE001
            _log.warning("ring_buffer_push_failed", err=repr(e),
                         component="main")
        self.signals.update_volatility(bundle.simulated_date, bundle.ticks)

        # 3) Build scores via SignalsEnsemble (regime-gated, fail-soft).
        # M5: regime detector runs against today's bundle; the label flows into
        # build_target_weights via `regime=` + `gross_long_target=` so the
        # constructor applies the regime scaling directly.
        try:
            regime_label = self.regime.update(bundle)
            # Expose the regime to the mom_fn overlays (mean reversion in
            # risk_off reads this attribute at evaluate() time).
            self._current_regime = regime_label
            # P3-15: if RL tuner is on, override the ensemble weights for
            # THIS day's evaluation. The tuner updates its weights via UCB1
            # every `update_every_trades` fills, so the ensemble adaptively
            # tilts between ranker-heavy / momentum-heavy / balanced.
            if self.rl_tuner is not None:
                self.signals_ensemble.settings.weights = self.rl_tuner.current_weights()
            evals = self.signals_ensemble.evaluate(bundle, regime_label)
        except Exception as e:  # noqa: BLE001
            _log.error("ensemble_failed", err=repr(e), component="strategy")
            return

        # 4) Build target weights + trade deltas
        scores_for_constructor: list[TickerScore] = []
        for e in evals:
            price = bundle.ticks[e.ticker].close if e.ticker in bundle.ticks else 0.0
            score = e.score
            # P3-22: per-ticker RL bandit — penalise chronic losers (mult
            # toward floor) and mildly reward consistent winners (capped).
            if self.ticker_bandit is not None:
                mult = self.ticker_bandit.multiplier(e.ticker)
                if abs(mult - 1.0) > 1e-9:
                    _log.info("ticker_bandit_adjust", ticker=e.ticker,
                              raw=round(score, 4), mult=round(mult, 3),
                              adjusted=round(score * mult, 4), component="main")
                    score *= mult
            scores_for_constructor.append(TickerScore(
                ticker=e.ticker, score=score,
                vol=self.signals.volatility.vol(e.ticker),
                sector=self._sector_of.get(e.ticker),
                price=price,
            ))

        # P3-17: cache the latest conviction scores + context so the idle-cash
        # deployer (_maybe_deploy_idle_cash) can place extra BUYs between
        # day-bundles when realized profits free up cash.
        self._last_scores = scores_for_constructor
        self._last_regime = regime_label
        self._last_sim_date = bundle.simulated_date

        # --- Entry-gate context (entry timing + fresh entry) ---------------
        # Rank today's tickers by ensemble score; gates below use these sets.
        ranked_today = sorted(evals, key=lambda e: e.score, reverse=True)
        et_cfg = getattr(self.settings.strategy, "entry_timing", None)
        et_enabled = bool(et_cfg and getattr(et_cfg, "enabled", False))
        et_top_n = int(getattr(et_cfg, "top_n", 2) or 2) if et_cfg else 2
        top_n_set = {e.ticker for e in ranked_today[:max(1, et_top_n)]}
        fresh_only = bool(getattr(self.settings.strategy,
                                  "fresh_entry_only", False))
        top3_today = {e.ticker for e in ranked_today[:3]}
        prev_top3 = self._prev_top3  # None -> gate no-ops (fail-soft)

        equity = self.ledger.equity or 1.0
        # M5: regime + gross_long_target are applied inside build_target_weights.
        target_w = build_target_weights(
            scores_for_constructor, self.settings.strategy, equity,
            regime=regime_label,
            sector_caps={"Tech": self.settings.risk.max_sector_weight,
                         "Finance": self.settings.risk.max_sector_weight},
            sector_of=self._sector_of,
            gross_long_target=self.settings.risk.gross_long_target,
            current_holdings={t: q for t, q in self.ledger.holdings.items()},
            current_shorts={t: sp.qty for t, sp in self.ledger.shorts.items()},
            last_close={t: bundle.ticks[t].close for t in bundle.ticks if t in bundle.ticks},
            target_cash_utilization=float(
                getattr(self.settings.risk, "target_cash_utilization", 0.85) or 0.85
            ),
        )
        # P4-01: Short Alpha Engine — separate bearish pipeline. Negative
        # target weights merge into target_w; the constructor emits SHORT /
        # resize-COVER / exit-long-first as needed. Never forced: if no
        # candidate passes its own entry threshold, the short allocation
        # simply stays unused (idle-cash deployer is long-only by design).
        if self.short_engine is not None and self.short_engine.enabled:
            se_cfg = self.short_engine.settings
            held_shorts = {t: sp.qty for t, sp in self.ledger.shorts.items()}
            short_sigs = self.short_engine.evaluate(
                evals, regime=regime_label,
                vol_fn=self.signals.volatility.vol,
                held_shorts=held_shorts,
                has_rank=self.signals_ensemble.rank_fn is not None,
                has_news=self.signals_ensemble.news_fn is not None,
                has_mom=self.signals_ensemble.mom_fn is not None,
            )
            # Hard adverse stop: price up >= adverse_stop_pct vs short entry
            # -> drop the target so the constructor COVERs today.
            stop_pct = float(getattr(se_cfg, "adverse_stop_pct", 0.10) or 0.10)
            for t, sp in list(self.ledger.shorts.items()):
                px = float(bundle.ticks[t].close) if t in bundle.ticks else 0.0
                entry = float(getattr(sp, "entry_price", 0.0) or 0.0)
                if px > 0 and entry > 0 and (px - entry) / entry >= stop_pct:
                    short_sigs.pop(t, None)
                    _log.info("short_adverse_stop", ticker=t,
                              entry=round(entry, 2), px=round(px, 2),
                              adverse_pct=round((px - entry) / entry * 100, 1),
                              component="short_engine")
            short_w = self.short_engine.target_weights(
                short_sigs, equity=equity,
                max_short_gross=float(getattr(self.settings.risk,
                                              "max_short_gross", 0.20) or 0.20),
                max_weight_name=self.settings.strategy.max_weight_name,
            )
            if short_w:
                target_w.update(short_w)

        # Per-name cap is HARD (no relaxation): concentration risk outweighs
        # utilization. Clear any stale override so the static cap binds.
        self.validator.name_cap_override = None
        max_orders_remaining = await self.lifecycle.gate.orders_remaining(max_orders)
        deltas = build_trade_deltas(
            current_holdings={t: q for t, q in self.ledger.holdings.items()},
            current_shorts={t: sp.qty for t, sp in self.ledger.shorts.items()},
            last_close={t: bundle.ticks[t].close for t in bundle.ticks if t in bundle.ticks},
            target_weights=target_w,
            equity=equity,
            settings=self.settings.strategy,
            max_orders_remaining=max_orders_remaining,
            short_allowed=short_allowed,
        )
        # P4-01 trace: candidates were seen live but no SHORT orders reached
        # the server (2026-10-30) — make every short target's fate visible.
        for t, w in target_w.items():
            if w >= 0:
                continue
            emitted = next((d for d in deltas.trades
                            if d.ticker == t and d.action in ("SHORT", "COVER", "SELL")),
                           None)
            skipped = next((r for (tk, r) in deltas.skipped if tk == t), None)
            _log.info("short_delta_trace", ticker=t, target_w=round(w, 4),
                      equity=round(equity, 0),
                      price=round(bundle.ticks[t].close, 2) if t in bundle.ticks else None,
                      emitted_action=emitted.action if emitted else None,
                      emitted_qty=emitted.qty if emitted else 0,
                      skipped_reason=skipped, short_allowed=short_allowed,
                      orders_remaining=max_orders_remaining,
                      component="short_engine")

        # P3-23: take-profit — a long that gained >= take_profit_pct from its
        # FIFO entry is SOLD here, instead of riding back down until the
        # score decays. Winners were being held 30+ days and giving profits
        # back while losers exited fast. The sold ticker is barred from
        # re-entry for the rest of the day (no same-day round-trip churn).
        tp_pct = float(getattr(self.settings.strategy, "take_profit_pct", 0.0) or 0.0)
        tp_exits: set[str] = set()
        if tp_pct > 0 and not self.settings.dry_run:
            for t, lots in list(self.ledger.lots.items()):
                if not lots or t not in bundle.ticks:
                    continue
                entry = float(lots[0].price)
                px = float(bundle.ticks[t].close)
                if entry <= 0 or (px - entry) / entry < tp_pct:
                    continue
                qty = sum(int(l.qty) for l in lots)
                server_h = getattr(self, "_server_holdings", None)
                if server_h is not None:
                    qty = min(qty, max(0, int(server_h.get(t, 0))))
                if qty <= 0:
                    continue
                remaining_tp = await self.lifecycle.gate.orders_remaining(max_orders)
                # Keep at least one slot for the regular deltas below.
                if remaining_tp <= 1:
                    break
                _log.info("take_profit_exit", ticker=t, qty=qty,
                          entry=round(entry, 2), px=round(px, 2),
                          gain_pct=round((px - entry) / entry * 100, 1),
                          component="main")
                await self.order_manager.submit(
                    OrderRequest(ticker=t, action="SELL", quantity=qty,
                                 date=bundle.simulated_date,
                                 reason=f"take_profit +{(px-entry)/entry:.1%}"),
                    scores={"ensemble": 0.0},
                    reasons=["take_profit",
                             f"regime={regime_label}"],
                )
                await self.lifecycle.gate.record_order()
                if server_h is not None:
                    server_h[t] = int(server_h.get(t, 0)) - qty
                tp_exits.add(t)
                self._last_sell_date[t] = bundle.simulated_date  # P4-02

        # P3-25: rung-ladder partial take-profit — at each configured gain
        # level (from FIFO entry), sell `frac` of the CURRENT holding; the
        # remainder after the last rung rides the native score-decay exit.
        # One rung per day per ticker; rungs fire once per position
        # lifetime (stage resets when flat).
        ptp_cfg = getattr(self.settings.strategy, "partial_take_profit", None)
        if ptp_cfg is not None and bool(getattr(ptp_cfg, "enabled", False)) \
                and not self.settings.dry_run:
            rungs = getattr(ptp_cfg, "rungs", None) or [
                {"pct": float(getattr(ptp_cfg, "pct", 0.08) or 0.08),
                 "frac": float(getattr(ptp_cfg, "fraction", 0.25) or 0.25)}]
            for t in list(self._ptp_stage):
                if not self.ledger.lots.get(t):
                    del self._ptp_stage[t]  # flat — reset for next position
            for t, lots in list(self.ledger.lots.items()):
                if not lots or t not in bundle.ticks:
                    continue
                stage = int(self._ptp_stage.get(t, 0))
                if stage >= len(rungs):
                    continue
                rung = rungs[stage]
                entry = float(lots[0].price)
                px = float(bundle.ticks[t].close)
                if entry <= 0 or (px - entry) / entry < float(rung["pct"]):
                    continue
                frac = float(rung["frac"])
                held = sum(int(l.qty) for l in lots)
                qty = int(held * frac)
                server_h = getattr(self, "_server_holdings", None)
                if server_h is not None:
                    qty = min(qty, max(0, int(server_h.get(t, 0))))
                if qty <= 0:
                    continue
                remaining_ptp = await self.lifecycle.gate.orders_remaining(max_orders)
                if remaining_ptp <= 1:
                    break
                _log.info("partial_take_profit", ticker=t, qty=qty,
                          rung=stage + 1, n_rungs=len(rungs),
                          frac=frac, entry=round(entry, 2), px=round(px, 2),
                          gain_pct=round((px - entry) / entry * 100, 1),
                          component="main")
                await self.order_manager.submit(
                    OrderRequest(ticker=t, action="SELL", quantity=qty,
                                 date=bundle.simulated_date,
                                 reason=f"partial_tp rung{stage+1} {frac:.0%}@{px/entry-1:.1%}"),
                    scores={"ensemble": 0.0},
                    reasons=["partial_take_profit", f"regime={regime_label}"],
                )
                await self.lifecycle.gate.record_order()
                if server_h is not None:
                    server_h[t] = int(server_h.get(t, 0)) - qty
                self._ptp_stage[t] = stage + 1
                # P4-02: rung sells also start the re-entry cooldown.
                self._last_sell_date[t] = bundle.simulated_date

        # P4-03: short profit rungs — mirror of the long ladder. At each
        # FAVORABLE move (price DOWN `pct` from short entry), COVER `frac`
        # of the current short; the rest rides the bearish-decay exit.
        # Fixes the measured short asymmetry (avg win +1,656 vs avg loss
        # -3,218 — decay clipped winners, the adverse stop let losers run).
        if self.short_engine is not None and self.short_engine.enabled \
                and not self.settings.dry_run:
            rungs = getattr(self.short_engine.settings, "profit_rungs", None) or []
            for t in list(self._sptp_stage):
                if t not in self.ledger.shorts:
                    del self._sptp_stage[t]  # flat — reset for next short
            for t, sp in list(self.ledger.shorts.items()):
                if not rungs or sp.qty <= 0 or t not in bundle.ticks:
                    continue
                stage = int(self._sptp_stage.get(t, 0))
                if stage >= len(rungs):
                    continue
                entry = float(getattr(sp, "entry_price", 0.0) or 0.0)
                px = float(bundle.ticks[t].close)
                if entry <= 0 or (entry - px) / entry < float(rungs[stage]["pct"]):
                    continue
                qty = int(int(sp.qty) * float(rungs[stage]["frac"]))
                server_s = getattr(self, "_server_shorts", None)
                if server_s is not None:
                    qty = min(qty, max(0, int(server_s.get(t, 0))))
                if qty <= 0:
                    continue
                remaining_sp = await self.lifecycle.gate.orders_remaining(max_orders)
                if remaining_sp <= 1:
                    break
                _log.info("short_profit_rung", ticker=t, qty=qty,
                          rung=stage + 1, n_rungs=len(rungs),
                          entry=round(entry, 2), px=round(px, 2),
                          gain_pct=round((entry - px) / entry * 100, 1),
                          component="short_engine")
                await self.order_manager.submit(
                    OrderRequest(ticker=t, action="COVER", quantity=qty,
                                 date=bundle.simulated_date,
                                 reason=f"short_tp rung{stage+1} {rungs[stage]['frac']:.0%}@{(entry-px)/entry:.1%}"),
                    scores={"ensemble": 0.0},
                    reasons=["short_profit_rung", f"regime={regime_label}"],
                )
                await self.lifecycle.gate.record_order()
                if server_s is not None:
                    server_s[t] = int(server_s.get(t, 0)) - qty
                self._sptp_stage[t] = stage + 1

        # P3-24: stepped trailing stop-loss (ratchets up, never lowers):
        #   peak gain < +3%  -> stop at entry - 10%
        #   peak gain >= +3% -> stop at breakeven (buy price)
        #   peak gain >= +5% -> stop at entry+3%, then +2% steps behind the
        #                       peak (+7% peak -> +5%, +9% -> +7%, ...)
        tsl_cfg = getattr(self.settings.strategy, "trailing_sl", None)
        if tsl_cfg is not None and bool(getattr(tsl_cfg, "enabled", False)) \
                and not self.settings.dry_run:
            import math as _math
            init_pct = float(getattr(tsl_cfg, "initial_pct", 0.10) or 0.10)
            be_trig = float(getattr(tsl_cfg, "breakeven_trigger_pct", 0.03) or 0.03)
            lock_start = float(getattr(tsl_cfg, "lock_start_pct", 0.05) or 0.05)
            first_lock = float(getattr(tsl_cfg, "first_lock_pct", 0.03) or 0.03)
            step_pct = float(getattr(tsl_cfg, "step_pct", 0.02) or 0.02)

            def _stepped_stop(entry: float, peak: float) -> float:
                gain = (peak - entry) / entry
                if gain >= lock_start:
                    locked = first_lock + step_pct * _math.floor(
                        (gain - lock_start) / step_pct)
                    return entry * (1.0 + locked)
                if gain >= be_trig:
                    return entry
                return entry * (1.0 - init_pct)
            for t in list(self._trail_peak):
                if not self.ledger.lots.get(t):
                    del self._trail_peak[t]  # flat — reset the trail
            for t, lots in list(self.ledger.lots.items()):
                if not lots or t not in bundle.ticks or t in tp_exits:
                    continue
                entry = float(lots[0].price)
                px = float(bundle.ticks[t].close)
                if entry <= 0 or px <= 0:
                    continue
                peak = max(float(self._trail_peak.get(t, entry)), px)
                self._trail_peak[t] = peak
                stop = _stepped_stop(entry, peak)
                if px > stop:
                    continue
                qty = sum(int(l.qty) for l in lots)
                server_h = getattr(self, "_server_holdings", None)
                if server_h is not None:
                    qty = min(qty, max(0, int(server_h.get(t, 0))))
                if qty <= 0:
                    continue
                remaining_ts = await self.lifecycle.gate.orders_remaining(max_orders)
                if remaining_ts <= 1:
                    break
                _log.info("trailing_sl_exit", ticker=t, qty=qty,
                          entry=round(entry, 2), peak=round(peak, 2),
                          px=round(px, 2), stop=round(stop, 2),
                          pnl_pct=round((px - entry) / entry * 100, 1),
                          component="main")
                await self.order_manager.submit(
                    OrderRequest(ticker=t, action="SELL", quantity=qty,
                                 date=bundle.simulated_date,
                                 reason=f"trailing_sl peak={peak:.2f} stop={stop:.2f}"),
                    scores={"ensemble": 0.0},
                    reasons=["trailing_sl", f"regime={regime_label}"],
                )
                await self.lifecycle.gate.record_order()
                if server_h is not None:
                    server_h[t] = int(server_h.get(t, 0)) - qty
                tp_exits.add(t)  # same no-same-day-reentry treatment
                self._last_sell_date[t] = bundle.simulated_date  # P4-02

        # P4-04: giveback guard — armed stop for the ladder REMAINDER. Once
        # the close reaches entry*(1+arm_pct) (rung-1 territory), a stop arms
        # at entry*(1+stop_pct); if the position round-trips back to it, the
        # whole remainder is SOLD. First overlay ever to beat recorded exits
        # in the 744-trip tournament: +71.2K on only 10 triggers (arm 8% /
        # stop -2%). Global trailing stops LOSE (-137K/-214K) because they
        # strangle pre-rung positions — this one is inert until +8%.
        gg_cfg = getattr(self.settings.strategy, "giveback_guard", None)
        if gg_cfg is not None and bool(getattr(gg_cfg, "enabled", False)) \
                and not self.settings.dry_run:
            gg_arm = float(getattr(gg_cfg, "arm_pct", 0.08) or 0.08)
            gg_stop_pct = float(getattr(gg_cfg, "stop_pct", -0.02))
            for t in list(self._gg_armed):
                if not self.ledger.lots.get(t):
                    self._gg_armed.discard(t)  # flat — disarm
            for t, lots in list(self.ledger.lots.items()):
                if not lots or t not in bundle.ticks or t in tp_exits:
                    continue
                entry = float(lots[0].price)
                px = float(bundle.ticks[t].close)
                if entry <= 0 or px <= 0:
                    continue
                if px >= entry * (1.0 + gg_arm):
                    self._gg_armed.add(t)
                if t not in self._gg_armed or px > entry * (1.0 + gg_stop_pct):
                    continue
                qty = sum(int(l.qty) for l in lots)
                server_h = getattr(self, "_server_holdings", None)
                if server_h is not None:
                    qty = min(qty, max(0, int(server_h.get(t, 0))))
                if qty <= 0:
                    continue
                remaining_gg = await self.lifecycle.gate.orders_remaining(max_orders)
                if remaining_gg <= 1:
                    break
                _log.info("giveback_guard_exit", ticker=t, qty=qty,
                          entry=round(entry, 2), px=round(px, 2),
                          arm_pct=gg_arm, stop_pct=gg_stop_pct,
                          pnl_pct=round((px - entry) / entry * 100, 1),
                          component="main")
                await self.order_manager.submit(
                    OrderRequest(ticker=t, action="SELL", quantity=qty,
                                 date=bundle.simulated_date,
                                 reason=f"giveback_guard armed@{gg_arm:.0%} stop@{gg_stop_pct:.0%}"),
                    scores={"ensemble": 0.0},
                    reasons=["giveback_guard", f"regime={regime_label}"],
                )
                await self.lifecycle.gate.record_order()
                if server_h is not None:
                    server_h[t] = int(server_h.get(t, 0)) - qty
                tp_exits.add(t)
                self._last_sell_date[t] = bundle.simulated_date  # P4-02
                self._gg_armed.discard(t)

        # 5) Execute trades
        min_hold = int(getattr(self.settings.strategy, "min_hold_days", 0) or 0)
        from .signals.overlays import (passes_entry_timing, strong_disagreement,
                                       reentry_blocked)
        dg_cfg = getattr(self.settings.strategy, "disagreement_gate", None)
        dg_enabled = bool(dg_cfg and getattr(dg_cfg, "enabled", False))
        dg_thr = float(getattr(dg_cfg, "threshold", 0.1) or 0.1) if dg_cfg else 0.1
        # P4-06: underwater resize guard — defer lossy constructor SELLs
        # while the raw blend is still bullish (vol-deflation dip, not
        # signal decay). Counterfactual on 1,209 lossy vol_scaled exits:
        # +17.5L held to recovery (87% recover within 40 trading days).
        rg_cfg = getattr(self.settings.strategy, "resize_guard", None)
        rg_enabled = bool(rg_cfg and getattr(rg_cfg, "enabled", False)) \
            and not self.settings.dry_run
        rg_min_loss = float(getattr(rg_cfg, "min_loss_pct", 0.02) or 0.02) if rg_cfg else 0.02
        rg_max_days = int(getattr(rg_cfg, "max_defer_days", 40) or 40) if rg_cfg else 40
        for d in deltas.trades:
            qty = d.qty
            if rg_enabled and d.action == "SELL":
                lots_rg = self.ledger.lots.get(d.ticker)
                if lots_rg and d.ticker in bundle.ticks:
                    entry_rg = float(lots_rg[0].price)
                    px_rg = float(bundle.ticks[d.ticker].close)
                    ev_rg = next((e for e in evals if e.ticker == d.ticker), None)
                    raw_rg = float(ev_rg.raw_score) if ev_rg is not None else 0.0
                    if entry_rg > 0 and px_rg < entry_rg * (1.0 - rg_min_loss) \
                            and raw_rg > 0:
                        first_d = self._resize_defer.get(d.ticker)
                        # stale entry from a previous position -> reset
                        if first_d is None or first_d < lots_rg[0].buy_date:
                            first_d = bundle.simulated_date
                            self._resize_defer[d.ticker] = first_d
                        defer_days = (bundle.simulated_date - first_d).days
                        if defer_days < rg_max_days:
                            _log.info("resize_guard_defer", ticker=d.ticker,
                                      qty=qty, entry=round(entry_rg, 2),
                                      px=round(px_rg, 2),
                                      loss_pct=round((px_rg - entry_rg) / entry_rg * 100, 1),
                                      raw=round(raw_rg, 4), defer_day=defer_days,
                                      component="main")
                            continue
                        # defer expired -> allow the score-decay exit
                    else:
                        # recovered / not underwater / raw flipped -> clear
                        self._resize_defer.pop(d.ticker, None)
            # Minimum holding period: skip SELL/COVER for lots younger than
            # `min_hold_days` sim days UNLESS the exit is a stop-loss or the
            # settlement guard (those must always be allowed through).
            if min_hold > 0 and d.action in ("SELL", "COVER"):
                from .signals.overlays import min_hold_allows_exit
                age_days: Optional[int] = None
                if d.action == "SELL":
                    lots = self.ledger.lots.get(d.ticker)
                    if lots:
                        # FIFO: the oldest lot is the one being sold.
                        age_days = (bundle.simulated_date
                                    - lots[0].buy_date).days
                else:
                    sp = self.ledger.shorts.get(d.ticker)
                    if sp is not None:
                        age_days = (bundle.simulated_date
                                    - sp.open_date).days
                if not min_hold_allows_exit(age_days, min_hold, d.reason):
                    _log.info("hold_period_blocked", ticker=d.ticker,
                              action=d.action, age_days=age_days,
                              min_hold_days=min_hold, reason=d.reason,
                              component="main")
                    continue
            if d.action == "BUY":
                # P4-02: re-entry cooldown — a name SOLD within the last
                # `reentry_cooldown_days` sim days cannot be re-bought unless
                # its fresh score clears `reentry_score_override`. Kills the
                # sell-today/rebuy-tomorrow whipsaw that cost -3.0L lifetime
                # (169 trades held <=2d) while exits stay untouched.
                cd_days = int(getattr(self.settings.strategy,
                                      "reentry_cooldown_days", 0) or 0)
                if cd_days > 0:
                    last_sell = self._last_sell_date.get(d.ticker)
                    since = ((bundle.simulated_date - last_sell).days
                             if last_sell is not None else None)
                    ev_cd = next((e for e in evals if e.ticker == d.ticker), None)
                    cd_score = ev_cd.score if ev_cd is not None else 0.0
                    if reentry_blocked(
                            since, cd_days, cd_score,
                            float(getattr(self.settings.strategy,
                                          "reentry_score_override", 0.90) or 0.90)):
                        _log.info("reentry_cooldown_blocked", ticker=d.ticker,
                                  days_since_sell=since, score=round(cd_score, 4),
                                  component="main")
                        continue
                # P3-23: never re-enter a name we took profit on today.
                if d.ticker in tp_exits:
                    _log.info("take_profit_reentry_blocked", ticker=d.ticker,
                              component="main")
                    continue
                # P3-22b: bandit hard-veto — chronically losing tickers
                # (>=veto_pulls realised trades, edge z <= veto_z) get no
                # new BUYs. Self-heals via day_decay (not a static blacklist).
                if self.ticker_bandit is not None and self.ticker_bandit.is_vetoed(
                        d.ticker, today=bundle.simulated_date):
                    _log.info("ticker_bandit_veto", ticker=d.ticker,
                              stats=self.ticker_bandit.stats()["tickers"].get(d.ticker),
                              component="main")
                    continue
                ev_gate = next((e for e in evals if e.ticker == d.ticker), None)
                # Ensemble disagreement gate: veto BUYs where rank and mom
                # strongly disagree in sign. Journaled as HOLD with reason
                # `signal_disagreement` via the hold-record loop below.
                if dg_enabled and ev_gate is not None and strong_disagreement(
                        ev_gate.rank_score, ev_gate.momentum_score, dg_thr):
                    ev_gate.reasons.append("signal_disagreement")
                    _log.info("signal_disagreement", ticker=d.ticker,
                              rank=round(ev_gate.rank_score, 4),
                              mom=round(ev_gate.momentum_score, 4),
                              threshold=dg_thr, component="main")
                    continue
                # Universe-relative entry timing: must be in today's top-N
                # AND >= its own rolling median score.
                if et_enabled:
                    hist = self._score_hist.get(d.ticker, [])
                    lookback = int(getattr(et_cfg, "lookback_days", 20) or 20)
                    min_hist = int(getattr(et_cfg, "min_history", 3) or 3)
                    window = hist[-lookback:] if lookback > 0 else hist
                    score_now = ev_gate.score if ev_gate is not None else 0.0
                    if (d.ticker not in top_n_set
                            or not passes_entry_timing(score_now, window, min_hist)):
                        _log.info("entry_timing_blocked", ticker=d.ticker,
                                  score=round(score_now, 4),
                                  in_top_n=d.ticker in top_n_set,
                                  n_history=len(window), component="main")
                        continue
                # Fresh-entry only: only BUY tickers that newly entered the
                # top-3 today (weren't in yesterday's top-3 by score).
                if fresh_only and prev_top3 is not None:
                    if d.ticker not in top3_today or d.ticker in prev_top3:
                        _log.info("fresh_entry_blocked", ticker=d.ticker,
                                  in_top3=d.ticker in top3_today,
                                  was_in_top3=d.ticker in prev_top3,
                                  component="main")
                        continue
            # P3-20: clamp SELL/COVER to the SERVER-verified position so the
            # server never sees an order it would reject with
            # INSUFFICIENT_MARGIN / INSUFFICIENT_HOLDINGS. The local ledger
            # can drift (unparseable order responses, async fills); the
            # reconcile cache is the source of truth.
            server_h = getattr(self, "_server_holdings", None)
            server_s = getattr(self, "_server_shorts", None)
            if d.action == "SELL" and server_h is not None:
                sellable = max(0, int(server_h.get(d.ticker, 0)))
                if qty > sellable:
                    _log.info("sell_clamped_to_server", ticker=d.ticker,
                              wanted=qty, sellable=sellable, component="main")
                    qty = sellable
                if qty <= 0:
                    _log.info("sell_skipped_no_server_holding", ticker=d.ticker,
                              server_qty=int(server_h.get(d.ticker, 0)), component="main")
                    continue
            elif d.action == "COVER" and server_s is not None:
                coverable = max(0, int(server_s.get(d.ticker, 0)))
                if qty > coverable:
                    _log.info("cover_clamped_to_server", ticker=d.ticker,
                              wanted=qty, coverable=coverable, component="main")
                    qty = coverable
                if qty <= 0:
                    continue
            elif d.action == "BUY":
                # P3-21: clamp BUYs to server-verified cash with a 2%
                # price-drift buffer + brokerage, so the server never sees
                # an order it would reject with INSUFFICIENT_CAPITAL.
                server_cash = getattr(self, "_server_cash", None)
                if server_cash is not None and d.price > 0:
                    max_frac = float(getattr(self.settings.risk, "max_cash_per_trade", 0.90) or 0.90)
                    affordable = (server_cash * max_frac - self.settings.tax.brokerage_per_order) * 0.98
                    max_qty = int(affordable / d.price)
                    if qty > max_qty:
                        _log.info("buy_clamped_to_server_cash", ticker=d.ticker,
                                  wanted=qty, max_qty=max_qty,
                                  server_cash=round(server_cash, 0), component="main")
                        qty = max_qty
                    if qty <= 0 or qty * d.price < self.settings.strategy.min_trade_notional:
                        _log.info("buy_skipped_unaffordable", ticker=d.ticker,
                                  server_cash=round(server_cash, 0), component="main")
                        continue
            req = OrderRequest(
                ticker=d.ticker, action=d.action, quantity=qty,
                date=bundle.simulated_date, reason=d.reason,
            )
            # Find the matching eval for journal context
            ev = next((e for e in evals if e.ticker == d.ticker), None)
            scores = {
                "rank": ev.rank_score if ev else 0.0,
                "news": ev.news_score if ev else 0.0,
                "mom": ev.momentum_score if ev else 0.0,
                "ensemble": ev.score if ev else 0.0,
            }
            reasons = (ev.reasons if ev else []) + [d.reason, f"regime={regime_label}"]
            if self.settings.dry_run:
                _log.info("dry_run_trade", ticker=d.ticker, action=d.action,
                          qty=d.qty, price=d.price, target_w=d.target_weight,
                          component="main")
                continue
            resp = await self.order_manager.submit(req, scores=scores, reasons=reasons)
            # P4-02: remember SELLs so the re-entry cooldown can block a
            # whipsaw re-buy of the same name within the cooldown window.
            if d.action == "SELL" and getattr(resp, "status", "") == "FILLED":
                self._last_sell_date[d.ticker] = bundle.simulated_date
            # P3-21: keep the server-truth caches fresh between reconciles so
            # the NEXT order in this batch is clamped against what the server
            # actually holds right now (fills change cash + positions).
            if getattr(resp, "status", "") == "FILLED":
                self._apply_fill_to_server_cache(resp)
            await self.lifecycle.gate.record_order()

        # 6) Hold records (one per ticker that had a tick but no trade)
        traded = {d.ticker for d in deltas.trades}
        for ev in evals:
            if ev.ticker in traded:
                continue
            price = bundle.ticks[ev.ticker].close if ev.ticker in bundle.ticks else 0.0
            self.journal.record({
                "ts_wall": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "sim_date": bundle.simulated_date.isoformat(),
                "ticker": ev.ticker,
                "decision": "HOLD", "qty": 0, "price": price,
                "source": "core",
                "scores": {"rank": ev.rank_score, "news": ev.news_score,
                            "mom": ev.momentum_score, "ensemble": ev.score},
                "regime": regime_label, "target_w": None, "current_w": None,
                "reasons": ev.reasons + [f"regime={regime_label}"], "shap_top3": [],
                "latency_ms": {"assemble": bundle.assembly_latency_ms},
                "order_id": "", "status": "HOLD", "reject_code": None,
            })

        # 7) Update rolling score history (entry-timing gate) + yesterday's
        # top-3 set (fresh-entry gate) AFTER all decisions for the day.
        lookback_cap = 20
        if et_cfg is not None:
            lookback_cap = max(1, int(getattr(et_cfg, "lookback_days", 20) or 20))
        for e in evals:
            h = self._score_hist.setdefault(e.ticker, [])
            h.append(e.score)
            if len(h) > lookback_cap:
                del h[: len(h) - lookback_cap]
        self._prev_top3 = top3_today

        self.lifecycle.stats.days_traded += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
async def run(args: argparse.Namespace) -> int:
    """Load settings, boot the bot, install SIGINT/SIGTERM handlers, and
    block until a signal or the bot stops itself."""
    settings = load_settings(env=args.env, dry_run=args.dry_run)
    fail_fast(settings)
    bot = TradingBot(settings)
    # Install signal handlers
    loop = asyncio.get_running_loop()
    stop_evt = asyncio.Event()
    def _on_signal(*_):
        stop_evt.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except NotImplementedError:
            pass

    await bot.start()

    # Wait for either stop signal or bot self-stop
    stop_task = asyncio.create_task(stop_evt.wait(), name="stop-signal")
    bot_task = asyncio.create_task(bot.wait_until_stopped(), name="bot-stop")
    done, pending = await asyncio.wait(
        {stop_task, bot_task}, return_when=asyncio.FIRST_COMPLETED,
    )
    for t in pending:
        t.cancel()
    await bot.stop()
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    """CLI: --env development|sandbox|live (default $BOT_ENV), --dry-run."""
    ap = argparse.ArgumentParser(prog="bot.main")
    ap.add_argument("--env", choices=["development", "sandbox", "live"],
                    default=os.environ.get("BOT_ENV", "sandbox"))
    ap.add_argument("--dry-run", action="store_true",
                    help="connect + score, but don't submit orders")
    return ap.parse_args(argv)


def main() -> int:
    """Synchronous entry point — wraps the async runner in asyncio.run."""
    args = parse_args(sys.argv[1:])
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())