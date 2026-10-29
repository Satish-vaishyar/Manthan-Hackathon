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
        self.signals_ensemble = SignalsEnsemble(
            settings.signals,
            rank_fn=rank_fn,
            news_fn=news_fn,
            mom_fn=mom_fn,  # momentum now has its own model3 scorer
            rules=self.rules_fallback,
        )
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
        )
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._universe: set[str] = set()
        self._sector_of: dict[str, str] = {}

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
        try:
            from .models.model2_bridge import Model2NewsScorer, KeywordNewsScorer
            scorer = Model2NewsScorer()
            keyword = KeywordNewsScorer()
            if scorer.ready:
                _log.info("model2_bridge_ready", component="main")
            else:
                _log.info("model2_bridge_not_ready_using_keyword_fallback",
                          component="main")

            def _news_fn(bundle: DayBundle) -> dict[str, float]:
                try:
                    if scorer.ready:
                        feats = scorer.score(bundle)
                        return {t: feats[t]["news_score"] for t in feats}
                except Exception as e:  # noqa: BLE001
                    _log.warning("model2_score_failed", err=repr(e), component="main")
                # Fallback path: keyword scorer
                kw = keyword.score(bundle)
                return {t: kw[t]["news_score"] for t in kw}

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
    async def _discover_universe(self) -> None:
        try:
            raw = await self.api.get("/api/v1/stocks")
            # Platform returns a list of StockInfo-shaped dicts
            items = raw.get("stocks", raw) if isinstance(raw, dict) else raw
            if isinstance(items, list):
                self._universe = {it.get("ticker") for it in items if isinstance(it, dict) and it.get("ticker")}
                self._sector_of = {it.get("ticker"): (it.get("sector") or "")
                                    for it in items if isinstance(it, dict) and it.get("ticker")}
                _log.info("universe_discovered", n=len(self._universe), component="main")
            else:
                _log.warning("universe_discovery_failed", shape=type(items).__name__, component="main")
        except Exception as e:  # noqa: BLE001
            _log.warning("universe_discovery_error", err=repr(e), component="main")

    def _expected_tickers(self) -> set[str]:
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
    # Reconciliation
    # ------------------------------------------------------------------
    async def _reconcile(self) -> None:
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
            # Reset positions from server snapshot (drift correction)
            from .portfolio.ledger import Lot, ShortPos
            from collections import defaultdict, deque
            self.ledger.lots = defaultdict(deque)
            self.ledger.shorts = {}
            for pos in p.positions:
                self.ledger.lots[pos.ticker].append(Lot(
                    ticker=pos.ticker, qty=pos.quantity, price=pos.avg_price,
                    buy_date=p.timestamp.date() if hasattr(p.timestamp, "date") else _date.today(),
                    fees=0.0,
                ))
                if pos.current_price:
                    self.ledger.last_close[pos.ticker] = pos.current_price
            for sp in p.shorts:
                self.ledger.shorts[sp.ticker] = ShortPos(
                    ticker=sp.ticker, qty=sp.quantity, entry_price=sp.entry_price,
                    open_date=sp.open_date, margin_locked=sp.margin_locked,
                )
                self.ledger.last_close[sp.ticker] = sp.entry_price
            _log.info("reconciled", cash=p.cash, equity=p.equity,
                      longs=len(p.positions), shorts=len(p.shorts), component="main")
        except Exception as e:  # noqa: BLE001
            _log.warning("reconcile_failed", err=repr(e), component="main")

    # ------------------------------------------------------------------
    # Preflight router — drain preflight_ticks out of the queue
    # ------------------------------------------------------------------
    async def _preflight_router(self) -> None:
        while not self._stop.is_set():
            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return
            if event.event_type == "preflight_tick":
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
        while not self._stop.is_set():
            try:
                equity = self.ledger.equity
                v = self.risk.update_equity(equity)
                if not v.ok:
                    _log.warning("dd_breaker", code=v.code, msg=v.message, component="health")
                    if v.code == "RISK_DD_HARD":
                        await self.lifecycle.enter_safe_mode(reason=v.message or v.code)
                        break
                # Reconcile every 30s during trading
                if self.lifecycle.state == BotState.TRADING:
                    await self._reconcile()
            except Exception as e:  # noqa: BLE001
                _log.warning("health_loop_err", err=repr(e), component="health")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=30.0)
                return
            except asyncio.TimeoutError:
                continue

    # ------------------------------------------------------------------
    # Settlement guard loop
    # ------------------------------------------------------------------
    async def _settle_loop(self) -> None:
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
        # 1) Mark trading day on calendar + gate
        self.calendar.observe_trading_day(bundle.simulated_date)
        max_orders = self.settings.strategy.max_orders_per_day
        short_allowed = self.lifecycle.state == BotState.TRADING  # tighten if needed
        await self.lifecycle.gate.open_for_day(bundle.simulated_date, max_orders, short_allowed)

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
            evals = self.signals_ensemble.evaluate(bundle, regime_label)
        except Exception as e:  # noqa: BLE001
            _log.error("ensemble_failed", err=repr(e), component="strategy")
            return

        # 4) Build target weights + trade deltas
        scores_for_constructor: list[TickerScore] = []
        for e in evals:
            price = bundle.ticks[e.ticker].close if e.ticker in bundle.ticks else 0.0
            scores_for_constructor.append(TickerScore(
                ticker=e.ticker, score=e.score,
                vol=self.signals.volatility.vol(e.ticker),
                sector=self._sector_of.get(e.ticker),
                price=price,
            ))

        equity = self.ledger.equity or 1.0
        # M5: regime + gross_long_target are applied inside build_target_weights.
        target_w = build_target_weights(
            scores_for_constructor, self.settings.strategy, equity,
            regime=regime_label,
            sector_caps={"Tech": self.settings.risk.max_sector_weight,
                         "Finance": self.settings.risk.max_sector_weight},
            sector_of=self._sector_of,
            gross_long_target=self.settings.risk.gross_long_target,
        )
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

        # 5) Execute trades
        for d in deltas.trades:
            req = OrderRequest(
                ticker=d.ticker, action=d.action, quantity=d.qty,
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
            await self.order_manager.submit(req, scores=scores, reasons=reasons)
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

        self.lifecycle.stats.days_traded += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
async def run(args: argparse.Namespace) -> int:
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
    ap = argparse.ArgumentParser(prog="bot.main")
    ap.add_argument("--env", choices=["development", "sandbox", "live"],
                    default=os.environ.get("BOT_ENV", "sandbox"))
    ap.add_argument("--dry-run", action="store_true",
                    help="connect + score, but don't submit orders")
    return ap.parse_args(argv)


def main() -> int:
    args = parse_args(sys.argv[1:])
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())