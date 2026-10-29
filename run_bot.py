"""run_bot.py — single-file launcher for the Manthan trading bot.

Run it and pick a data source:

    python run_bot.py

    [1] KAFKA       — live stream. Connects to Kafka (market.ticks /
                      market.news), ACKs every pre-flight heartbeat
                      (POST /api/v1/bot/heartbeat {"seq": N}), and trades
                      ONLY on data received from Kafka.
    [2] SIMULATION  — no Kafka. Polls the REST API endpoints
                      (GET /api/v1/stocks/{ticker}/history) for simulated
                      market data and trades on that instead.

Both modes share everything else: universe discovery, M1/M2/M3 models,
ensemble (0.45 rank / 0.35 news / 0.20 momentum), M4 vol sizing, M5 regime,
M6 event tags, risk checks, order manager, ledger, journal.

Optional non-interactive usage:
    python run_bot.py --mode kafka
    python run_bot.py --mode sim
    python run_bot.py --check        # validate config/imports, then exit

Environment (same as bot.main):
    BOT_ACCESS_TOKEN   — pre-issued platform token (required; falls back to
                         %TEMP%\\opencode\\jwt.txt if the env var is missing)
    TEAM_USERNAME      — e.g. mn2632
    SERVER_IP          — e.g. 192.168.65.157
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import date as _date, datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment bootstrap — must happen before importing bot.* so that
# FinBERT loads fully offline from the local snapshot (model2/finbert_cache).
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("BOT_ENV", "sandbox")

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _ensure_token() -> None:
    """BOT_ACCESS_TOKEN must be set; fall back to the saved jwt.txt."""
    if os.environ.get("BOT_ACCESS_TOKEN"):
        return
    jwt_file = Path(os.environ.get("TEMP", "/tmp")) / "opencode" / "jwt.txt"
    if jwt_file.exists():
        os.environ["BOT_ACCESS_TOKEN"] = jwt_file.read_text().strip()
        print(f"[run_bot] BOT_ACCESS_TOKEN loaded from {jwt_file}")
    else:
        print("[run_bot] ERROR: BOT_ACCESS_TOKEN is not set and no jwt.txt fallback found.")
        print("          set BOT_ACCESS_TOKEN=<token> and re-run.")
        sys.exit(2)


# ---------------------------------------------------------------------------
# Mode 1 — KAFKA (the standard pipeline, identical to `python -m bot.main`)
# ---------------------------------------------------------------------------
async def _run_kafka_mode(env: str, dry_run: bool) -> int:
    """Full live pipeline: Kafka consumer + pre-flight heartbeat ACKs +
    trading strictly on Kafka-received ticks/news."""
    from bot.main import TradingBot
    from bot.config import load_settings, fail_fast
    from bot.lifecycle import BotState

    settings = load_settings(env=env, dry_run=dry_run)
    fail_fast(settings)
    bot = TradingBot(settings)
    print("[run_bot] KAFKA mode — connecting to "
          f"{settings.broker} (group {settings.group_id}) ...")
    print("[run_bot] Pre-flight ticks will be ACKed automatically via "
          "POST /api/v1/bot/heartbeat.")
    await bot.start()
    try:
        await bot.wait_until_stopped()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        if bot.lifecycle.state != BotState.SHUTDOWN:
            await bot.stop()
    return 0


# ---------------------------------------------------------------------------
# Mode 2 — SIMULATION (REST-polled market data, no Kafka at all)
# ---------------------------------------------------------------------------
# Poll pacing: 10 tickers x 1 request/s => ~60 req/min + stage watcher ~12/min
# = ~72/min, safely under the platform's 100/min rate limit.
_SIM_REQUEST_GAP_S = 1.0
_SIM_IDLE_S = 5.0


async def _rest_poller(bot) -> None:
    """Feed the bot's event queue from REST history endpoints.

    For each ticker in the universe, fetch the most recent candles from
    GET /api/v1/stocks/{ticker}/history?days=5 and emit a PriceTickEvent for
    every candle newer than the last one emitted. The DayAssembler downstream
    groups these into DayBundles exactly as it does for Kafka events, so
    scoring + order placement are identical in both modes.

    Note: the platform exposes no REST news endpoint, so M2 receives an empty
    news set in this mode and the ensemble renormalizes over rank+momentum
    (graceful degradation per docs/04 §4).
    """
    import structlog
    from bot.contracts import PriceTickData, PriceTickEvent
    _log = structlog.get_logger("rest_poller")

    last_emitted: dict[str, _date] = {}
    _log.info("rest_poller_started", gap_s=_SIM_REQUEST_GAP_S, component="sim")

    while not bot._stop.is_set():
        tickers = sorted(bot._universe)
        if not tickers:
            # Universe not discovered yet (server warming up) — retry.
            try:
                await bot._discover_universe(retries=1, delay_s=0.0)
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(_SIM_IDLE_S)
            continue

        for ticker in tickers:
            if bot._stop.is_set():
                return
            try:
                raw = await bot.api.get(f"/api/v1/stocks/{ticker}/history?days=5")
                prices = raw.get("prices") or raw.get("history") or raw.get("data") or []
                candles = sorted(
                    (c for c in prices if isinstance(c, dict) and c.get("date")),
                    key=lambda c: c["date"],
                )
                emitted = last_emitted.get(ticker)
                for c in candles:
                    cdate = _date.fromisoformat(str(c["date"])[:10])
                    if emitted is not None and cdate <= emitted:
                        continue
                    ev = PriceTickEvent(
                        timestamp=datetime.now(timezone.utc),
                        simulated_date=cdate,
                        data=PriceTickData(
                            ticker=ticker,
                            open=float(c.get("open", c["close"])),
                            high=float(c.get("high", c["close"])),
                            low=float(c.get("low", c["close"])),
                            close=float(c["close"]),
                            volume=int(c.get("volume", 0)),
                        ),
                    )
                    await bot.queue.put(ev)
                    last_emitted[ticker] = cdate
                    _log.info("rest_tick_emitted", ticker=ticker,
                              sim_date=cdate.isoformat(), component="sim")
            except Exception as e:  # noqa: BLE001
                _log.warning("rest_poll_failed", ticker=ticker,
                             err=repr(e), component="sim")
            await asyncio.sleep(_SIM_REQUEST_GAP_S)


async def _run_sim_mode(env: str, dry_run: bool) -> int:
    """Same bot, but market data comes from REST endpoints instead of Kafka.

    Mirrors TradingBot.start() minus `kafka.start()`; everything downstream
    (DayAssembler -> _on_day -> ensemble -> orders) is unchanged.
    """
    from bot.main import TradingBot
    from bot.config import load_settings, fail_fast
    from bot.lifecycle import BotState

    settings = load_settings(env=env, dry_run=dry_run)
    fail_fast(settings)
    bot = TradingBot(settings)
    print("[run_bot] SIMULATION mode — NO Kafka. Market data will be polled "
          "from REST endpoints (/api/v1/stocks/{ticker}/history).")
    print("[run_bot] NOTE: pre-flight heartbeats only exist on Kafka; this "
          "mode does not participate in pre-flight readiness.")
    print("[run_bot] NOTE: no REST news endpoint exists — M2 news scores are "
          "neutral here; ensemble uses rank+momentum.")

    await bot.api.start()
    await bot.lifecycle.transition(BotState.DISCOVER, reason="bot.start(sim)")
    await bot._discover_universe()
    await bot._backfill_ring_buffer()
    await bot._reconcile()
    # --- the only difference vs Kafka mode: no `await bot.kafka.start()` ---
    await bot.assembler.start()
    await bot.heartbeat.start()
    await bot.stage_watcher.start()

    bot._tasks = [
        asyncio.create_task(bot._preflight_router(), name="preflight-router"),
        asyncio.create_task(bot._health_loop(), name="health"),
        asyncio.create_task(bot._settle_loop(), name="settle"),
        asyncio.create_task(_rest_poller(bot), name="rest-poller"),
    ]

    try:
        await bot.wait_until_stopped()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        if bot.lifecycle.state != BotState.SHUTDOWN:
            await bot.stop()  # kafka.stop() is a no-op when never started
    return 0


# ---------------------------------------------------------------------------
# Prompt + CLI
# ---------------------------------------------------------------------------
def _prompt_mode() -> str:
    """Interactive picker -> 'kafka' or 'sim' (loops until valid input)."""
    print("=" * 64)
    print("  Manthan Trading Bot — choose a market-data source")
    print("=" * 64)
    print("  [1] KAFKA       live stream (market.ticks / market.news),")
    print("                  pre-flight heartbeat ACKs, trade on Kafka data only")
    print("  [2] SIMULATION  no Kafka — poll simulated data from the REST API")
    print("                  endpoints (/api/v1/stocks/{ticker}/history)")
    print("=" * 64)
    while True:
        choice = input("Enter 1 (Kafka) or 2 (Simulation): ").strip().lower()
        if choice in ("1", "kafka", "k"):
            return "kafka"
        if choice in ("2", "sim", "simulation", "s"):
            return "sim"
        print("  -> please enter 1 or 2")


def _check_only(env: str) -> int:
    """Validate token/settings/imports without starting anything."""
    from bot.config import load_settings, fail_fast
    settings = load_settings(env=env, dry_run=False)
    fail_fast(settings)
    import bot.main  # noqa: F401
    import bot.contracts  # noqa: F401
    print(f"[run_bot] check OK — env={settings.env} broker={settings.broker} "
          f"group={settings.group_id} team={settings.team_username}")
    return 0


def main() -> int:
    """Entry point: ensure the token, then dispatch to --check / kafka / sim."""
    ap = argparse.ArgumentParser(prog="run_bot")
    ap.add_argument("--mode", choices=["kafka", "sim"], default=None,
                    help="skip the prompt")
    ap.add_argument("--env", default=os.environ.get("BOT_ENV", "sandbox"))
    ap.add_argument("--dry-run", action="store_true",
                    help="connect + score, but don't submit orders")
    ap.add_argument("--check", action="store_true",
                    help="validate config/imports and exit")
    args = ap.parse_args()

    _ensure_token()
    if args.check:
        return _check_only(args.env)

    mode = args.mode or _prompt_mode()
    if mode == "kafka":
        return asyncio.run(_run_kafka_mode(args.env, args.dry_run))
    return asyncio.run(_run_sim_mode(args.env, args.dry_run))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[run_bot] stopped by user")
        sys.exit(0)
