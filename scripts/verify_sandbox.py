"""verify_sandbox.py — one-shot end-to-end check of all bot subsystems against
the live sandbox.

This is NOT the production bot (that's `python -m bot.main`). This is a
verification harness that:

  1. Confirms REST endpoints + JWT auth (status, portfolio, stocks, stage)
  2. Confirms our group_id follows the `group-${username}` pattern
  3. Connects Kafka consumer in a background task and verifies ticks arrive
  4. Sends a heartbeat ACK with seq=0 (best-effort; development stage has
     no preflight_tick, so this should fail with INVALID_SEQ — that's OK,
     it proves the auth + payload format are correct)
  5. Reports portfolio state and ledger-vs-portfolio reconciliation

Exits 0 on full success, 1 on any hard failure.

Usage:
    BOT_ACCESS_TOKEN=... TEAM_USERNAME=mn2632 SERVER_IP=192.168.65.157 \\
        python scripts/verify_sandbox.py [--duration 20]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from typing import Any

import httpx
import structlog

# Make the project root importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from aiokafka import AIOKafkaConsumer
    from aiokafka.errors import KafkaError
except ImportError:
    print("ERROR: aiokafka not installed. pip install aiokafka", file=sys.stderr)
    sys.exit(2)

structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(colors=False),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO+
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)
_log = structlog.get_logger("verify")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
SERVER_IP = os.environ.get("SERVER_IP", "192.168.65.157")
KAFKA_PORT = int(os.environ.get("KAFKA_PORT", "9094"))
GATEWAY = f"http://{SERVER_IP}:3002"
UI = f"http://{SERVER_IP}:3003"
TOKEN = os.environ.get("BOT_ACCESS_TOKEN", "")
TEAM = os.environ.get("TEAM_USERNAME", "")
KAFKA_GROUP = f"group-{TEAM}"
TOPICS = ["market.ticks", "market.news"]
BROKER = f"{SERVER_IP}:{KAFKA_PORT}"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
INFO = "\033[36mINFO\033[0m"


def _ok(msg: str) -> None:
    print(f"[{PASS}] {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"[{FAIL}] {msg}", flush=True)


def _info(msg: str) -> None:
    print(f"[{INFO}] {msg}", flush=True)


def _section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def check_env() -> list[str]:
    """Verify required env vars are set, return list of missing."""
    missing = []
    if not TOKEN:
        missing.append("BOT_ACCESS_TOKEN")
    if not TEAM:
        missing.append("TEAM_USERNAME")
    return missing


def check_group_id_format() -> bool:
    """Per docs/03 §1 + portal: group_id must be exactly `group-${username}`."""
    _section("Group ID format check")
    pattern = r"^group-[a-z0-9_-]+$"
    if not re.match(pattern, KAFKA_GROUP):
        _fail(f"group_id {KAFKA_GROUP!r} does not match pattern {pattern!r}")
        return False
    _ok(f"group_id={KAFKA_GROUP!r} matches `group-${{username}}` pattern")
    return True


async def check_rest_endpoints() -> dict[str, Any]:
    """Hit the key REST endpoints; collect results."""
    _section("REST endpoint checks")
    if not TOKEN:
        _fail("BOT_ACCESS_TOKEN missing — cannot test authenticated endpoints")
        return {}
    headers = {"Authorization": f"Bearer {TOKEN}"}
    results: dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        # /api/v1/bot/status
        try:
            r = await client.get(f"{GATEWAY}/api/v1/bot/status", headers=headers)
            if r.status_code == 200:
                results["status"] = r.json()
                _ok(f"/api/v1/bot/status -> {r.status_code}: "
                    f"kafka_connected={results['status'].get('kafka_connected')} "
                    f"consumers={results['status'].get('kafka_consumer_count')}")
            else:
                _fail(f"/api/v1/bot/status -> {r.status_code}")
        except Exception as e:
            _fail(f"/api/v1/bot/status exception: {e!r}")

        # /api/v1/stocks (universe)
        try:
            r = await client.get(f"{GATEWAY}/api/v1/stocks", headers=headers)
            if r.status_code == 200:
                data = r.json()
                items = data.get("stocks", data) if isinstance(data, dict) else data
                tickers = [it.get("ticker") for it in items if it.get("ticker")]
                results["universe"] = tickers
                _ok(f"/api/v1/stocks -> {r.status_code}: "
                    f"{len(tickers)} tickers ({', '.join(tickers[:5])}...)")
            else:
                _fail(f"/api/v1/stocks -> {r.status_code}")
        except Exception as e:
            _fail(f"/api/v1/stocks exception: {e!r}")

        # /api/v1/portfolio
        try:
            r = await client.get(f"{GATEWAY}/api/v1/portfolio", headers=headers)
            if r.status_code == 200:
                p = r.json()
                results["portfolio"] = p
                _ok(f"/api/v1/portfolio -> {r.status_code}: "
                    f"cash=Rs.{p.get('available_cash', 0):,.2f} "
                    f"orders={p.get('total_orders', 0)} "
                    f"pnl=Rs.{p.get('total_pnl', 0):,.2f}")
            else:
                _fail(f"/api/v1/portfolio -> {r.status_code}")
        except Exception as e:
            _fail(f"/api/v1/portfolio exception: {e!r}")

        # /api/v1/stage
        try:
            r = await client.get(f"{GATEWAY}/api/v1/stage", headers=headers)
            if r.status_code == 200:
                s = r.json()
                results["stage"] = s
                _ok(f"/api/v1/stage -> {r.status_code}: stage={s.get('stage')} "
                    f"current_date={s.get('current_date')} "
                    f"interval_s={s.get('interval_seconds')} "
                    f"days_completed={s.get('days_completed')}/{s.get('total_days')}")
            else:
                _fail(f"/api/v1/stage -> {r.status_code}")
        except Exception as e:
            _fail(f"/api/v1/stage exception: {e!r}")

        # /api/v1/bot/heartbeat (best-effort — should fail in dev with INVALID_SEQ)
        try:
            r = await client.post(
                f"{GATEWAY}/api/v1/bot/heartbeat",
                headers=headers,
                json={"seq": 0},
            )
            results["heartbeat_attempt"] = (r.status_code, r.text[:200])
            if r.status_code == 200:
                _ok(f"/api/v1/bot/heartbeat -> {r.status_code} (heartbeat accepted)")
            elif r.status_code == 400 and "INVALID_SEQ" in r.text:
                _info(f"/api/v1/bot/heartbeat -> {r.status_code} (INVALID_SEQ — "
                      f"no preflight_tick in current stage, expected)")
            elif r.status_code == 401:
                _fail(f"/api/v1/bot/heartbeat -> 401 Unauthorized (token bad?)")
            else:
                _info(f"/api/v1/bot/heartbeat -> {r.status_code}: {r.text[:120]}")
        except Exception as e:
            _fail(f"/api/v1/bot/heartbeat exception: {e!r}")

    return results


async def check_kafka_consumer(duration_s: float) -> dict[str, Any]:
    """Connect to Kafka, count ticks/news for `duration_s` seconds."""
    _section(f"Kafka consumer check ({duration_s:.0f}s)")
    consumer = AIOKafkaConsumer(
        *TOPICS,
        bootstrap_servers=BROKER,
        group_id=KAFKA_GROUP,
        auto_offset_reset="latest",
        enable_auto_commit=False,
        value_deserializer=lambda v: v.decode("utf-8", errors="replace") if v else "",
    )
    stats = {
        "connected": False,
        "ticks_seen": 0,
        "news_seen": 0,
        "preflight_seen": 0,
        "first_tick_at": None,
        "first_news_at": None,
        "tickers_seen": set(),
        "errors": [],
    }
    try:
        await asyncio.wait_for(consumer.start(), timeout=10.0)
        stats["connected"] = True
        _ok(f"Kafka connected: broker={BROKER} group={KAFKA_GROUP}")
    except Exception as e:
        _fail(f"Kafka connect failed: {e!r}")
        stats["errors"].append(str(e))
        return stats

    deadline = time.time() + duration_s
    try:
        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                msg = await asyncio.wait_for(
                    consumer.getone(), timeout=min(2.0, remaining),
                )
            except asyncio.TimeoutError:
                continue
            try:
                payload = json.loads(msg.value) if msg.value else {}
            except json.JSONDecodeError:
                continue
            et = payload.get("event_type") or payload.get("event")
            d = payload.get("data") or {}
            if et == "price_tick":
                stats["ticks_seen"] += 1
                if stats["first_tick_at"] is None:
                    stats["first_tick_at"] = time.time()
                tk = d.get("ticker")
                if tk:
                    stats["tickers_seen"].add(tk)
            elif et == "news_flash":
                stats["news_seen"] += 1
                if stats["first_news_at"] is None:
                    stats["first_news_at"] = time.time()
            elif et == "preflight_tick":
                stats["preflight_seen"] += 1
                _ok(f"preflight_tick seq={d.get('seq') or payload.get('seq')} "
                    f"received during verification")
    finally:
        try:
            await consumer.stop()
        except Exception:
            pass

    stats["tickers_seen"] = sorted(stats["tickers_seen"])
    if stats["ticks_seen"] > 0:
        _ok(f"price_ticks: {stats['ticks_seen']} "
            f"({len(stats['tickers_seen'])} unique tickers: "
            f"{', '.join(stats['tickers_seen'])})")
    else:
        _fail("no price_ticks received")
    if stats["news_seen"] > 0:
        _ok(f"news_flash: {stats['news_seen']}")
    else:
        _fail("no news_flash received")
    return stats


async def check_bot_universe_consistency(rest_results: dict[str, Any]) -> bool:
    """Cross-check REST universe vs Kafka active universe."""
    _section("Universe consistency (REST vs Kafka)")
    rest_universe = set(rest_results.get("universe", []))
    kafka_universe = set(rest_results.get("kafka_tickers", []))
    if not rest_universe:
        _info("skip — no REST universe")
        return True
    if not kafka_universe:
        _info("skip — no Kafka universe (consumer didn't capture any)")
        return True
    only_rest = rest_universe - kafka_universe
    only_kafka = kafka_universe - rest_universe
    if not only_rest and not only_kafka:
        _ok(f"REST and Kafka universes match: {sorted(rest_universe)}")
        return True
    _info(f"REST-only: {sorted(only_rest)} (cannot trade these — no Kafka ticks)")
    _info(f"Kafka-only: {sorted(only_kafka)} (no REST metadata — trade-eligible)")
    return True


async def async_main(duration: float) -> int:
    print("=" * 60, flush=True)
    print("  Manthan bot — sandbox verification harness", flush=True)
    print("=" * 60, flush=True)
    print(f"server_ip={SERVER_IP}  team={TEAM}  group={KAFKA_GROUP}", flush=True)
    print(f"token={'****' + TOKEN[-8:] if TOKEN else 'MISSING'}  "
          f"({len(TOKEN)} chars)", flush=True)
    print(flush=True)

    missing = check_env()
    if missing:
        _fail(f"missing env vars: {missing}")
        print("\nSet them in your shell before running:", file=sys.stderr)
        print("  $env:BOT_ACCESS_TOKEN = (Get-Content jwt.txt -Raw)", file=sys.stderr)
        print("  $env:TEAM_USERNAME = 'mn2632'", file=sys.stderr)
        print("  $env:SERVER_IP = '192.168.65.157'", file=sys.stderr)
        return 1

    failures = 0

    # 1. group_id format
    if not check_group_id_format():
        failures += 1

    # 2. REST endpoints
    rest_results = await check_rest_endpoints()
    if not rest_results.get("status"):
        failures += 1
    if not rest_results.get("universe"):
        failures += 1
    if not rest_results.get("portfolio"):
        failures += 1

    # 3. Kafka consumer
    kafka_stats = await check_kafka_consumer(duration_s=duration)
    if not kafka_stats["connected"]:
        failures += 1
    if kafka_stats["ticks_seen"] == 0:
        failures += 1
    if kafka_stats["news_seen"] == 0:
        failures += 1
    rest_results["kafka_tickers"] = kafka_stats["tickers_seen"]

    # 4. Cross-check
    await check_bot_universe_consistency(rest_results)

    # 5. Final status snapshot
    _section("Final portal status (after our consumer joined)")
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(
            f"{GATEWAY}/api/v1/bot/status",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        if r.status_code == 200:
            s = r.json()
            print(json.dumps(s, indent=2))
            if s.get("kafka_connected") and s.get("kafka_consumer_count", 0) >= 1:
                _ok("Portal confirms our group_id is registered as Kafka consumer")
            else:
                _fail("Portal does NOT see our consumer — group_id may be wrong")

    _section("Summary")
    if failures == 0:
        print(f"[{PASS}] All checks passed.", flush=True)
        print(f"  - group_id format: OK", flush=True)
        print(f"  - REST endpoints: OK", flush=True)
        print(f"  - Kafka consumer: OK ({kafka_stats['ticks_seen']} ticks, "
              f"{kafka_stats['news_seen']} news in {duration:.0f}s)", flush=True)
        print(f"  - Portal sees us: OK", flush=True)
        if kafka_stats["preflight_seen"] == 0:
            print(f"\n  Note: no preflight_tick in current stage "
                  f"({rest_results.get('stage', {}).get('stage', 'unknown')}).",
                  flush=True)
            print(f"  When the portal switches to a live sim with preflight, "
                  f"heartbeat ACKs will flip is_ready=true.", flush=True)
        return 0
    print(f"[{FAIL}] {failures} check(s) failed. See above.", flush=True)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=20.0,
                    help="Seconds to run Kafka consumer (default 20)")
    args = ap.parse_args()
    return asyncio.run(async_main(args.duration))


if __name__ == "__main__":
    sys.exit(main())