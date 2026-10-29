"""
connect.py — minimal Kafka consumer for the Manthan trading bot.

Subscribes to market.ticks and market.news, and ACKs every preflight_tick
via POST /api/v1/bot/heartbeat. Once both are flowing, the portal flips
the team's readiness panel from "Waiting for heartbeat and Kafka connection"
to "Ready".

Usage:
    python scripts/connect.py

Env vars (set in .env or shell):
    SERVER_IP           e.g. 192.168.65.157
    BOT_ACCESS_TOKEN    JWT from the portal login (Bearer token)
    TEAM_USERNAME       your team handle — used for Kafka group_id
                        (must be group-<TEAM_USERNAME>, the readiness tracker
                        keys off this prefix)
"""

import asyncio
import json
import os
import signal
import sys
import time
from typing import Optional

import httpx
from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
SERVER_IP = os.environ.get("SERVER_IP", "192.168.65.157")
KAFKA_PORT = int(os.environ.get("KAFKA_PORT", "9094"))
GATEWAY = f"http://{SERVER_IP}:3002"
BOT_TOKEN = os.environ.get("BOT_ACCESS_TOKEN", "")
TEAM_USERNAME = os.environ.get("TEAM_USERNAME", "mn2632")
KAFKA_GROUP = f"group-{TEAM_USERNAME}"

TOPICS = ["market.ticks", "market.news"]
BROKER = f"{SERVER_IP}:{KAFKA_PORT}"

# --------------------------------------------------------------------------
# Stats / state
# --------------------------------------------------------------------------
state = {
    "started_at": time.time(),
    "ticks_seen": 0,
    "news_seen": 0,
    "preflight_seen": 0,
    "heartbeats_sent": 0,
    "heartbeat_failures": 0,
    "last_preflight_seq": None,
    "last_heartbeat_at": None,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Heartbeat sender
# --------------------------------------------------------------------------
async def send_heartbeat(seq: Optional[int]) -> bool:
    """POST /api/v1/bot/heartbeat with the seq from the preflight_tick."""
    if not BOT_TOKEN:
        log("  !! BOT_ACCESS_TOKEN not set — skipping heartbeat (set it in env)")
        return False

    payload = {"seq": seq} if seq is not None else {}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(
                f"{GATEWAY}/api/v1/bot/heartbeat",
                headers={"Authorization": f"Bearer {BOT_TOKEN}"},
                json=payload,
            )
            if r.status_code == 200:
                state["heartbeats_sent"] += 1
                state["last_heartbeat_at"] = time.time()
                return True
            else:
                state["heartbeat_failures"] += 1
                log(f"  !! heartbeat HTTP {r.status_code}: {r.text[:200]}")
                return False
    except Exception as e:
        state["heartbeat_failures"] += 1
        log(f"  !! heartbeat exception: {e!r}")
        return False


# --------------------------------------------------------------------------
# Optional: periodic heartbeat watchdog (≥1 ACK per 15 s)
# --------------------------------------------------------------------------
async def heartbeat_watchdog(stop_event: asyncio.Event) -> None:
    """Re-send the last seq every 10 s so the portal sees us as alive
    even between preflight_ticks."""
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            await send_heartbeat(state["last_preflight_seq"])
        else:
            break


# --------------------------------------------------------------------------
# Status poller (logs the portal's view of our readiness)
# --------------------------------------------------------------------------
async def status_poller(stop_event: asyncio.Event) -> None:
    if not BOT_TOKEN:
        return
    while not stop_event.is_set():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    f"{GATEWAY}/api/v1/bot/status",
                    headers={"Authorization": f"Bearer {BOT_TOKEN}"},
                )
                if r.status_code == 200:
                    s = r.json()
                    log(
                        f"status: kafka_connected={s.get('kafka_connected')} "
                        f"consumers={s.get('kafka_consumer_count')} "
                        f"heartbeat_received={s.get('heartbeat_received')} "
                        f"last_seq={s.get('last_seq')} "
                        f"is_ready={s.get('is_ready')}"
                    )
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            continue
        else:
            break


# --------------------------------------------------------------------------
# Kafka loop
# --------------------------------------------------------------------------
async def consume(stop_event: asyncio.Event) -> None:
    """Connect Kafka, subscribe, and react to messages."""
    while not stop_event.is_set():
        consumer = AIOKafkaConsumer(
            *TOPICS,
            bootstrap_servers=BROKER,
            group_id=KAFKA_GROUP,
            auto_offset_reset="latest",
            enable_auto_commit=False,
            value_deserializer=lambda v: v.decode("utf-8", errors="replace") if v else "",
        )
        try:
            log(f"connecting to Kafka {BROKER} (group={KAFKA_GROUP}) …")
            await consumer.start()
            log(f"connected — subscribed to {TOPICS}")
        except KafkaError as e:
            log(f"Kafka connect failed: {e!r}; retrying in 5 s")
            try:
                await consumer.stop()
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            else:
                return

        try:
            async for msg in consumer:
                if stop_event.is_set():
                    break

                # Parse JSON; tolerate bad payloads
                try:
                    payload = json.loads(msg.value) if msg.value else {}
                except json.JSONDecodeError:
                    log(f"  (skip non-json msg on {msg.topic})")
                    continue

                # Dispatch: price_tick / news_flash / preflight_tick
                et = payload.get("event_type") or payload.get("event")
                if msg.topic == "market.ticks" and et == "preflight_tick":
                    seq = payload.get("seq") or (
                        payload.get("data") or {}
                    ).get("seq")
                    state["preflight_seen"] += 1
                    state["last_preflight_seq"] = seq
                    log(f"preflight_tick seq={seq} -> ACK")
                    await send_heartbeat(seq)
                elif et == "price_tick":
                    state["ticks_seen"] += 1
                    if state["ticks_seen"] <= 3 or state["ticks_seen"] % 50 == 0:
                        d = payload.get("data") or {}
                        log(
                            f"price_tick #{state['ticks_seen']} "
                            f"{payload.get('simulated_date')} "
                            f"{d.get('ticker')} close={d.get('close')}"
                        )
                elif msg.topic == "market.news" or et == "news_flash":
                    state["news_seen"] += 1
                    if state["news_seen"] <= 3 or state["news_seen"] % 50 == 0:
                        d = payload.get("data") or {}
                        log(
                            f"news_flash #{state['news_seen']} "
                            f"{payload.get('simulated_date')} "
                            f"{d.get('ticker')}: "
                            f"{(d.get('headline') or '')[:60]}"
                        )
                else:
                    # Unknown event_type — log once and continue
                    log(f"  (unknown event on {msg.topic}: {et})")
        except KafkaError as e:
            log(f"Kafka stream error: {e!r}; reconnecting in 5 s")
        finally:
            try:
                await consumer.stop()
                log("Kafka consumer stopped")
            except Exception:
                pass


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
async def main() -> int:
    log("=== Manthan bot connect ===")
    log(f"server={SERVER_IP}  kafka={BROKER}  group={KAFKA_GROUP}")
    log(f"gateway={GATEWAY}")
    if BOT_TOKEN:
        log(f"token=Bearer ****{BOT_TOKEN[-8:]}  ({len(BOT_TOKEN)} chars)")
    else:
        log("!! BOT_ACCESS_TOKEN is empty — heartbeat ACKs will fail")

    stop_event = asyncio.Event()

    def _on_signal(*_):
        log("shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, _on_signal)
        except NotImplementedError:
            signal.signal(sig, lambda *_: _on_signal())

    # Run the three coroutines concurrently
    tasks = [
        asyncio.create_task(consume(stop_event), name="kafka"),
        asyncio.create_task(heartbeat_watchdog(stop_event), name="watchdog"),
        asyncio.create_task(status_poller(stop_event), name="status"),
    ]
    try:
        await stop_event.wait()
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    uptime = time.time() - state["started_at"]
    log(
        f"=== shutdown after {uptime:.0f}s — "
        f"ticks={state['ticks_seen']} news={state['news_seen']} "
        f"preflight={state['preflight_seen']} "
        f"hb_sent={state['heartbeats_sent']} "
        f"hb_fail={state['heartbeat_failures']} ==="
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(0)