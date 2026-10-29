"""bot.ingest.kafka_consumer — aiokafka consumer(s) pushing into asyncio.Queue.

Per docs/03 §1 and docs/10 §1:
    - bootstrap_servers = <SERVER_IP>:9094
    - group_id = group-<TEAM_USERNAME> (exact prefix)
    - auto_offset_reset="latest", enable_auto_commit=False
    - up to 10 partitions => up to 10 consumer tasks (we use settings.kafka.consumers)
    - reconnect with backoff on broker drop
    - parses each message via contracts.KafkaEvent

The consumer yields parsed events into a single bounded asyncio.Queue.
The day_assembler drains this queue per simulated day.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Optional

import structlog
from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError
from pydantic import ValidationError

from ..config import Settings
from ..contracts import (
    KafkaEvent,
    NewsFlashEvent,
    PreflightTickEvent,
    PriceTickEvent,
)

_log = structlog.get_logger("kafka")


@dataclass
class ConsumerStats:
    started_at: float = 0.0
    connected_at: float = 0.0
    disconnected_count: int = 0
    reconnected_count: int = 0
    messages_total: int = 0
    parse_failures: int = 0
    by_topic: dict[str, int] = None

    def __post_init__(self) -> None:
        if self.by_topic is None:
            self.by_topic = {}


class KafkaIngest:
    """N concurrent consumer tasks, all in the same group, draining into one queue.

    Each task is responsible for its own subset of partitions (Kafka rebalances
    automatically). If the broker drops, we sleep with backoff and retry.
    """

    def __init__(self, settings: Settings, queue: asyncio.Queue[KafkaEvent]) -> None:
        self.settings = settings
        self.queue = queue
        self.stats = ConsumerStats()
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._consumers: list[AIOKafkaConsumer] = []

    async def start(self) -> None:
        """Spawn N consumer tasks. Returns when all are connected."""
        self.stats.started_at = time.time()
        n = max(1, self.settings.kafka.consumers)
        # We use ONE consumer (single group) to keep ordering across partitions
        # simple — but we still create `n` tasks for parallelism by topic.
        # For the spec (10 partitions, N consumers in same group), Kafka does
        # the partition assignment. So we start one consumer and let the
        # broker assign all partitions to it; future-proof: create N if needed.
        self._tasks = [
            asyncio.create_task(self._run_consumer(idx), name=f"kafka-{idx}")
            for idx in range(n)
        ]
        # Wait briefly for at least one to connect (best-effort)
        try:
            await asyncio.wait_for(self._first_connected.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            _log.warning("kafka_connect_timeout", component="kafka")

    @property
    def _first_connected(self) -> asyncio.Event:
        """Internal: fires when first consumer is connected."""
        if not hasattr(self, "_first_connected_event"):
            self._first_connected_event = asyncio.Event()
        return self._first_connected_event

    async def stop(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._consumers.clear()

    async def _run_consumer(self, idx: int) -> None:
        """Single consumer task. Loops with backoff on disconnects."""
        s = self.settings
        backoff = 1.0
        while not self._stop.is_set():
            consumer = AIOKafkaConsumer(
                *s.kafka.topics,
                bootstrap_servers=s.broker,
                group_id=s.group_id,
                auto_offset_reset=s.kafka.auto_offset_reset,
                enable_auto_commit=s.kafka.enable_auto_commit,
                value_deserializer=lambda v: v.decode("utf-8", errors="replace") if v else "",
                client_id=f"manthan-{s.team_username}-{idx}",
            )
            try:
                await consumer.start()
                self._consumers.append(consumer)
                self.stats.connected_at = time.time() if not self.stats.connected_at else self.stats.connected_at
                if idx == 0:
                    self._first_connected.set()
                backoff = 1.0
                _log.info(
                    "kafka_connected",
                    **{"broker": s.broker, "group": s.group_id, "topics": s.kafka.topics, "idx": idx},
                    component="kafka",
                )
                await self._drain(consumer)
            except KafkaError as e:
                _log.warning(
                    "kafka_error",
                    **{"err": repr(e), "idx": idx, "backoff_s": backoff},
                    component="kafka",
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                _log.warning(
                    "kafka_unexpected",
                    **{"err": repr(e), "idx": idx, "backoff_s": backoff},
                    component="kafka",
                )
            finally:
                try:
                    await consumer.stop()
                except Exception:  # noqa: BLE001
                    pass
                if consumer in self._consumers:
                    self._consumers.remove(consumer)

            if self._stop.is_set():
                return
            self.stats.disconnected_count += 1
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
            self.stats.reconnected_count += 1

    async def _drain(self, consumer: AIOKafkaConsumer) -> None:
        """Read messages, parse, push to queue."""
        async for msg in consumer:
            if self._stop.is_set():
                return
            self.stats.messages_total += 1
            self.stats.by_topic[msg.topic] = self.stats.by_topic.get(msg.topic, 0) + 1
            try:
                payload = json.loads(msg.value) if msg.value else {}
            except json.JSONDecodeError:
                self.stats.parse_failures += 1
                _log.warning("non_json_message", topic=msg.topic, component="kafka")
                continue

            event = self._parse_event(msg.topic, payload)
            if event is None:
                self.stats.parse_failures += 1
                continue
            try:
                # Bounded queue — drop oldest under extreme pressure rather
                # than block the consumer (Kafka is the source of truth).
                if self.queue.full():
                    try:
                        _ = self.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await self.queue.put(event)
            except Exception as e:  # noqa: BLE001
                _log.warning("queue_put_failed", err=repr(e), component="kafka")

    def _parse_event(self, topic: str, payload: dict) -> Optional[KafkaEvent]:
        """Dispatch to the right pydantic model based on topic + event_type."""
        et = payload.get("event_type") or payload.get("event")
        try:
            if et == "price_tick" or topic == "market.ticks":
                return PriceTickEvent.model_validate(payload)
            if et == "news_flash" or topic == "market.news":
                return NewsFlashEvent.model_validate(payload)
            if et == "preflight_tick":
                return PreflightTickEvent.model_validate(payload)
            _log.warning("unknown_event_type", event_type=et, topic=topic, component="kafka")
            return None
        except ValidationError as e:
            _log.warning(
                "event_parse_failed",
                **{"event_type": et, "topic": topic, "errors": e.errors()[:3]},
                component="kafka",
            )
            return None