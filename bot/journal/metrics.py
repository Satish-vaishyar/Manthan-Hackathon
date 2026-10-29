"""bot.journal.metrics — aggregate run-time metrics from journal + state.

Per docs/12 §1 (Pillar 1 — reliability): uptime, reject rate, latency stats,
cash utilisation. Cheap to compute; recompute from journal.jsonl on demand.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class RunMetrics:
    """Aggregate counters/percentiles computed from one journal.jsonl."""
    started_at: float = 0.0
    uptime_s: float = 0.0
    decisions_total: int = 0
    buys: int = 0
    sells: int = 0
    shorts: int = 0
    covers: int = 0
    holds: int = 0
    fills: int = 0
    rejections: int = 0
    reject_codes: dict[str, int] = field(default_factory=dict)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    orders_per_day: dict[str, int] = field(default_factory=dict)


def compute_metrics(journal_path: Path, *, started_at: Optional[float] = None) -> RunMetrics:
    """Stream the journal file once and compute aggregate metrics."""
    m = RunMetrics()
    m.started_at = started_at or time.time()
    latencies: list[float] = []
    if not Path(journal_path).exists():
        return m
    with open(journal_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            m.decisions_total += 1
            d = rec.get("decision")
            if d == "BUY":
                m.buys += 1
            elif d == "SELL":
                m.sells += 1
            elif d == "SHORT":
                m.shorts += 1
            elif d == "COVER":
                m.covers += 1
            elif d == "HOLD":
                m.holds += 1
            sd = rec.get("sim_date")
            if sd:
                m.orders_per_day[sd] = m.orders_per_day.get(sd, 0) + 1
            if rec.get("status") == "FILLED":
                m.fills += 1
            elif rec.get("status") == "REJECTED":
                m.rejections += 1
                code = rec.get("reject_code")
                if code:
                    m.reject_codes[code] = m.reject_codes.get(code, 0) + 1
            lat = rec.get("latency_ms") or {}
            if isinstance(lat, dict):
                v = lat.get("order") or lat.get("ack") or lat.get("decide")
                if isinstance(v, (int, float)):
                    latencies.append(float(v))
    if latencies:
        latencies.sort()
        m.p50_latency_ms = latencies[len(latencies) // 2]
        m.p95_latency_ms = latencies[int(len(latencies) * 0.95)]
    m.uptime_s = max(0.0, time.time() - m.started_at)
    return m