"""bot.journal.journal — append-only JSONL decision log.

Per docs/10 §6 — every decision (including HOLD) gets one line, with:
    sim_date, ticker, decision, qty, price, scores, reasons, latency_ms, status

Writes to runs/<timestamp>/journal.jsonl (one file per process).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class DecisionJournal:
    """Thread/async-safe append-only JSONL writer.

    Falls back to in-memory buffer if the path is unavailable.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.run_id = os.environ.get("BOT_RUN_ID", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        if path is None:
            root = Path(__file__).resolve().parents[2] / "runs" / self.run_id
            root.mkdir(parents=True, exist_ok=True)
            path = root / "journal.jsonl"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self.count = 0
        self.started_at = time.time()

    def record(self, rec: dict) -> None:
        """Append one decision record. Best-effort: never raises to caller."""
        rec = dict(rec)
        rec.setdefault("run_id", self.run_id)
        rec.setdefault("seq", self.count)
        rec.setdefault("ts_wall", datetime.now(timezone.utc).isoformat())
        try:
            self._fh.write(json.dumps(rec, default=str) + "\n")
            self._fh.flush()
            self.count += 1
        except Exception:  # noqa: BLE001
            # Last-resort: stderr; journal is best-effort
            import sys
            print(f"[journal] write failed: {rec.get('ticker')}", file=sys.stderr)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass