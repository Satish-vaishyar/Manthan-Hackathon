"""Record market.ticks + market.news from the venue Kafka into data/raw/captures/<run_id>/.

  python src/record_kafka.py --minutes 45            # one full sandbox loop at 5 s/day is ~42 min
  python src/record_kafka.py --bootstrap 192.168.65.157:9094 --minutes 90

Uses its own consumer group (recorder-<random>), so it never takes partitions away from the trading bot's
group-<team_username>. Output format matches the existing captures: one JSON object per line with
received_at, topic, partition, offset, payload.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml
from kafka import KafkaConsumer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "model2.yaml"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default=cfg["kafka"]["bootstrap"])
    ap.add_argument("--minutes", type=float, default=45)
    args = ap.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    out = ROOT / cfg["data"]["captures_dir"] / run_id
    out.mkdir(parents=True, exist_ok=True)
    files = {"market.ticks": open(out / "ticks.jsonl", "a", encoding="utf8"),
             "market.news": open(out / "news.jsonl", "a", encoding="utf8")}
    group = f"{cfg['kafka']['group_prefix']}-{uuid.uuid4().hex[:8]}"
    consumer = KafkaConsumer(*cfg["kafka"]["topics"], bootstrap_servers=args.bootstrap, group_id=group,
                             auto_offset_reset="latest", enable_auto_commit=False,
                             value_deserializer=lambda b: json.loads(b.decode("utf8")))
    print(f"recording to {out} (group {group}) for {args.minutes} min — Ctrl+C to stop early")
    counts = {"market.ticks": 0, "market.news": 0}
    started, last_print, loops_seen = time.time(), 0.0, set()
    try:
        while time.time() - started < args.minutes * 60:
            for tp, msgs in consumer.poll(timeout_ms=1000).items():
                for m in msgs:
                    rec = {"received_at": datetime.now(timezone.utc).isoformat(), "topic": m.topic,
                           "partition": m.partition, "offset": m.offset, "payload": m.value}
                    files[m.topic].write(json.dumps(rec, ensure_ascii=False) + "\n")
                    counts[m.topic] += 1
                    sb = (m.value or {}).get("sandbox") or {}
                    if "loop_count" in sb:
                        loops_seen.add(sb["loop_count"])
            if time.time() - last_print > 30:
                for f in files.values():
                    f.flush()
                print(f"  {time.time() - started:6.0f}s ticks={counts['market.ticks']} news={counts['market.news']}"
                      f" loops={sorted(loops_seen)}", flush=True)
                last_print = time.time()
    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        for f in files.values():
            f.close()
        consumer.close()
        json.dump({"run_id": run_id, "broker": args.bootstrap, "group_id": group,
                   "duration_seconds": round(time.time() - started, 1), "messages": counts,
                   "loops_seen": sorted(loops_seen)}, open(out / "summary.json", "w"), indent=2)
        print(f"done: {counts} -> {out}")


if __name__ == "__main__":
    main()
