# Mocks — offline development

Per `docs/10 §1`: mock Kafka producer + mock FastAPI gateway for offline dev.

Two pieces:

1. `mocks/mock_kafka_producer.py` — produces synthetic price_tick and
   news_flash events to a local Kafka or in-memory queue, so the bot can be
   tested without the platform.

2. `mocks/mock_gateway.py` — a FastAPI app exposing the same REST surface as
   `<SERVER_IP>:3002/api/v1/...`, with state stored in-memory. Lets you run
   the full `bot.main` end-to-end without touching the platform.

Sketch:
```python
# mocks/mock_kafka_producer.py (sketch)
# - generate a deterministic synthetic universe
# - publish at 5s cadence (configurable)
# - emit price_tick + news_flash with the same schemas the platform uses

# mocks/mock_gateway.py (sketch)
# - FastAPI app on :3002
# - POST /orders -> update in-memory positions, return OrderResponse
# - GET /portfolio -> return snapshot
# - GET /stage -> always return development
# - POST /bot/heartbeat -> always 200
```

No code here yet — separate workstream, but useful when training the model.