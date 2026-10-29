"""bot — Manthan autonomous trading bot package.

Pipeline: ingest (Kafka -> DayAssembler) -> features/signals (M1a, M4, M5,
ensemble) -> portfolio (constructor, risk) -> execution (validator,
order_manager) -> journal. Entry point: `python -m bot.main`.
"""
