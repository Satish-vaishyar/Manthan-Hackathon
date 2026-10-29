"""bot.contracts — pydantic v2 models for Kafka events + REST API responses.

Per docs/03 §1 & §2: every event and API response is parsed here with
extra="ignore" so unknown fields don't crash the bot.

Importing this module is cheap; pydantic v2 does NOT validate at import time.
"""
from __future__ import annotations

from datetime import date as _date, datetime
from typing import Any, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared config: ignore unknown fields
# ---------------------------------------------------------------------------
class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# ===========================================================================
# Kafka events — docs/03 §1
# ===========================================================================
class PriceTickData(_Base):
    """One OHLCV candle payload (the `data` field of a price_tick)."""
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class PriceTickEvent(_Base):
    """market.ticks message: one ticker candle for one simulated_date."""
    event_type: Literal["price_tick"] = "price_tick"
    timestamp: datetime
    simulated_date: _date
    data: PriceTickData


class NewsFlashData(_Base):
    """News payload; sentiment_hint is optional (platform may pre-score)."""
    ticker: str
    headline: str
    description: Optional[str] = None
    source: Optional[str] = None
    sentiment_hint: Optional[float] = None  # if the platform pre-scores


class NewsFlashEvent(_Base):
    """market.news message: one headline for one ticker on a simulated day."""
    event_type: Literal["news_flash"] = "news_flash"
    timestamp: datetime
    simulated_date: _date
    data: NewsFlashData


class PreflightTickEvent(_Base):
    """Sent ONLY during preflight stage. Must be ACK'd via /bot/heartbeat.

    The platform's standby message has no simulated_date — it's a connectivity
    probe, not a market event. Shape (observed live):
        {"event": "preflight_tick", "event_type": "preflight_tick",
         "timestamp": "...", "seq": 624, "message": "Pre-flight standby tick...",
         "data": {"status": "STANDBY", "seq": 624}}
    """
    event_type: Literal["preflight_tick", "preflight_news"] = "preflight_tick"
    seq: int
    timestamp: datetime
    simulated_date: Optional[_date] = None
    message: Optional[str] = None
    data: dict[str, Any] = Field(default_factory=dict)


# Discriminated union — used by the day assembler
KafkaEvent = PriceTickEvent | NewsFlashEvent | PreflightTickEvent


# ===========================================================================
# REST API responses — docs/03 §2
# ===========================================================================
class OrderRequest(_Base):
    """Request body for POST /api/v1/orders.

    `quantity` is intentionally NOT constrained to gt=0 at the pydantic layer —
    we want the validator to receive bad input and reject it with the canonical
    `INVALID_QUANTITY` rejection code (per docs/03 §3). Pydantic raising here
    would crash the bot.
    """
    ticker: str
    action: str  # validated by OrderValidator
    quantity: int = 0
    date: _date  # simulated date — REQUIRED per docs/04 (decision-latency score)
    reason: Optional[str] = None


class OrderResponse(_Base):
    """Response from POST /api/v1/orders or GET /api/v1/orders.

    The live server returns a different shape than the original docs:
        {"order_id": "...", "execution_price": 403.08, "simulated_date": "...",
         "executed_at": "...", "dp_charges": 0, ...}
    so we accept both shapes via AliasChoices (server keys first, doc keys as
    fallback). Locally-constructed rejections still use the doc field names.
    """
    id: str = Field(validation_alias=AliasChoices("order_id", "id"))
    ticker: str
    action: str
    quantity: int
    price: float = Field(validation_alias=AliasChoices("execution_price", "price", "fill_price"))
    brokerage: float = 0.0
    dp_charge: float = Field(default=0.0, validation_alias=AliasChoices("dp_charges", "dp_charge"))
    # P3-20: "UNKNOWN" = server returned an empty/unparseable success body;
    # the order's true outcome is resolved by an immediate portfolio
    # reconcile (never applied to the local ledger as a phantom fill).
    status: Literal["FILLED", "REJECTED", "PENDING", "UNKNOWN"] = "FILLED"
    reject_code: Optional[str] = Field(default=None, validation_alias=AliasChoices("reject_code", "rejection_reason"))
    reject_message: Optional[str] = Field(default=None, validation_alias=AliasChoices("reject_message", "message"))
    date: _date = Field(validation_alias=AliasChoices("simulated_date", "order_date", "date"))
    timestamp: datetime = Field(validation_alias=AliasChoices("executed_at", "server_received_at", "timestamp"))


class Position(_Base):
    """One holding in the portfolio response. NOTE: the server can report a
    NEGATIVE quantity here — that is a short carried inside `holdings`, and
    main._reconcile converts it into a ledger ShortPos."""
    ticker: str
    quantity: int
    # Live server sends `avg_buy_price` / `average_buy_price`; docs say avg_price.
    avg_price: float = Field(validation_alias=AliasChoices(
        "avg_buy_price", "average_buy_price", "avg_price", "buy_price"))
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None


class ShortPosition(_Base):
    """Explicit short from the portfolio response (`shorts` list)."""
    ticker: str
    quantity: int
    entry_price: float
    margin_locked: float
    open_date: _date
    days_to_settlement: Optional[int] = None


class PortfolioResponse(_Base):
    """From GET /api/v1/portfolio

    The portal uses both American (realized, available) and British (realised)
    spellings across its endpoints, and may add fields like `total_wealth`,
    `margin_locked`, `total_charges`, etc. that we don't model. We alias the
    common variants so either spell works.
    """
    # Cash & equity — portal returns `available_cash` and `total_wealth` /
    # `total_portfolio_value` for equity. Fall back to defaults if neither
    # is present so a partial response still parses.
    cash: float = Field(default=0.0, validation_alias=AliasChoices(
        "available_cash", "cash_balance", "cash"))
    equity: float = Field(default=0.0, validation_alias=AliasChoices(
        "total_wealth", "total_portfolio_value", "equity"))
    # P&L — portal uses American spelling (realized_pnl, unrealized_pnl).
    realised_pnl: float = Field(default=0.0, validation_alias=AliasChoices(
        "realized_pnl", "realised_pnl"))
    unrealised_pnl: float = Field(default=0.0, validation_alias=AliasChoices(
        "unrealized_pnl", "unrealised_pnl"))
    # Positions — live server returns `holdings` (longs); docs say `positions`.
    positions: list[Position] = Field(default_factory=list,
                                      validation_alias=AliasChoices("holdings", "positions"))
    shorts: list[ShortPosition] = Field(default_factory=list)
    day_trades_remaining: Optional[int] = None
    orders_today: Optional[int] = None
    # Portal does not include a timestamp on the portfolio response; default
    # to "now" so downstream code always sees a valid timestamp.
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StageInfo(_Base):
    """From GET /api/v1/stage"""
    stage: Literal["development", "ready", "preflight", "simulation", "paused", "completed"]
    status: Optional[str] = None  # "running" | "paused" | "idle" | "completed"
    interval_seconds: float = 5.0
    simulated_date: Optional[_date] = None
    total_days: Optional[int] = None
    days_completed: Optional[int] = None
    active_dataset: Optional[str] = None
    short_selling_enabled: bool = False
    current_date: Optional[_date] = None


class BotStatus(_Base):
    """From GET /api/v1/bot/status"""
    kafka_connected: bool = False
    kafka_consumer_count: int = 0
    heartbeat_received: bool = False
    last_seq: Optional[int] = None
    last_seq_at: Optional[datetime] = None
    is_ready: bool = False
    stage: Optional[str] = None


class HeartbeatRequest(_Base):
    """POST /api/v1/bot/heartbeat body — echoes the preflight tick seq."""
    seq: Optional[int] = None
    timestamp: Optional[datetime] = None


class HeartbeatResponse(_Base):
    """Heartbeat ACK result; accepted=False means the watchdog will retry."""
    accepted: bool
    seq: Optional[int] = None
    error: Optional[str] = None


class StockInfo(_Base):
    """One row of GET /api/v1/stocks (universe + sector metadata)."""
    ticker: str
    name: str
    description: Optional[str] = None
    sector: Optional[str] = None
    active: bool = True


# ===========================================================================
# Rejection codes — docs/03 §3
# ===========================================================================
class RejectCode:
    """Canonical rejection codes. The validator pre-screens these client-side
    so we should never see them at the gateway, but if we do we log + don't retry."""

    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
    INSUFFICIENT_MARGIN = "INSUFFICIENT_MARGIN"
    INSUFFICIENT_HOLDINGS = "INSUFFICIENT_HOLDINGS"
    INVALID_TICKER = "INVALID_TICKER"
    INVALID_QUANTITY = "INVALID_QUANTITY"
    INVALID_PRICE = "INVALID_PRICE"
    STALE_DATE = "STALE_DATE"
    FUTURE_DATE = "FUTURE_DATE"
    MARKET_CLOSED = "MARKET_CLOSED"
    RATE_LIMITED = "RATE_LIMITED"
    DUPLICATE_ORDER = "DUPLICATE_ORDER"
    SHORT_NOT_ALLOWED = "SHORT_NOT_ALLOWED"
    ACCOUNT_BLOCKED = "ACCOUNT_BLOCKED"
    RISK_LIMIT = "RISK_LIMIT"
    SETTLEMENT_LOCK = "SETTLEMENT_LOCK"
    INVALID_SEQ = "INVALID_SEQ"
    SERVER_ERROR = "SERVER_ERROR"

    ALL = {
        INSUFFICIENT_CASH, INSUFFICIENT_MARGIN, INSUFFICIENT_HOLDINGS,
        INVALID_TICKER, INVALID_QUANTITY, INVALID_PRICE,
        STALE_DATE, FUTURE_DATE, MARKET_CLOSED, RATE_LIMITED,
        DUPLICATE_ORDER, SHORT_NOT_ALLOWED, ACCOUNT_BLOCKED,
        RISK_LIMIT, SETTLEMENT_LOCK, INVALID_SEQ, SERVER_ERROR,
    }


# ===========================================================================
# DayBundle — passed between assembler → features → strategy → constructor
# ===========================================================================
class DayBundle(_Base):
    """Everything we know about one simulated trading day.

    Populated by day_assembler once all ticks + news for `simulated_date` are
    in (or the hard deadline fires), then handed to the strategy stack.
    """
    simulated_date: _date
    ticks: dict[str, PriceTickData] = Field(default_factory=dict)
    news: list[NewsFlashData] = Field(default_factory=list)
    expected_tickers: set[str] = Field(default_factory=set)
    assembled_at: datetime
    assembly_latency_ms: float
    complete_by_count: bool = False  # True = all expected tickers arrived
    complete_by_deadline: bool = False  # True = hard deadline fired