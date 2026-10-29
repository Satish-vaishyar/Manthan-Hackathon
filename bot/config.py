"""bot.config — typed Settings object loaded from configs/*.yaml + env vars.

Per docs/10 §3: no magic numbers; every threshold lives in configs/*.yaml.
Per docs/10 §4: configs are merged (default <- env-specific), then env vars
override.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _read_yaml(p: Path) -> dict[str, Any]:
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge — override wins."""
    out = dict(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _expand_env(v: Any) -> Any:
    """Expand ${VAR} placeholders in string leaves using os.environ."""
    if isinstance(v, str) and "${" in v:
        # Replace ${VAR} tokens; missing → empty string (logged at boot).
        def _sub(m: str) -> str:
            var = m.group(1)
            val = os.environ.get(var, "")
            if not val:
                # Surface missing env vars loudly but don't crash the loader.
                import structlog
                structlog.get_logger("config").warning(
                    "missing_env_var", var=var, component="config"
                )
            return val
        import re
        return re.sub(r"\$\{([A-Z0-9_]+)\}", _sub, v)
    return v


@dataclass
class KafkaSettings:
    port: int = 9094
    consumers: int = 4
    topics: list[str] = field(default_factory=lambda: ["market.ticks", "market.news"])
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = False
    group_id: str = "group-default"


@dataclass
class APISettings:
    port: int = 3002
    timeout_s: float = 1.5
    rate_per_min: int = 100  # planned; hard cap is 120 (see docs/06)
    base_url: str = ""


@dataclass
class TimingSettings:
    news_window_ms: int = 400
    hard_deadline_ms: int = 800
    heartbeat_interval_s: float = 10.0  # re-ACK watchdog cadence
    stage_poll_preflight_s: float = 2.5
    stage_poll_simulation_s: float = 30.0


@dataclass
class StrategySettings:
    weights: dict[str, float] = field(default_factory=lambda: {"rank": 0.45, "news": 0.35, "mom": 0.20})
    top_k: int = 8
    max_weight_name: float = 0.15
    rebalance_threshold: float = 0.025
    min_trade_notional: float = 50_000.0
    max_orders_per_day: int = 4
    event_long_threshold: float = 0.6
    event_short_threshold: float = -0.6


@dataclass
class RiskSettings:
    stop_atr_mult: float = 2.5
    dd_soft: float = 0.08
    dd_hard: float = 0.15
    cash_buffer: float = 0.05
    max_short_gross: float = 0.20
    max_name_weight: float = 0.15
    max_sector_weight: float = 0.30
    gross_long_target: dict[str, float] = field(default_factory=lambda: {"risk_on": 0.92, "neutral": 0.75, "risk_off": 0.40})


@dataclass
class TaxSettings:
    ltcg_wait_days: int = 30
    harvest_window_days: int = 10
    min_tax_saving: float = 70.0
    brokerage_per_order: float = 20.0
    dp_charge_per_sell: float = 15.34
    stcg_rate: float = 0.15
    ltcg_rate: float = 0.10
    ltcg_exemption: float = 100_000.0
    fy_start_month: int = 4  # April
    fy_start_day: int = 1
    st_lt_boundary_days: int = 365


@dataclass
class SettlementSettings:
    interval_trading_days: int = 10
    force_close_before_days: int = 2  # close short positions N trading days early


@dataclass
class RegimeSettings:
    """M3 — risk-on / neutral / risk-off (docs/05 §5)."""

    # Hysteresis: must hold `hold_days` consecutive days before switching.
    hold_days: int = 2
    # Inputs (per docs/05 §5):
    breadth_risk_off: float = 0.35   # breadth_pct < this AND market_20d_ret < 0 -> risk-off
    breadth_risk_on: float = 0.60    # breadth_pct > this AND market_20d_ret > 0 -> risk-on
    # Vol regime: market_vol > vol_spike_mult × 120-d median vol -> risk-off
    vol_spike_mult: float = 1.5
    # History window for breadth + vol median
    breadth_window_days: int = 50    # SMA window for "price > SMA" breadth
    market_vol_window_days: int = 20
    market_vol_median_window: int = 120
    # If we have < this many market days of history, stay neutral (no regime call yet)
    warmup_days: int = 20


@dataclass
class EventTaggerSettings:
    """M1a — keyword tagger (docs/05 §3.1)."""

    # Magnitude scaling for signed % features (e.g. "profit jumps 12%")
    magnitude_scale: float = 0.05  # 1.0 maps to ±1.0 when pct * (1/scale) >= 1


@dataclass
class SectorClusterSettings:
    """P2-08 — TF-IDF + KMeans over /stocks descriptions."""

    n_clusters: int = 6
    min_n_clusters: int = 2
    max_n_clusters: int = 10
    random_seed: int = 42
    # TF-IDF params
    max_features: int = 4000
    ngram_max: int = 2
    min_df: int = 2


@dataclass
class SignalsSettings:
    """Container for the M3/M4/M1a + sector clustering + ensemble knobs."""

    regime: RegimeSettings = field(default_factory=RegimeSettings)
    event_tagger: EventTaggerSettings = field(default_factory=EventTaggerSettings)
    sector_cluster: SectorClusterSettings = field(default_factory=SectorClusterSettings)
    # Ensemble weights (sum need not be 1; we renormalise over the signals we have).
    weights: dict[str, float] = field(
        default_factory=lambda: {"rank": 0.45, "news": 0.35, "mom": 0.20}
    )


@dataclass
class Settings:
    env: Literal["development", "sandbox", "live"] = "sandbox"
    server_ip: str = "127.0.0.1"
    team_username: str = ""
    bot_access_token: str = ""  # pre-issued JWT
    dry_run: bool = False
    kafka: KafkaSettings = field(default_factory=KafkaSettings)
    api: APISettings = field(default_factory=APISettings)
    timing: TimingSettings = field(default_factory=TimingSettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    tax: TaxSettings = field(default_factory=TaxSettings)
    settlement: SettlementSettings = field(default_factory=SettlementSettings)
    signals: SignalsSettings = field(default_factory=SignalsSettings)

    # ---- derived helpers ----
    @property
    def group_id(self) -> str:
        return f"group-{self.team_username}" if self.team_username else self.kafka.group_id

    @property
    def broker(self) -> str:
        return f"{self.server_ip}:{self.kafka.port}"

    @property
    def gateway_url(self) -> str:
        return self.api.base_url or f"http://{self.server_ip}:{self.api.port}"


def load_settings(env: str | None = None, dry_run: bool = False) -> Settings:
    """Load + merge configs/{default,<env>}.yaml, then overlay env vars.

    Args:
        env: "development" | "sandbox" | "live". If None, read from $BOT_ENV.
        dry_run: if True, no orders will be sent (still connects to Kafka).

    Returns a fully-typed Settings. Caller is expected to fail-fast if any
    required field (BOT_ACCESS_TOKEN, SERVER_IP, TEAM_USERNAME) is empty.
    """
    env = (env or os.environ.get("BOT_ENV", "sandbox")).lower()

    base = _read_yaml(CONFIGS / "default.yaml")
    env_specific = _read_yaml(CONFIGS / f"{env}.yaml")
    merged = _merge(base, env_specific)

    # Recursively expand ${VAR} in string leaves
    def _expand(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: _expand(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_expand(v) for v in o]
        return _expand_env(o)

    merged = _expand(merged)

    # Build typed object
    s = Settings(
        env=merged.get("env", env),
        server_ip=merged.get("server_ip", "127.0.0.1"),
        team_username=merged.get("team_username", os.environ.get("TEAM_USERNAME", "")),
        bot_access_token=os.environ.get("BOT_ACCESS_TOKEN", merged.get("bot_access_token", "")),
        dry_run=dry_run,
    )

    # Section parsers (only set if present in YAML)
    if "kafka" in merged:
        s.kafka = KafkaSettings(**merged["kafka"])
        s.kafka.auto_offset_reset = merged["kafka"].get("auto_offset_reset", "latest")
        s.kafka.enable_auto_commit = merged["kafka"].get("enable_auto_commit", False)
    if "api" in merged:
        s.api = APISettings(**{k: v for k, v in merged["api"].items() if k in APISettings.__dataclass_fields__})
    if "timing" in merged:
        s.timing = TimingSettings(**{k: v for k, v in merged["timing"].items() if k in TimingSettings.__dataclass_fields__})
    if "strategy" in merged:
        strat = dict(merged["strategy"])
        if "weights" in strat:
            s.strategy.weights = {k: float(v) for k, v in strat["weights"].items()}
            strat.pop("weights")
        s.strategy = StrategySettings(**{k: v for k, v in strat.items() if k in StrategySettings.__dataclass_fields__})
    if "risk" in merged:
        risk = dict(merged["risk"])
        if "gross_long_target" in risk:
            s.risk.gross_long_target = {k: float(v) for k, v in risk["gross_long_target"].items()}
            risk.pop("gross_long_target")
        s.risk = RiskSettings(**{k: v for k, v in risk.items() if k in RiskSettings.__dataclass_fields__})
    if "tax" in merged:
        s.tax = TaxSettings(**{k: v for k, v in merged["tax"].items() if k in TaxSettings.__dataclass_fields__})
    if "settlement" in merged:
        s.settlement = SettlementSettings(**{k: v for k, v in merged["settlement"].items() if k in SettlementSettings.__dataclass_fields__})
    if "signals" in merged:
        sig = dict(merged["signals"])
        if "weights" in sig:
            s.signals.weights = {k: float(v) for k, v in sig["weights"].items()}
            sig.pop("weights")
        # Nested: regime / event_tagger / sector_cluster
        if "regime" in sig:
            s.signals.regime = RegimeSettings(
                **{k: v for k, v in sig["regime"].items() if k in RegimeSettings.__dataclass_fields__}
            )
            sig.pop("regime")
        if "event_tagger" in sig:
            s.signals.event_tagger = EventTaggerSettings(
                **{k: v for k, v in sig["event_tagger"].items() if k in EventTaggerSettings.__dataclass_fields__}
            )
            sig.pop("event_tagger")
        if "sector_cluster" in sig:
            s.signals.sector_cluster = SectorClusterSettings(
                **{k: v for k, v in sig["sector_cluster"].items() if k in SectorClusterSettings.__dataclass_fields__}
            )
            sig.pop("sector_cluster")
        s.signals = SignalsSettings(
            **{k: v for k, v in sig.items() if k in SignalsSettings.__dataclass_fields__},
            weights=s.signals.weights,
            regime=s.signals.regime,
            event_tagger=s.signals.event_tagger,
            sector_cluster=s.signals.sector_cluster,
        )

    return s


def fail_fast(s: Settings) -> None:
    """Refuse to boot if any required env-derived field is missing."""
    import structlog
    log = structlog.get_logger("config")
    missing = []
    if not s.bot_access_token:
        missing.append("BOT_ACCESS_TOKEN")
    if not s.server_ip or s.server_ip == "127.0.0.1":
        missing.append("SERVER_IP")
    if not s.team_username:
        missing.append("TEAM_USERNAME")
    if missing:
        log.error("config_missing_required_vars", missing=missing, component="config")
        raise SystemExit(
            "Refusing to boot: missing required env vars: "
            + ", ".join(missing)
            + ". Set them in your shell or .env file."
        )