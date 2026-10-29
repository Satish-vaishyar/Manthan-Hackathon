"""bot.config — typed Settings object loaded from configs/*.yaml + env vars.

Per docs/10 §3: no magic numbers; every threshold lives in configs/*.yaml.
Per docs/10 §4: configs are merged (default <- env-specific), then env vars
override.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _read_yaml(p: Path) -> dict[str, Any]:
    """Read a YAML file into a dict; missing file -> {} (env yaml optional)."""
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
    """Kafka ingest knobs. `consumers` is the number of consumer tasks in
    one group (broker assigns partitions); auto-commit stays off — the
    simulated feed is replayed per run, so offsets aren't ours to commit."""
    port: int = 9094
    consumers: int = 4
    topics: list[str] = field(default_factory=lambda: ["market.ticks", "market.news"])
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = False
    group_id: str = "group-default"


@dataclass
class APISettings:
    """REST gateway knobs. timeout_s is per-request; rate_per_min feeds the
    token bucket (stay below the platform's 120/min hard cap or you get
    429-penalised)."""
    port: int = 3002
    timeout_s: float = 1.5
    rate_per_min: int = 100  # planned; hard cap is 120 (see docs/06)
    base_url: str = ""


@dataclass
class TimingSettings:
    """Day-assembly + polling cadence (milliseconds/seconds).

    news_window_ms: wait this long after the day's ticks for trailing news
        before finalising a DayBundle by count.
    hard_deadline_ms: finalise the day regardless after this budget.
    """
    news_window_ms: int = 400
    hard_deadline_ms: int = 800
    heartbeat_interval_s: float = 10.0  # re-ACK watchdog cadence
    stage_poll_preflight_s: float = 2.5
    # P3-14: faster stage polling — when the org flips simulation on/off we
    # detect it in <=10s instead of <=30s, so less downtime.
    stage_poll_simulation_s: float = 10.0
    # P3-14: reconcile interval. Was implicitly 30s in _health_loop; make
    # it explicit and tighter (15s) so cash/positions stay in sync with the
    # server.
    reconcile_interval_s: float = 15.0


@dataclass
class StrategySettings:
    """Signal-ensemble + trade-generation knobs.

    weights: static blend of {rank, news, mom} sub-scores (renormalised).
    top_k: max names in the target portfolio; max_weight_name: per-name cap.
    rebalance_threshold: min |target-current| weight move worth trading.
    min_trade_notional: fee-efficiency floor for entries (~Rs).
    max_orders_per_day: hard daily order budget (concentrates conviction).
    """
    weights: dict[str, float] = field(default_factory=lambda: {"rank": 0.45, "news": 0.35, "mom": 0.20})
    top_k: int = 8
    max_weight_name: float = 0.15
    rebalance_threshold: float = 0.025
    # P3-14: raised from 50k -> 75k. Each round-trip costs ~Rs 35 in
    # brokerage+DP, so Rs 50k trades pay 0.07% per side; Rs 75k trades pay
    # 0.05%. Fewer, larger trades also concentrate conviction.
    min_trade_notional: float = 75_000.0
    # P3-19: exits use a LOWER notional floor than entries. A legacy
    # position from a previous run that's worth Rs 20K should still be
    # sellable when the model no longer wants it — Rs 20 brokerage on a
    # Rs 20K exit is fine; letting a loser rot is not. Entries stay at
    # min_trade_notional (75K) for fee efficiency.
    exit_min_notional: float = 5_000.0
    max_orders_per_day: int = 4
    event_long_threshold: float = 0.6
    event_short_threshold: float = -0.6
    # P3-13: tickers to exclude from scoring entirely (per-ticker blacklist
    # from tradebook analysis — see runs/_tradebook_analysis.md §6.6).
    exclude_tickers: list[str] = field(default_factory=list)
    # P3-14: minimum PnL/charge ratio that justifies an order. Trades
    # expected to make less than this multiple of the brokerage cost are
    # skipped (the broker charges ~Rs 35 round-trip). 3.0x means we only
    # take trades expected to gross >Rs 105 in PnL.
    min_pnl_to_charge_ratio: float = 3.0
    # P3-14: forwarded into the constructor so cash-utilization targeting
    # reads from one place. The RiskSettings value takes precedence if it
    # is non-zero.
    _target_cash_utilization: float = 0.0
    # P3-15: optional RL tuner (UCB1 bandit over ensemble-weight triples).
    rl_tuner: "RLTunerSettings" = None  # type: ignore[assignment]
    # P3-17: idle-cash deployer (extra BUYs between day-bundles when
    # realised profits free up cash).
    idle_cash_deploy: "IdleCashDeploySettings" = None  # type: ignore[assignment]
    # P3-18: "static" (use `weights` from YAML) or "optimized" (load
    # configs/optimized_weights.json produced by scripts/optimize_weights.py
    # — Monte-Carlo + Markowitz-on-IC estimates from the trade journals).
    weights_mode: str = "static"
    optimized_weights_path: str = "configs/optimized_weights.json"
    # --- Strategy improvements (2026-10-29) — all opt-in / fail-soft ---
    # Multi-horizon momentum blend {d5, d10, d20}. None = disabled (mom
    # sub-score unchanged).
    mom_horizon_weights: Optional[dict[str, float]] = None
    # Minimum holding period in sim days before a SELL/COVER is allowed
    # (stop-loss + settlement-guard exits are exempt). 0 disables.
    min_hold_days: int = 3
    # Universe-relative entry timing gate (top-N + >= own rolling median).
    entry_timing: "EntryTimingSettings" = None  # type: ignore[assignment]
    # Exponential decay half-life (sim days) for news sentiment. 0 disables.
    news_half_life_days: float = 0.0
    # Veto BUYs where rank and mom strongly disagree in sign.
    disagreement_gate: "DisagreementGateSettings" = None  # type: ignore[assignment]
    # Replace momentum with mean reversion in risk_off regimes.
    mean_reversion_in_risk_off: bool = False
    # Only BUY tickers that newly entered the top-3 today.
    fresh_entry_only: bool = False
    # P4-02: re-entry cooldown — after SELLing a ticker, block BUY re-entry
    # for this many sim days unless its fresh score >= reentry_score_override.
    # Data 2026-10-30 (server orderbook): trades held <=2 sim days LOST
    # -3.0L over 169 trades (deployer/day-pipeline whipsaw); >2d holds
    # earned +4.7L. Attack the churn at ENTRY; exits stay untouched.
    reentry_cooldown_days: int = 3
    reentry_score_override: float = 0.90
    # P3-22: per-ticker RL bandit — penalise chronic losers (score multiplier
    # toward `floor`), mildly reward consistent winners (capped at `cap`).
    ticker_bandit: "TickerBanditSettings" = None  # type: ignore[assignment]
    # P3-23: take-profit — SELL a long whose unrealised gain from FIFO entry
    # reaches this fraction (0 disables). 0.10 = lock in +10% winners.
    take_profit_pct: float = 0.0
    # P3-25: partial take-profit — book `fraction` of a long once when gain
    # >= pct from FIFO entry; the rest rides the native score-decay exit.
    # Tournament: full TP cost -2.3L; 25%@8% costs only -30K (cheapest
    # profit-booking insurance that keeps the winners' tail).
    partial_take_profit: "PartialTakeProfitSettings" = None  # type: ignore[assignment]
    # P4-01: Short Alpha Engine — separate bearish pipeline; off by default
    # until diagnostics validate the negative tail (scripts/short_alpha_analysis.py).
    short_engine: "ShortEngineSettings" = None  # type: ignore[assignment]
    # P4-02: deployer regime cap — when True, the idle-cash deployer's
    # utilization target is capped by risk.gross_long_target[regime] so it
    # stops fighting the regime scaler in risk_off (churn source #2).
    deployer_regime_cap: bool = True
    # P3-24: trailing stop-loss — initial stop at entry*(1-initial_pct);
    # as price rises the stop trails the peak by trail_pct (never lowers).
    trailing_sl: "TrailingSLSettings" = None  # type: ignore[assignment]


@dataclass
class EntryTimingSettings:
    """Universe-relative entry timing: a BUY is allowed only if the ticker
    is in the day's top-`top_n` ensemble scores AND its score is >= its own
    rolling `lookback_days` median (uses available days once >=
    `min_history`; fewer -> gate no-ops)."""
    enabled: bool = False
    top_n: int = 2
    lookback_days: int = 20
    min_history: int = 3


@dataclass
class DisagreementGateSettings:
    """Veto BUY entries where sign(rank_score) != sign(mom_score) and both
    |scores| exceed `threshold` (strong disagreement)."""
    enabled: bool = False
    threshold: float = 0.1


@dataclass
class ShortEngineSettings:
    """P4-01: Short Alpha Engine — separate bearish pipeline (NOT a reversal
    of the long strategy). mode: off (control) | hedge (risk_off only) |
    independent (full long-short)."""
    enabled: bool = False
    mode: str = "off"                # off | hedge | independent
    weights: dict = field(default_factory=lambda: {
        "rank": 1.0 / 3.0, "news": 1.0 / 3.0, "mom": 1.0 / 3.0})
    entry_threshold: float = 0.85    # stricter than the long 0.75
    exit_threshold: float = 0.40     # bearish-score decay -> COVER
    adverse_stop_pct: float = 0.10   # hard stop: price up 10% vs short entry
    max_names: int = 3               # max simultaneous short names
    max_weight_short: float = 0.10   # per-name short cap (<= long cap)


@dataclass
class PartialTakeProfitSettings:
    """Rung-ladder partial take-profit: at each `pct` gain level (from FIFO
    entry), sell `frac` of the CURRENT holding. Whatever remains after the
    last rung rides the native score-decay exit. One rung per day per
    ticker; each rung fires once per position lifetime.

    `rungs` example: [{pct: 0.08, frac: 0.25}, {pct: 0.14, frac: 0.25},
    {pct: 0.25, frac: 0.25}] -> 25% off at +8%, 25% of the rest at +14%,
    25% of the rest at +25%, final ~42% rides. Legacy single-shot config
    (fraction+pct, no rungs) is treated as one rung."""
    enabled: bool = False
    fraction: float = 0.25
    pct: float = 0.08
    rungs: list[dict] = field(default_factory=list)


@dataclass
class TrailingSLSettings:
    """Stepped trailing stop-loss for longs (ratchets up, never lowers).

    Peak gain (from FIFO entry) maps to a stop level in steps:
      gain < breakeven_trigger_pct        -> stop = entry*(1 - initial_pct)
      gain >= breakeven_trigger_pct       -> stop = entry (breakeven)
      gain >= lock_start_pct              -> stop = entry*(1 + locked),
        locked = first_lock_pct + step_pct*floor((gain-lock_start_pct)/step_pct)
    e.g. defaults: -10% initial; +3% gain -> SL at buy price; +5% -> buy+3%;
    +7% -> buy+5%; +9% -> buy+7%; ... (always 2% locked behind the peak).
    """
    enabled: bool = False
    initial_pct: float = 0.10
    breakeven_trigger_pct: float = 0.03
    lock_start_pct: float = 0.05
    first_lock_pct: float = 0.03
    step_pct: float = 0.02


@dataclass
class TickerBanditSettings:
    """Per-ticker reward/penalty bandit (see bot/portfolio/rl_tuner.py).

    multiplier = clamp(1 + strength * EMA(pnl)/EMA(|pnl|), floor, cap).
    Bounded both ways: losers get rotated away from, winners get at most a
    `cap` boost — no gambling on streaks."""
    enabled: bool = False
    ema_alpha: float = 0.35
    strength: float = 0.5
    floor: float = 0.5
    cap: float = 1.15
    min_obs: int = 2
    veto_pulls: int = 5    # hard-BUY-veto after this many realised trades
    veto_z: float = -0.55  # ...when edge z <= this (strongly negative)
    # ...AND avg realised P&L per trade <= -this (magnitude guard: spares
    # low-win-rate / big-winner names whose EMA z is negative but small).
    veto_avg_rupees: float = 5000.0
    day_decay: float = 0.90  # EMA decay per quiet sim day (self-healing)


@dataclass
class IdleCashDeploySettings:
    """Deploy cash freed by realised profits between day-bundles.

    After every reconcile, if utilisation < target - epsilon, place one
    extra BUY for the highest-conviction ticker sized to the gap.
    """
    enabled: bool = True
    epsilon: float = 0.05  # only deploy when util is >5pp below target


@dataclass
class RLTunerSettings:
    """UCB1 bandit over ensemble-weight triples.

    See bot/portfolio/rl_tuner.py for the algorithm.
    """
    enabled: bool = False
    arms: list[dict[str, float]] = field(default_factory=list)
    update_every_trades: int = 10
    exploration_c: float = 1.4
    warmup_trades: int = 20


@dataclass
class RiskSettings:
    """Pre-trade risk caps (fractions of equity unless noted).

    dd_soft/dd_hard: drawdown breakers (soft warns, hard -> SAFE_MODE).
    max_short_gross: total short notional cap; max_name_weight: long cap per
    name; max_sector_weight: per-sector cap; gross_long_target: per-regime
    total long exposure (M5).
    """
    stop_atr_mult: float = 2.5
    dd_soft: float = 0.08
    dd_hard: float = 0.15
    # P3-14: cash buffer reduced 5% -> 2% so we keep >80% of equity invested
    # most of the time. The buffer only needs to cover brokerage + DP + a
    # small float for rounding errors.
    cash_buffer: float = 0.02
    max_short_gross: float = 0.20
    max_name_weight: float = 0.15
    max_sector_weight: float = 0.30
    gross_long_target: dict[str, float] = field(default_factory=lambda: {"risk_on": 0.92, "neutral": 0.75, "risk_off": 0.40})
    # P3-14: pause opening new positions after N consecutive losing days;
    # resets on the first green day. Set to 0 to disable.
    max_consecutive_loss_pause: int = 0
    # P3-14: target cash utilisation as a fraction of equity (long market
    # value + cash = equity). When below target, the constructor inflates
    # gross exposure. When above, it deflates. Default 0.85 -> we aim for
    # 80-90% invested most of the time.
    target_cash_utilization: float = 0.85
    # P3-14: max fraction of available cash that a single BUY may consume
    # (after fees + buffer). Prevents the orderbook's INSUFFICIENT_CAPITAL
    # rejections where we try to spend 99% of cash on one trade.
    max_cash_per_trade: float = 0.90


@dataclass
class TaxSettings:
    """Cost model + Indian capital-gains tax knobs.

    brokerage_per_order (flat Rs per order) and dp_charge_per_sell are
    charged on EVERY simulated fill, so they gate trade frequency;
    stcg/ltcg rates drive the harvest engine's benefit math.
    """
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
    """Simulated settlement calendar: shorts must be closed before each
    settlement boundary (every interval_trading_days trading days)."""
    interval_trading_days: int = 10
    force_close_before_days: int = 2  # close short positions N trading days early


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
    """Container for the M1a + sector clustering + ensemble knobs.

    M5 regime detection (bot.models.regime.RegimeDetector) owns its own
    `RegimeConfig` defaults and is wired directly by bot.main.TradingBot —
    no regime section here.
    """

    event_tagger: EventTaggerSettings = field(default_factory=EventTaggerSettings)
    sector_cluster: SectorClusterSettings = field(default_factory=SectorClusterSettings)
    # Ensemble weights (sum need not be 1; we renormalise over the signals we have).
    weights: dict[str, float] = field(
        default_factory=lambda: {"rank": 0.45, "news": 0.35, "mom": 0.20}
    )
    # Volatility-scaled conviction: divide each ticker's final score by its
    # realized vol before ranking (vol<=0 -> skip scaling for that ticker).
    vol_scale_scores: bool = False


@dataclass
class Settings:
    """Root settings object: identity/credentials + all section dataclasses.

    `dry_run=True` connects and scores but never submits orders.
    """
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
        # P3-15: rl_tuner sub-block needs to be parsed before the dataclass
        # filter strips it. Build it here and assign after.
        rl_block = strat.pop("rl_tuner", None)
        # NOTE: do NOT pop `weights` here — StrategySettings owns the canonical
        # field and we need to pass it through to the constructor below so
        # YAML values aren't silently replaced by the dataclass default.
        s.strategy = StrategySettings(**{k: v for k, v in strat.items() if k in StrategySettings.__dataclass_fields__})
        if rl_block:
            s.strategy.rl_tuner = RLTunerSettings(
                enabled=bool(rl_block.get("enabled", False)),
                arms=[{str(k): float(v) for k, v in arm.items()} for arm in (rl_block.get("arms") or [])],
                update_every_trades=int(rl_block.get("update_every_trades", 10) or 10),
                exploration_c=float(rl_block.get("exploration_c", 1.4) or 1.4),
                warmup_trades=int(rl_block.get("warmup_trades", 20) or 20),
            )
        # P3-17: idle-cash deployer sub-block (parsed before the dataclass
        # filter strips it, same as rl_tuner).
        icd_block = strat.pop("idle_cash_deploy", None)
        if icd_block is not None:
            s.strategy.idle_cash_deploy = IdleCashDeploySettings(
                enabled=bool(icd_block.get("enabled", True)),
                epsilon=float(icd_block.get("epsilon", 0.05) or 0.05),
            )
        # Strategy improvements: entry-timing + disagreement-gate sub-blocks
        # need pre-parsing before the dataclass filter strips them (same
        # pattern as rl_tuner / idle_cash_deploy above).
        et_block = strat.pop("entry_timing", None)
        if et_block is not None:
            s.strategy.entry_timing = EntryTimingSettings(
                enabled=bool(et_block.get("enabled", False)),
                top_n=int(et_block.get("top_n", 2) or 2),
                lookback_days=int(et_block.get("lookback_days", 20) or 20),
                min_history=int(et_block.get("min_history", 3) or 3),
            )
        dg_block = strat.pop("disagreement_gate", None)
        if dg_block is not None:
            s.strategy.disagreement_gate = DisagreementGateSettings(
                enabled=bool(dg_block.get("enabled", False)),
                threshold=float(dg_block.get("threshold", 0.1) or 0.1),
            )
        tb_block = strat.pop("ticker_bandit", None)
        if tb_block is not None:
            s.strategy.ticker_bandit = TickerBanditSettings(
                enabled=bool(tb_block.get("enabled", False)),
                ema_alpha=float(tb_block.get("ema_alpha", 0.35) or 0.35),
                strength=float(tb_block.get("strength", 0.5) or 0.5),
                floor=float(tb_block.get("floor", 0.5) or 0.5),
                cap=float(tb_block.get("cap", 1.15) or 1.15),
                min_obs=int(tb_block.get("min_obs", 2) or 2),
                veto_pulls=int(tb_block.get("veto_pulls", 5) or 5),
                veto_z=float(tb_block.get("veto_z", -0.55) or -0.55),
                veto_avg_rupees=float(tb_block.get("veto_avg_rupees", 5000.0) or 5000.0),
                day_decay=float(tb_block.get("day_decay", 0.90) or 0.90),
            )
        se_block = strat.pop("short_engine", None)
        if se_block is not None:
            s.strategy.short_engine = ShortEngineSettings(
                enabled=bool(se_block.get("enabled", False)),
                mode=str(se_block.get("mode", "off") or "off"),
                weights={k: float(v) for k, v in
                         (se_block.get("weights") or
                          {"rank": 1/3, "news": 1/3, "mom": 1/3}).items()},
                entry_threshold=float(se_block.get("entry_threshold", 0.85) or 0.85),
                exit_threshold=float(se_block.get("exit_threshold", 0.40) or 0.40),
                adverse_stop_pct=float(se_block.get("adverse_stop_pct", 0.10) or 0.10),
                max_names=int(se_block.get("max_names", 3) or 3),
                max_weight_short=float(se_block.get("max_weight_short", 0.10) or 0.10),
            )
        ptp_block = strat.pop("partial_take_profit", None)
        if ptp_block is not None:
            s.strategy.partial_take_profit = PartialTakeProfitSettings(
                enabled=bool(ptp_block.get("enabled", False)),
                fraction=float(ptp_block.get("fraction", 0.25) or 0.25),
                pct=float(ptp_block.get("pct", 0.08) or 0.08),
                rungs=[{"pct": float(r.get("pct", 0.08)),
                        "frac": float(r.get("frac", 0.25))}
                       for r in (ptp_block.get("rungs") or [])],
            )
        ts_block = strat.pop("trailing_sl", None)
        if ts_block is not None:
            s.strategy.trailing_sl = TrailingSLSettings(
                enabled=bool(ts_block.get("enabled", False)),
                initial_pct=float(ts_block.get("initial_pct", 0.10) or 0.10),
                breakeven_trigger_pct=float(ts_block.get("breakeven_trigger_pct", 0.03) or 0.03),
                lock_start_pct=float(ts_block.get("lock_start_pct", 0.05) or 0.05),
                first_lock_pct=float(ts_block.get("first_lock_pct", 0.03) or 0.03),
                step_pct=float(ts_block.get("step_pct", 0.02) or 0.02),
            )
        if "mom_horizon_weights" in (merged.get("strategy") or {}):
            mhw = merged["strategy"].get("mom_horizon_weights")
            s.strategy.mom_horizon_weights = (
                {str(k): float(v) for k, v in mhw.items()} if mhw else None
            )
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
        # `regime` key in YAML is now ignored — M5 owns its own RegimeConfig.
        if "regime" in sig:
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