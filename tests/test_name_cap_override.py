"""Constructor/validator per-name-cap agreement (RISK_NAME_CAP live bug)."""
from datetime import date

from bot.config import (APISettings, KafkaSettings, RiskSettings, Settings,
                        StrategySettings, TaxSettings, TimingSettings)
from bot.contracts import OrderRequest
from bot.execution.validator import OrderValidator
from bot.portfolio.ledger import Ledger
from bot.portfolio.risk import RiskManager
from bot.portfolio.settlement import SettlementCalendar


def _validator():
    s = Settings(
        api=APISettings(port=3002, timeout_s=1.5, rate_per_min=100),
        kafka=KafkaSettings(),
        timing=TimingSettings(),
        strategy=StrategySettings(min_trade_notional=50_000.0),
        risk=RiskSettings(max_name_weight=0.20, max_sector_weight=0.30,
                          max_short_gross=0.20, cash_buffer=0.02,
                          dd_soft=0.12, dd_hard=0.20),
        tax=TaxSettings(brokerage_per_order=20.0, dp_charge_per_sell=15.34),
    )
    risk = RiskManager(s.risk, SettlementCalendar())
    return OrderValidator(s, risk)


def _ledger():
    L = Ledger(cash=1_000_000.0, initial_cash=1_000_000.0)
    L.last_close["AAA"] = 100.0
    return L


_REQ = OrderRequest(ticker="AAA", action="BUY", quantity=3000, date=date(2024, 5, 1))


def test_static_cap_denies_big_buy_by_default():
    v = _validator()
    res = v.validate(_REQ, _ledger())
    assert not res.ok and res.code == "RISK_NAME_CAP"


def test_relaxed_cap_allows_same_buy():
    v = _validator()
    # 3 qualifying names -> relaxed cap = 0.95/3 = 0.3167 (31.67% > 30% trade)
    v.name_cap_override = 0.95 / 3
    res = v.validate(_REQ, _ledger())
    assert res.ok, res
