"""tests/test_ledger.py — coverage for docs/11 §1 ledger cases."""
from __future__ import annotations

from datetime import date

import pytest

from bot.portfolio.ledger import Ledger


def _led(cash: float = 1_000_000.0) -> Ledger:
    return Ledger(cash=cash, initial_cash=cash)


# ----------------------------------------------------------------- BUY
def test_buy_long_creates_lot_and_deducts_cash():
    L = _led()
    L.apply_buy("AAA", 100, 100.0, date(2024, 5, 1), brokerage=20.0, dp_charge=0.0)
    assert L.holding_qty("AAA") == 100
    # Cash: -(100*100 + 20)
    assert L.cash == pytest.approx(1_000_000 - 10_020)


def test_buy_covers_short_first_then_longs():
    L = _led(cash=500_000.0)
    # Open a short at 200
    L.apply_short("AAA", 50, 200.0, date(2024, 5, 1), brokerage=20.0, dp_charge=15.34)
    cash_after_short = L.cash
    assert L.shorts["AAA"].qty == 50

    # BUY 80 @ 150 -> covers 50, longs 30
    L.apply_buy("AAA", 80, 150.0, date(2024, 5, 10), brokerage=20.0, dp_charge=15.34)
    assert L.shorts == {}  # short closed
    assert L.holding_qty("AAA") == 30
    # Cover realised = (200 - 150) * 50 = 2500
    assert L.realised["AAA"][0].gross_pnl == pytest.approx(2500.0)


# ----------------------------------------------------------------- SELL FIFO
def test_sell_fifo_partial_lots():
    L = _led()
    L.apply_buy("AAA", 50, 100.0, date(2024, 5, 1), brokerage=20.0, dp_charge=0.0)
    L.apply_buy("AAA", 50, 120.0, date(2024, 5, 2), brokerage=20.0, dp_charge=0.0)
    # Sell 70 @ 130 -> first 50 from lot1, next 20 from lot2
    L.apply_sell("AAA", 70, 130.0, date(2024, 5, 10), brokerage=20.0, dp_charge=15.34)
    # Lot1 realised: (130-100)*50 = 1500; Lot2: (130-120)*20 = 200; total = 1700
    pnls = sorted([g.gross_pnl for g in L.realised["AAA"]])
    assert pnls == pytest.approx([200.0, 1500.0])
    # Remaining: 30 shares from lot2
    assert L.holding_qty("AAA") == 30
    assert L.lots["AAA"][0].price == 120.0
    assert L.lots["AAA"][0].qty == 30


def test_sell_more_than_holding_raises():
    L = _led()
    L.apply_buy("AAA", 10, 100.0, date(2024, 5, 1), brokerage=20.0, dp_charge=0.0)
    with pytest.raises(ValueError):
        L.apply_sell("AAA", 20, 100.0, date(2024, 5, 2), brokerage=20.0, dp_charge=15.34)


# ----------------------------------------------------------------- SHORT / COVER
def test_short_locks_margin_and_cover_releases():
    L = _led(cash=1_000_000.0)
    L.apply_short("AAA", 100, 200.0, date(2024, 5, 1), brokerage=20.0, dp_charge=15.34)
    # cash += 100*200 - 20 - 15.34 = 19_964.66
    assert L.cash == pytest.approx(1_000_000 + 19_964.66)
    assert L.shorts["AAA"].margin_locked == pytest.approx(20_000.0)

    # Cover at 180 -> realised = (200-180)*100 = 2000
    L.apply_cover("AAA", 100, 180.0, date(2024, 5, 10), brokerage=20.0, dp_charge=15.34)
    assert L.shorts == {}
    assert L.realised["AAA"][0].gross_pnl == pytest.approx(2000.0)


def test_short_more_than_cash_raises():
    L = _led(cash=1000.0)
    with pytest.raises(ValueError):
        L.apply_short("AAA", 100, 200.0, date(2024, 5, 1), brokerage=20.0, dp_charge=0.0)


# ----------------------------------------------------------------- FY label
def test_fy_label_april_boundary():
    from bot.portfolio.ledger import Ledger
    assert Ledger._fy_label(date(2024, 3, 31)) == "FY2023-24"
    assert Ledger._fy_label(date(2024, 4, 1)) == "FY2024-25"
    assert Ledger._fy_label(date(2025, 1, 15)) == "FY2024-25"


# ----------------------------------------------------------------- Settlement P&L
def test_apply_buy_cover_short_settles_pnl():
    """End-to-end: short at 200, BUY covers at 150 -> realised (entry-cover)*qty."""
    L = _led()
    L.apply_short("AAA", 50, 200.0, date(2024, 5, 1), brokerage=20.0, dp_charge=15.34)
    L.apply_buy("AAA", 50, 150.0, date(2024, 5, 10), brokerage=20.0, dp_charge=15.34)
    assert L.shorts == {}
    assert L.holding_qty("AAA") == 0
    assert L.realised["AAA"][0].gross_pnl == pytest.approx(2500.0)


def test_dp_charge_only_on_sells():
    L = _led()
    L.apply_buy("AAA", 100, 100.0, date(2024, 5, 1), brokerage=20.0, dp_charge=15.34)
    # DP not subtracted on BUY
    assert L.cash == pytest.approx(1_000_000 - 10_020)


def test_ledger_equity_and_holdings():
    L = _led()
    L.apply_buy("AAA", 100, 100.0, date(2024, 5, 1), brokerage=20.0, dp_charge=0.0)
    L.last_close["AAA"] = 110.0
    assert L.holding_qty("AAA") == 100
    assert L.long_market_value == pytest.approx(11_000.0)
    assert L.equity == pytest.approx(L.cash + 11_000.0)