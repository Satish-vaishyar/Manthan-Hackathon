"""tests/test_tax_engine.py — ST/LT boundary, exemption, set-off, FY, harvesting."""
from __future__ import annotations

from datetime import date

import pytest

from bot.portfolio.ledger import RealisedGain
from bot.portfolio.tax_engine import (
    TaxEngineConfig,
    compute_fy_tax,
    find_harvest_candidates,
    fy_label,
    is_short_term,
    total_tax,
)


def _gain(buy_d, sell_d, pnl, fees=0.0, ticker="AAA"):
    fy = fy_label(sell_d)
    return RealisedGain(
        ticker=ticker, qty=100,
        buy_date=buy_d, sell_date=sell_d,
        buy_price=100.0, sell_price=100.0 + pnl / 100,
        gross_pnl=pnl, fees=fees, fy_label=fy,
    )


# ----------------------------------------------------------------- ST vs LT
def test_short_term_boundary_365_vs_366():
    # Pick dates that produce exactly 365 and 366 calendar days (no leap-day)
    # 2025-01-02 -> 2026-01-02 = 365 days, -> 2026-01-03 = 366 days
    assert (date(2026, 1, 2) - date(2025, 1, 2)).days == 365
    assert is_short_term(date(2025, 1, 2), date(2026, 1, 2)) is True   # 365 days = ST
    assert is_short_term(date(2025, 1, 2), date(2026, 1, 3)) is False  # 366 days = LT


def test_fy_label_april():
    assert fy_label(date(2024, 3, 31)) == "FY2023-24"
    assert fy_label(date(2024, 4, 1)) == "FY2024-25"


# ----------------------------------------------------------------- Pure STCG tax
def test_stcg_only_tax():
    gains = [_gain(date(2024, 6, 1), date(2024, 6, 30), pnl=15_000.0)]
    r = compute_fy_tax(gains)
    assert len(r) == 1
    fy = list(r.values())[0]
    assert fy.stcg_taxable == pytest.approx(15_000.0)
    assert fy.ltcg_taxable == 0.0
    # 15% of 15k = 2250
    assert fy.tax_payable == pytest.approx(2_250.0)


# ----------------------------------------------------------------- LTCG exemption
def test_ltcg_exemption_1l():
    gains = [_gain(date(2023, 1, 1), date(2024, 6, 30), pnl=150_000.0)]
    r = compute_fy_tax(gains)
    fy = list(r.values())[0]
    # 150k - 100k exemption = 50k taxable; 10% = 5000
    assert fy.ltcg_taxable == pytest.approx(50_000.0)
    assert fy.tax_payable == pytest.approx(5_000.0)


def test_ltcg_under_exemption_zero_tax():
    gains = [_gain(date(2023, 1, 1), date(2024, 6, 30), pnl=80_000.0)]
    r = compute_fy_tax(gains)
    fy = list(r.values())[0]
    assert fy.ltcg_taxable == 0.0
    assert fy.tax_payable == 0.0


# ----------------------------------------------------------------- Set-off order
def test_setoff_stcl_first_against_stcg_then_ltcg():
    # STCL of 10k, STCG of 6k, LTCG of 50k
    gains = [
        _gain(date(2024, 6, 1), date(2024, 6, 30), pnl=-10_000.0),  # STCL
        _gain(date(2024, 6, 1), date(2024, 6, 30), pnl=6_000.0),    # STCG
        _gain(date(2023, 1, 1), date(2024, 6, 30), pnl=50_000.0),  # LTCG
    ]
    r = compute_fy_tax(gains)
    fy = list(r.values())[0]
    # STCL 10k -> 6k STCG (STCG=0, 4k remaining)
    # STCL remaining 4k -> LTCG (LTCG becomes 46k)
    # Exemption 100k -> LTCG taxable = 0
    # Tax = 0
    assert fy.stcg_taxable == 0.0
    assert fy.ltcg_taxable == 0.0
    assert fy.tax_payable == 0.0


def test_ltcl_only_offsets_ltcg():
    # LTCG 200k, LTCL 30k -> LTCG becomes 170k -> 70k taxable
    gains = [
        _gain(date(2023, 1, 1), date(2024, 6, 30), pnl=200_000.0),
        _gain(date(2022, 1, 1), date(2024, 6, 30), pnl=-30_000.0),
    ]
    r = compute_fy_tax(gains)
    fy = list(r.values())[0]
    assert fy.ltcg_taxable == pytest.approx(70_000.0)
    assert fy.tax_payable == pytest.approx(7_000.0)


# ----------------------------------------------------------------- FY split
def test_fy_split_groups_separately():
    g24 = _gain(date(2024, 1, 1), date(2024, 3, 30), pnl=10_000.0)  # FY2023-24
    g25 = _gain(date(2024, 5, 1), date(2024, 6, 30), pnl=20_000.0)   # FY2024-25
    r = compute_fy_tax([g24, g25])
    assert "FY2023-24" in r and "FY2024-25" in r
    assert r["FY2023-24"].tax_payable == pytest.approx(1_500.0)
    assert r["FY2024-25"].tax_payable == pytest.approx(3_000.0)


# ----------------------------------------------------------------- Harvesting
def test_harvest_candidates_finds_losing_lot():
    from bot.portfolio.ledger import Lot
    lots = [Lot(ticker="AAA", qty=100, price=200.0, buy_date=date(2024, 5, 1), fees=0.0)]
    cands = find_harvest_candidates(
        {"AAA": lots}, {"AAA": 150.0}, today=date(2024, 6, 1),
    )
    assert len(cands) == 1
    assert cands[0].unrealised_loss == pytest.approx(5_000.0)
    assert cands[0].estimated_tax_saving > 0


def test_harvest_skips_small_savings():
    from bot.portfolio.ledger import Lot
    lots = [Lot(ticker="AAA", qty=1, price=100.0, buy_date=date(2024, 5, 1), fees=0.0)]
    cands = find_harvest_candidates(
        {"AAA": lots}, {"AAA": 99.0}, today=date(2024, 6, 1),
        min_tax_saving=100.0,
    )
    assert cands == []


def test_total_tax_matches_sum_of_reports():
    gains = [
        _gain(date(2024, 6, 1), date(2024, 6, 30), pnl=15_000.0),
        _gain(date(2023, 1, 1), date(2024, 6, 30), pnl=150_000.0),
    ]
    r = compute_fy_tax(gains)
    assert total_tax(gains) == pytest.approx(sum(v.tax_payable for v in r.values()))