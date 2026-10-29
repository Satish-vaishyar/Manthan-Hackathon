"""bot.portfolio.tax_engine — STCG 15% / LTCG 10% above ₹1L/FY exemption.

Per docs/01 §8 + docs/06 §2:
    - STCG = 15% on gains where holding period <= 365 days
    - LTCG = 10% on gains where holding period > 365 days, above ₹1L/FY exemption
    - Set-off order: STCL -> STCG -> LTCG ; LTCL -> LTCG only
    - FY = 1 Apr -> 31 Mar
    - Sec 112A: LTCG on equity above ₹1L taxed at 10%
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date as _date
from typing import Iterable, Optional

from .ledger import RealisedGain


@dataclass
class FYTaxReport:
    """Per-financial-year tax summary (gross gains/losses, taxable after
    set-off + exemption, and final tax payable)."""
    fy_label: str
    stcg_gross: float = 0.0
    ltcg_gross: float = 0.0
    stcl: float = 0.0   # short-term capital losses (positive number = loss)
    ltcl: float = 0.0   # long-term capital losses
    stcg_taxable: float = 0.0
    ltcg_taxable: float = 0.0
    tax_payable: float = 0.0


@dataclass
class TaxEngineConfig:
    """Tax rates + boundaries (Indian equity: STCG 15%, LTCG 10% over ₹1L,
    ST/LT split at 365 days, FY starting 1 April)."""
    stcg_rate: float = 0.15
    ltcg_rate: float = 0.10
    ltcg_exemption: float = 100_000.0
    st_lt_boundary_days: int = 365
    fy_start_month: int = 4
    fy_start_day: int = 1


def fy_label(d: _date, start_month: int = 4, start_day: int = 1) -> str:
    """Indian FY: 1 Apr -> 31 Mar. Label like 'FY2025-26'."""
    if (d.month, d.day) >= (start_month, start_day):
        start = d.year
        end = d.year + 1
    else:
        start = d.year - 1
        end = d.year
    return f"FY{str(start)[-4:]}-{str(end)[-2:]}"


def is_short_term(buy: _date, sell: _date, boundary_days: int = 365) -> bool:
    """ST if holding period <= boundary_days. Use sell - buy (calendar days)."""
    return (sell - buy).days <= boundary_days


def compute_fy_tax(
    gains: Iterable[RealisedGain],
    config: Optional[TaxEngineConfig] = None,
) -> dict[str, FYTaxReport]:
    """Group gains by FY, apply ST/LT split, apply set-off, compute tax.

    Set-off order (Sec 70-71):
        1. STCL  -> set off against STCG first
        2. Remaining STCL -> set off against LTCG
        3. LTCL  -> set off against LTCG only
        4. Apply ₹1L LTCG exemption to remaining LTCG
        5. Tax STCG @ stcg_rate, LTCG @ ltcg_rate
    """
    config = config or TaxEngineConfig()
    fy_data: dict[str, dict[str, float]] = defaultdict(
        lambda: {"stcg": 0.0, "ltcg": 0.0, "stcl": 0.0, "ltcl": 0.0}
    )

    for g in gains:
        st = is_short_term(g.buy_date, g.sell_date, config.st_lt_boundary_days)
        net = g.gross_pnl - g.fees
        fy = g.fy_label
        if st:
            if net >= 0:
                fy_data[fy]["stcg"] += net
            else:
                fy_data[fy]["stcl"] += -net  # store as positive
        else:
            if net >= 0:
                fy_data[fy]["ltcg"] += net
            else:
                fy_data[fy]["ltcl"] += -net

    reports: dict[str, FYTaxReport] = {}
    for fy, d in fy_data.items():
        stcg, ltcg, stcl, ltcl = d["stcg"], d["ltcg"], d["stcl"], d["ltcl"]

        # Set-off
        stcl_used_on_stcg = min(stcl, stcg)
        stcg_net = stcg - stcl_used_on_stcg
        stcl_remaining = stcl - stcl_used_on_stcg

        stcl_used_on_ltcg = min(stcl_remaining, ltcg)
        ltcg_after_stcl = ltcg - stcl_used_on_ltcg
        stcl_final = stcl_remaining - stcl_used_on_ltcg

        ltcl_used_on_ltcg = min(ltcl, ltcg_after_stcl)
        ltcg_net = ltcg_after_stcl - ltcl_used_on_ltcg

        # Exemption
        ltcg_taxable = max(0.0, ltcg_net - config.ltcg_exemption)

        tax = stcg_net * config.stcg_rate + ltcg_taxable * config.ltcg_rate

        reports[fy] = FYTaxReport(
            fy_label=fy,
            stcg_gross=stcg, ltcg_gross=ltcg,
            stcl=stcl, ltcl=ltcl,
            stcg_taxable=stcg_net, ltcg_taxable=ltcg_taxable,
            tax_payable=tax,
        )

    return reports


def total_tax(gains: Iterable[RealisedGain], config: Optional[TaxEngineConfig] = None) -> float:
    """Convenience: sum of tax_payable across all FYs."""
    return sum(r.tax_payable for r in compute_fy_tax(gains, config).values())


# ---------------------------------------------------------------------------
# Harvesting helpers — docs/06 §2.3
# ---------------------------------------------------------------------------
@dataclass
class HarvestOpportunity:
    """One loss-harvest candidate lot, with the estimated tax saving net of
    round-trip costs (sorted by saving, best first)."""
    ticker: str
    lot_index: int
    qty: int
    buy_date: _date
    buy_price: float
    current_price: float
    unrealised_loss: float
    estimated_tax_saving: float
    days_to_lt: int  # negative = already LT, 0+ = days remaining
    holding_days: int


def find_harvest_candidates(
    lots_by_ticker: dict[str, list],
    last_close: dict[str, float],
    today: _date,
    config: Optional[TaxEngineConfig] = None,
    *,
    harvest_window_days: int = 10,
    min_tax_saving: float = 70.0,
    stcg_rate: float = 0.15,
) -> list[HarvestOpportunity]:
    """Identify lots with unrealised losses worth harvesting.

    Only considers lots whose loss, if realised now, would offset gains at the
    higher STCG rate (or LTCG if already LT). `min_tax_saving` filters out
    trivial harvests where the round-trip fees would exceed the benefit.
    """
    config = config or TaxEngineConfig()
    out: list[HarvestOpportunity] = []
    for ticker, lots in lots_by_ticker.items():
        px = last_close.get(ticker)
        if px is None:
            continue
        for i, lot in enumerate(lots):
            if lot.qty <= 0:
                continue
            pnl = (px - lot.price) * lot.qty
            if pnl >= 0:
                continue
            days = (today - lot.buy_date).days
            rate = stcg_rate if is_short_term(lot.buy_date, today, config.st_lt_boundary_days) else config.ltcg_rate
            saving = -pnl * rate  # loss × rate
            # Subtract approx round-trip costs: ₹20 brokerage per order
            # (buy-back) + ₹15.34 DP charge on the sell = ₹35.34 per lot.
            saving -= 35.34
            if saving < min_tax_saving:
                continue
            # Only harvest if we're inside the window
            if days > harvest_window_days and is_short_term(lot.buy_date, today, config.st_lt_boundary_days):
                # ST loss that's been held past the window — still harvestable
                pass
            out.append(HarvestOpportunity(
                ticker=ticker,
                lot_index=i,
                qty=lot.qty,
                buy_date=lot.buy_date,
                buy_price=lot.price,
                current_price=px,
                unrealised_loss=-pnl,
                estimated_tax_saving=saving,
                days_to_lt=max(0, config.st_lt_boundary_days + 1 - days) if days <= config.st_lt_boundary_days else 0,
                holding_days=days,
            ))
    out.sort(key=lambda o: o.estimated_tax_saving, reverse=True)
    return out