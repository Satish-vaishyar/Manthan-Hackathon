"""Unit tests for the event tagger:  python -m pytest tests -q   (from model6/)"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tagger import EventTagger  # noqa: E402
from text import compile_aliases  # noqa: E402

TICKERS = ["ALPHA", "BETA", "CSTC"]
PATS = compile_aliases({"ALPHA": ["AlphaWorks"], "BETA": [], "CSTC": []})


@pytest.fixture(scope="module")
def tagger() -> EventTagger:
    return EventTagger.from_files(ROOT, use_calibration=False)


def events(tagger, headline, description="", ticker="ALPHA"):
    return {e.event: e for e in tagger.tag_item(headline, description, ticker, TICKERS, PATS).events}


@pytest.mark.parametrize("headline,expected", [
    ("AlphaWorks Q3 net profit surges 24% YoY", "EARNINGS_BEAT"),
    ("AlphaWorks beats analyst estimates on strong cloud sales", "EARNINGS_BEAT"),
    ("AlphaWorks quarterly profit falls 30% as costs rise", "EARNINGS_MISS"),
    ("AlphaWorks raises full-year guidance", "GUIDANCE_UP"),
    ("AlphaWorks cuts outlook amid weak demand", "GUIDANCE_DOWN"),
    ("AlphaWorks declares interim dividend", "DIVIDEND"),
    ("AlphaWorks announces $2 billion share buyback", "BUYBACK"),
    ("AlphaWorks Awarded $437M Task Order to Support Army Mission", "CONTRACT_WIN"),
    ("L3Harris + AlphaWorks enter a strategic partnership", "PARTNERSHIP"),
    ("FDA approves AlphaWorks cancer drug", "APPROVAL"),
    ("Regulators open probe into AlphaWorks accounting", "REGULATORY_ACTION"),
    ("AlphaWorks CEO steps down effective immediately", "MGMT_EXIT"),
    ("AlphaWorks files for Chapter 11 bankruptcy", "DEBT_DISTRESS"),
    ("AlphaWorks raises $27 million in registered direct offering", "CAPITAL_RAISE"),
    ("AlphaWorks CEO Nets $254.6M from Strategic Stock Sale as Shares Surge", "INSIDER_SELL"),
    ("AlphaWorks shares surge after upbeat forecast", "STOCK_UP"),
    ("AlphaWorks upgraded to buy at Morgan", "ANALYST_UPGRADE"),
    ("AlphaWorks downgraded on valuation", "ANALYST_DOWNGRADE"),
    ("AlphaWorks hails 'momentum' as new films boost box office returns", "DEMAND_STRONG"),
    ("AlphaWorks could profit from the conflict. But here's why its stock is overpriced.", "VALUATION_CONCERN"),
])
def test_detects_event(tagger, headline, expected):
    assert expected in events(tagger, headline)


@pytest.mark.parametrize("headline", [
    "The best garden parasols for every budget, chosen by our expert",
    "NBA finals 2025 Game 7: Thunder beat Pacers to claim title",
    "Schubert: Piano Sonata in A major album review",
])
def test_ignores_unrelated_headlines(tagger, headline):
    assert events(tagger, headline) == {}


def test_negation_flips_and_softens(tagger):
    e = events(tagger, "AlphaWorks denies probe into its accounts")["REGULATORY_ACTION"]
    assert e.negated and e.contribution_bps > 0          # denial of bad news is mildly positive


def test_hedging_reduces_weight(tagger):
    firm = events(tagger, "AlphaWorks wins contract from navy")["CONTRACT_WIN"]
    soft = events(tagger, "AlphaWorks reportedly wins contract from navy")["CONTRACT_WIN"]
    assert soft.hedged and abs(soft.weight) < abs(firm.weight)


def test_percent_magnitude_scales_impact(tagger):
    small = events(tagger, "AlphaWorks profit rises 5%")["EARNINGS_BEAT"]
    big = events(tagger, "AlphaWorks profit rises 60%")["EARNINGS_BEAT"]
    assert big.magnitude > small.magnitude


def test_relevance_weights(tagger):
    rel = tagger.tag_item("AlphaWorks profit surges", "", "ALPHA", TICKERS, PATS)
    irr = tagger.tag_item("Rival profit surges", "", "ALPHA", TICKERS, PATS)
    assert rel.relevant and not irr.relevant
    assert rel.events[0].weight > irr.events[0].weight


def test_headline_beats_description(tagger):
    e = events(tagger, "AlphaWorks wins contract", "The company won a contract worth $1 billion")["CONTRACT_WIN"]
    assert e.field == "headline"


def test_no_ticker_symbol_reaches_matching(tagger):
    # 'could CSTC' is the venue's masking artefact for 'could cost'; other symbols are stripped before matching.
    tags = tagger.tag_item("Drivers who carried passengers could CSTC rideshare giant $1.5m", "", "ALPHA",
                           TICKERS, PATS)
    assert all("CSTC" not in e.evidence for e in tags.events)


def test_score_day_shapes(tagger):
    items = [{"ticker": "ALPHA", "headline": "AlphaWorks profit surges 40%", "description": ""},
             {"ticker": "BETA", "headline": "Beta files for bankruptcy", "description": ""},
             {"ticker": "ZZZ", "headline": "ignored: not in universe", "description": ""}]
    out = tagger.score_day(items, ["ALPHA", "BETA", "CSTC"], {"ALPHA": ["AlphaWorks"]}, with_evidence=True)
    assert set(out) == {"ALPHA", "BETA", "CSTC"}
    assert out["ALPHA"]["event_score"] > 0 > out["BETA"]["event_score"]
    assert out["CSTC"]["has_event"] == 0 and out["CSTC"]["event_score"] == 0
    assert out["ALPHA"]["evidence"]
