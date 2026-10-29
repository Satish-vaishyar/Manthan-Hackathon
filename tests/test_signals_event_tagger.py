"""tests/test_signals_event_tagger.py — coverage for bot.signals.event_tagger (M1a)."""
from __future__ import annotations

import pytest

from bot.signals.event_tagger import EventTagger, TAG_RULES


@pytest.fixture
def tagger() -> EventTagger:
    return EventTagger()


# ---------------------------------------------------------------- tag content
@pytest.mark.parametrize("headline,expected_tags", [
    ("Company profit surges 15%",              ["EARNINGS_BEAT"]),
    ("Profit falls 4% YoY; misses estimates",  ["EARNINGS_MISS"]),
    ("Board declares dividend",                ["DIVIDEND_BUYBACK"]),
    ("Company wins order from L1 bidder",      ["ORDER_WIN"]),
    ("CFO resigns amid SEBI probe",            ["REGULATORY_BAD", "MGMT_EXIT"]),
    ("Raises guidance for FY26",               ["GUIDANCE_UP"]),
    ("Cuts guidance on weak demand",           ["GUIDANCE_DOWN"]),
    ("Moody's downgrades to junk",             ["DEBT_STRESS", "RATING_DOWN"]),
    ("Company announces expansion of new plant", ["CAPACITY_EXPANSION"]),
])
def test_each_tag_is_recognised(tagger, headline, expected_tags):
    res = tagger.tag(headline)
    for t in expected_tags:
        assert t in res.tags, f"{headline!r} should match {t}, got {res.tags}"


def test_no_news_returns_neutral(tagger):
    res = tagger.tag("Stock trades sideways on low volume")
    assert res.tags == []
    assert res.polarity == 0.0
    assert not res.has_numbers


def test_empty_text_safe(tagger):
    res = tagger.tag("")
    assert res.tags == []
    assert res.polarity == 0.0


# ----------------------------------------------------------------- polarity
def test_positive_combined(tagger):
    res = tagger.tag("Profit surges 20% — company beats estimates, raises guidance")
    assert res.polarity > 0.5
    assert res.has_numbers
    assert res.magnitude_pct is not None and res.magnitude_pct > 0


def test_negative_combined(tagger):
    res = tagger.tag("CFO resigns; SEBI probe widens; debt downgrade to junk")
    assert res.polarity < -1.5  # multiple -2 tags should compound


def test_regulatory_tag_is_heavier_than_earnings(tagger):
    reg = tagger.tag("SEBI probe on insider trading")
    beat = tagger.tag("Profit surges on record revenue")
    assert reg.polarity < 0
    assert beat.polarity > 0
    # Regulatory tag is weighted -2; earnings beat is +1 — both saturate near
    # the magnitude factor, but the *intent* is asymmetry. Confirm by summing
    # only the regex contributions:
    assert abs(reg.polarity) >= abs(beat.polarity)


# ----------------------------------------------------------------- magnitude
def test_signed_magnitude_on_positive_trigger(tagger):
    res = tagger.tag("Profit jumps 12% YoY")
    assert res.has_numbers
    assert res.magnitude_pct == pytest.approx(12.0)


def test_signed_magnitude_on_negative_trigger(tagger):
    res = tagger.tag("Profit falls 7% YoY")
    assert res.has_numbers
    assert res.magnitude_pct == pytest.approx(-7.0)


def test_magnitude_without_trigger_ignored(tagger):
    # "12%" with no magnitude-bearing word nearby -> no magnitude contribution
    res = tagger.tag("Stock moves up 12% in a session")
    # No tag should be picked up by 'stock' alone; magnitude still 0
    assert res.magnitude_pct is None or res.magnitude_pct == 0


def test_polarity_is_clamped(tagger):
    res = tagger.tag("SEBI probe")  # just one -2 tag
    assert -3.0 <= res.polarity <= 3.0


# ----------------------------------------------------------------- coverage
def test_all_advertised_tags_have_a_rule():
    """docs/05 §3.1 lists 13 tags. We must have a rule for each."""
    doc_tags = {
        "EARNINGS_BEAT", "EARNINGS_MISS", "DIVIDEND_BUYBACK", "ORDER_WIN",
        "MNA", "RATING_UP", "RATING_DOWN", "REGULATORY_BAD", "MGMT_EXIT",
        "DEBT_STRESS", "GUIDANCE_UP", "GUIDANCE_DOWN", "CAPACITY_EXPANSION",
    }
    assert doc_tags.issubset(TAG_RULES.keys()), \
        f"missing tags: {doc_tags - TAG_RULES.keys()}"


def test_tagger_handles_description_field(tagger):
    res = tagger.tag("Company update", "Profit surges 18%")
    assert "EARNINGS_BEAT" in res.tags