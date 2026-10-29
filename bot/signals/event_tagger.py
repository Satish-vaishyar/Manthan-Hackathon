"""bot.signals.event_tagger — M1a keyword + regex tagger (docs/05 §3.1).

Instant, explainable, and the fallback if model2 (FinBERT + calibration) fails
or isn't shipped yet. Pure CPU: a regex + a keyword table.

Output per headline:
    tags:       list[str]                e.g. ["EARNINGS_BEAT", "GUIDANCE_UP"]
    polarity:   float in [-1, +1]        signed sum of tag polarities + magnitude
    has_numbers: bool                    True iff a numeric magnitude was parsed

Caller aggregates per ticker (mean or max) and integrates with the model2
calibration head if model2 is available. The tagger itself never touches
prices — it operates on text only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

# Tag polarity table — docs/05 §3.1.  Polarity is signed; magnitude parses
# `(\d+(\.\d+)?)\s?%` near magnitude-bearing words.
TAG_RULES: dict[str, dict[str, object]] = {
    # name -> {"pol": ±1, "patterns": [regex, ...]}
    "EARNINGS_BEAT":     {"pol": +1.0, "patterns": [
        r"\bprofit\s+(?:surges|jumps|rises|soars|climbs)\b",
        r"\b(?:record|highest|best)\s+(?:revenue|profit|exports|earnings)\b",
        r"\bbeats?\s+estimates?\b",
        r"\bprofits?\s+(?:up|rise|rise|grow|surge)\b",
    ]},
    "EARNINGS_MISS":     {"pol": -1.0, "patterns": [
        r"\bprofit\s+(?:falls|plunges|declines|slumps|tanks|drops)\b",
        r"\bloss(?:es)?\s+widens?\b",
        r"\bmisses\s+estimates?\b",
        r"\bprofits?\s+(?:down|fall|drop|slide|decline)\b",
    ]},
    "DIVIDEND_BUYBACK":  {"pol": +1.0, "patterns": [
        r"\bdividend\b", r"\bbuyback\b", r"\bbonus\s+issue\b",
    ]},
    "ORDER_WIN":         {"pol": +1.0, "patterns": [
        r"\bwins?\s+order\b", r"\bbags?\s+contract\b",
        r"\bsecures?\s+deal\b", r"\bL1\s+bidder\b",
    ]},
    "MNA":               {"pol": 0.0,  "patterns": [  # sign depends on acquirer
        r"\bacquire(?:s|d)?\b", r"\bmerger\b", r"\bstake\s+sale\b",
        r"\btakeover\b", r"\b(?:de)?merger\b",
    ]},
    "RATING_UP":         {"pol": +1.0, "patterns": [
        r"\bupgrade(?:s|d)?\b", r"\boutlook\s+positive\b",
    ]},
    "RATING_DOWN":       {"pol": -1.0, "patterns": [
        r"\bdowngrade(?:s|d)?\b", r"\boutlook\s+negative\b",
    ]},
    "REGULATORY_BAD":    {"pol": -2.0, "patterns": [  # weight: heavier than beats
        r"\bprobe\b", r"\braid\b", r"\bSEBI\b", r"\bpenalty\b",
        r"\bfraud\b", r"\binvestigation\b", r"\bban\b",
        r"\bshow[\-\s]cause\b",
    ]},
    "MGMT_EXIT":         {"pol": -1.0, "patterns": [
        r"\bresigns?\b", r"\bsteps\s+down\b", r"\bCFO\s+quits?\b",
        r"\bCEO\s+(?:resigns?|quits?)\b",
    ]},
    "DEBT_STRESS":       {"pol": -2.0, "patterns": [
        r"\bdefault\b", r"\bdowngrades?\s+to\s+junk\b",
        r"\binsolvency\b", r"\bNCLT\b",
    ]},
    "GUIDANCE_UP":       {"pol": +1.0, "patterns": [r"\braises?\s+guidance\b"]},
    "GUIDANCE_DOWN":     {"pol": -1.0, "patterns": [r"\bcuts?\s+guidance\b"]},
    "CAPACITY_EXPANSION": {"pol": +1.0, "patterns": [
        r"\bexpansion\b", r"\bnew\s+plant\b", r"\bcapex\b",
    ]},
}

# Magnitude-bearing words (regex captures `(pct)\s?%` near one of these).
_MAGNITUDE_TRIGGERS = re.compile(
    r"(?:yoy|qoq|growth|surge|fall|jump|drop|rise|decline|profit|revenue)",
    re.IGNORECASE,
)
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s?%")


@dataclass
class TagResult:
    """Tagger output for one headline."""

    headline: str
    tags: list[str] = field(default_factory=list)
    polarity: float = 0.0           # signed in roughly [-3, +3]
    has_numbers: bool = False
    magnitude_pct: Optional[float] = None  # signed by surrounding trigger words


class EventTagger:
    """M1a — pure-Python tagger. Cheap to instantiate; instances are stateless."""

    def __init__(self, magnitude_scale: float = 0.05) -> None:
        self.magnitude_scale = magnitude_scale
        # Pre-compile every regex once at construction.
        self._compiled: dict[str, list[re.Pattern]] = {
            tag: [re.compile(p, re.IGNORECASE) for p in spec["patterns"]]
            for tag, spec in TAG_RULES.items()
        }
        self._polarity: dict[str, float] = {
            tag: float(spec["pol"]) for tag, spec in TAG_RULES.items()
        }

    # ------------------------------------------------------------------
    def tag(self, headline: str, description: str = "") -> TagResult:
        """Tag a single headline (+ optional description)."""
        text = f"{headline or ''} {description or ''}".strip()
        if not text:
            return TagResult(headline=headline or "")

        tags: list[str] = []
        polarity = 0.0
        for tag, patterns in self._compiled.items():
            for pat in patterns:
                if pat.search(text):
                    tags.append(tag)
                    polarity += self._polarity[tag]
                    break  # one match per tag is enough

        # Magnitude: scan for `X%` near a magnitude-bearing word.
        # Sign: if a positive trigger (surge / growth / rise / jump / profit /
        # revenue) is the nearest, +; if a negative trigger (fall / drop /
        # decline / slump / miss) is nearest, -; otherwise 0.
        # The `s?` lets each stem match both singular and plural forms without
        # fighting `\b` boundaries ("falls", "drops", "misses", etc.).
        _POS_TRIG = re.compile(
            r"\b(surges?|jumps?|rises?|growth|profits?|revenue)s?\b"
        )
        _NEG_TRIG = re.compile(
            r"\b(falls?|drops?|slumps?|declines?|misses?|downs?)\b"
        )
        mag_signed: Optional[float] = None
        has_num = False
        for m in _PCT_RE.finditer(text):
            has_num = True
            pct = float(m.group(1))
            # Look at the surrounding ±40 chars for a trigger word.
            lo = max(0, m.start() - 40)
            hi = min(len(text), m.end() + 40)
            ctx = text[lo:hi].lower()
            if not _MAGNITUDE_TRIGGERS.search(ctx):
                continue
            # Determine sign by closest trigger word to the percentage.
            pos_hits = list(_POS_TRIG.finditer(ctx))
            neg_hits = list(_NEG_TRIG.finditer(ctx))
            if not pos_hits and not neg_hits:
                continue
            pos_dist = min(abs(p.start() - (m.start() - lo)) for p in pos_hits) if pos_hits else 10**9
            neg_dist = min(abs(p.start() - (m.start() - lo)) for p in neg_hits) if neg_hits else 10**9
            if pos_dist <= neg_dist:
                mag_signed = (mag_signed or 0.0) + pct
            else:
                mag_signed = (mag_signed or 0.0) - pct
        # Fold magnitude into polarity (scaled).
        if mag_signed is not None and self.magnitude_scale > 0:
            polarity += max(-1.0, min(1.0, mag_signed * self.magnitude_scale))

        # Clamp polarity for sanity; the ensemble re-clamps to [-1, +1] anyway.
        polarity = max(-3.0, min(3.0, polarity))
        return TagResult(
            headline=headline or "",
            tags=tags,
            polarity=polarity,
            has_numbers=has_num,
            magnitude_pct=mag_signed,
        )

    def tag_many(self, headlines: Iterable[tuple[str, str]]) -> list[TagResult]:
        """Tag a list of (headline, description) tuples."""
        return [self(h, d) for h, d in headlines]


# Convenience call syntax ----------------------------------------------------
def __call__(self, headline: str, description: str = "") -> TagResult:  # type: ignore[no-redef]
    return self.tag(headline, description)

EventTagger.__call__ = __call__  # type: ignore[attr-defined]