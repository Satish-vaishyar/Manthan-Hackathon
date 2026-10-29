"""model6 — rule-based corporate event tagger with calibrated impacts.

    tagger = EventTagger.from_files("model6")              # loads configs + artifacts/calibration.json if present
    tags = tagger.tag_item(headline, description, "PLNX", tickers, alias_patterns)
    feats = tagger.score_day(news_items, tickers, aliases)  # {ticker: {"event_score": ..., ...}}

Every tag carries its evidence (the matched words), so each trade the bot makes on news can be explained.
Company names are masked to "the company" before matching; ticker symbols never reach the scores.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from text import clean, compile_aliases, mask, mentions

_PCT = re.compile(r"(\d+(?:\.\d+)?)\s?(?:%|per ?cent\b)", re.I)
_AMOUNT = re.compile(r"\$?\s?(\d+(?:\.\d+)?)\s?(billion|bn|million|mn|crore|lakh)\b", re.I)
PCT_EVENTS = {"EARNINGS_BEAT", "EARNINGS_MISS", "GUIDANCE_UP", "GUIDANCE_DOWN", "DEMAND_STRONG", "DEMAND_WEAK",
              "STOCK_UP", "STOCK_DOWN", "DIVIDEND"}
AMOUNT_EVENTS = {"CONTRACT_WIN", "CAPITAL_RAISE", "BUYBACK", "REGULATORY_ACTION", "INSIDER_SELL", "INSIDER_BUY"}


@dataclass
class EventTag:
    event: str
    polarity: int
    field: str                 # "headline" or "description"
    evidence: str              # matched text
    negated: bool = False
    hedged: bool = False
    magnitude: float = 1.0
    impact_bps: float = 0.0    # calibrated (or prior) impact of this event type
    weight: float = 1.0        # field weight x relevance weight x negation x hedging x magnitude
    contribution_bps: float = 0.0


@dataclass
class ItemTags:
    relevant: bool
    events: list[EventTag] = field(default_factory=list)


class EventTagger:
    def __init__(self, events_cfg: dict, cfg: dict, calibration: dict | None = None):
        self.cfg = cfg
        self.events = {}
        for name, e in events_cfg["events"].items():
            self.events[name] = {"polarity": int(e["polarity"]), "prior_bps": float(e["prior_bps"]),
                                 "patterns": [re.compile(p, re.I) for p in e["patterns"]]}
        neg = events_cfg["negation"]
        self.neg_words = [w.lower() for w in neg["words"]]
        self.neg_window, self.neg_factor = int(neg["window_words"]), float(neg["factor"])
        hed = events_cfg["hedging"]
        self.hedge = re.compile(r"\b(" + "|".join(hed["words"]) + r")\b", re.I)
        self.hedge_factor = float(hed["factor"])
        self.impacts = {k: v["prior_bps"] for k, v in self.events.items()}
        self.calibrated = False
        if calibration:
            for k, v in calibration.get("events", {}).items():
                if k in self.impacts:
                    self.impacts[k] = float(v["posterior_bps"])
            self.calibrated = True

    @classmethod
    def from_files(cls, root: str | Path, use_calibration: bool = True) -> "EventTagger":
        root = Path(root)
        ev = yaml.safe_load(open(root / "configs" / "events.yaml"))
        cfg = yaml.safe_load(open(root / "configs" / "model6.yaml"))
        cal_path = root / cfg["export"]["out_dir"] / "calibration.json"
        cal = json.load(open(cal_path)) if use_calibration and cal_path.exists() else None
        return cls(ev, cfg, cal)

    # ---- matching -----------------------------------------------------------------------------------
    def _negated(self, text: str, start: int) -> bool:
        before = re.findall(r"[a-z']+", text[:start].lower())[-self.neg_window:]
        window = " ".join(before)
        return any(re.search(rf"\b{re.escape(w)}\b", window) for w in self.neg_words)

    def _magnitude(self, event: str, text: str) -> float:
        cap = float(self.cfg["tagger"]["max_magnitude"])
        if event in PCT_EVENTS:
            pcts = [float(m.group(1)) for m in _PCT.finditer(text)]
            if pcts:
                return min(1.0 + min(max(pcts), 100.0) / 100.0, cap)
        if event in AMOUNT_EVENTS:
            m = _AMOUNT.search(text)
            if m:
                unit = m.group(2).lower()
                return min(1.5 if unit in ("billion", "bn") else 1.2, cap)
        return 1.0

    def tag_text(self, text: str, field_name: str) -> list[EventTag]:
        out = []
        hedged = bool(self.hedge.search(text))
        for name, e in self.events.items():
            for p in e["patterns"]:
                m = p.search(text)
                if m:
                    out.append(EventTag(event=name, polarity=e["polarity"], field=field_name,
                                        evidence=m.group(0)[:80], negated=self._negated(text, m.start()),
                                        hedged=hedged, magnitude=self._magnitude(name, text)))
                    break                       # one hit per event per field is enough
        return out

    def tag_item(self, headline: str, description: str, ticker: str, tickers: list[str],
                 alias_patterns: dict) -> ItemTags:
        h_raw, d_raw = clean(headline), clean(description)
        relevant = mentions(h_raw + " " + d_raw, ticker, alias_patterns)
        h, d = mask(h_raw, ticker, tickers, alias_patterns), mask(d_raw, ticker, tickers, alias_patterns)
        tc = self.cfg["tagger"]
        by_event: dict[str, EventTag] = {}
        for tag in self.tag_text(h, "headline") + self.tag_text(d, "description"):
            fw = tc["headline_weight"] if tag.field == "headline" else tc["description_weight"]
            rw = tc["relevant_weight"] if relevant else tc["irrelevant_weight"]
            tag.weight = (fw * rw * (self.neg_factor if tag.negated else 1.0)
                          * (self.hedge_factor if tag.hedged else 1.0) * tag.magnitude)
            tag.impact_bps = self.impacts[tag.event]
            tag.contribution_bps = tag.impact_bps * tag.weight
            prev = by_event.get(tag.event)
            if prev is None or abs(tag.weight) > abs(prev.weight):   # headline beats description
                by_event[tag.event] = tag
        return ItemTags(relevant=relevant, events=list(by_event.values()))

    # ---- per ticker-day features --------------------------------------------------------------------
    def score_day(self, news_items: list[dict], tickers: list[str], aliases: dict[str, list[str]],
                  with_evidence: bool = False) -> dict[str, dict]:
        pats = compile_aliases(aliases)
        sc = self.cfg["scoring"]
        top = sc["top_events"]
        out = {t: self._empty(top) for t in tickers}
        acc: dict[str, list] = {}
        for it in news_items:
            t = it.get("ticker")
            if t not in out:
                continue
            acc.setdefault(t, []).append(self.tag_item(it.get("headline", ""), it.get("description", ""),
                                                       t, tickers, pats))
        for t, items in acc.items():
            tags = [e for it in items for e in it.events]
            total = sum(e.contribution_bps for e in tags)
            pol = sum(e.polarity * e.weight for e in tags)
            f = out[t]
            f["event_score"] = math.tanh(total / sc["scale_bps"])
            f["rule_polarity"] = math.tanh(pol)
            f["n_events"] = len(tags)
            f["n_relevant_items"] = sum(it.relevant for it in items)
            f["max_abs_impact_bps"] = max((abs(e.contribution_bps) for e in tags), default=0.0)
            f["has_event"] = float(bool(tags))
            for e in tags:
                if e.event in top:
                    f[f"ev_{e.event}"] += e.weight
            if with_evidence:
                f["evidence"] = [f"{e.event}{' (negated)' if e.negated else ''}{' (hedged)' if e.hedged else ''}: "
                                 f"'{e.evidence}' {e.contribution_bps:+.0f}bps" for e in tags]
        return out

    @staticmethod
    def _empty(top: list[str]) -> dict:
        base = {"event_score": 0.0, "rule_polarity": 0.0, "n_events": 0, "n_relevant_items": 0,
                "max_abs_impact_bps": 0.0, "has_event": 0.0}
        base.update({f"ev_{e}": 0.0 for e in top})
        return base


def tags_to_dict(t: ItemTags) -> dict:
    return {"relevant": t.relevant, "events": [asdict(e) for e in t.events]}
