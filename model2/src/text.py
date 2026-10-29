"""Text cleaning, company masking and alias mining shared by training and the live bot.

Ticker symbols and company aliases are replaced by the neutral phrase "the company" before the text reaches the
model, so model2 learns *what the news says about a company*, never *which* company it is.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

COMPANY = "the company"
_WS = re.compile(r"\s+")
_CAP = re.compile(r"\b[A-Z][a-zA-Z]{3,}(?:\s[A-Z][a-zA-Z]{3,})?\b")


def clean(text: str) -> str:
    text = (text or "").replace("�", "'").replace("’", "'").replace("‘", "'")
    return _WS.sub(" ", text).strip()


def compile_aliases(aliases: dict[str, list[str]]) -> dict[str, re.Pattern | None]:
    out = {}
    for t, names in aliases.items():
        names = sorted({n for n in names if n and len(n) >= 3}, key=len, reverse=True)
        out[t] = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.I) if names else None
    return out


def mentions(text: str, ticker: str, patterns: dict[str, re.Pattern | None]) -> bool:
    p = patterns.get(ticker)
    return bool(re.search(rf"\b{re.escape(ticker)}\b", text)) or bool(p and p.search(text))


def mask(text: str, ticker: str, all_tickers: list[str], patterns: dict[str, re.Pattern | None]) -> str:
    """Replace this ticker's symbol/aliases with 'the company' and drop every other ticker symbol
    (the venue's masking sometimes swaps ordinary words for symbols, e.g. 'could CSTC' = 'could cost')."""
    p = patterns.get(ticker)
    if p:
        text = p.sub(COMPANY, text)
    text = re.sub(rf"\b{re.escape(ticker)}\b", COMPANY, text)
    others = [t for t in all_tickers if t != ticker]
    if others:
        text = re.sub(r"\b(" + "|".join(map(re.escape, others)) + r")\b", "", text)
    return _WS.sub(" ", text).strip()


def mine_aliases(news, min_count: int = 3, min_share: float = 0.8, top: int = 5) -> dict[str, list[str]]:
    """Capitalised words/bigrams that appear (almost) only in one ticker's news are likely its masked brand names."""
    per: dict[str, collections.Counter] = {}
    total: collections.Counter = collections.Counter()
    for t, g in news.groupby("ticker"):
        c = collections.Counter()
        for s in (g["headline"].fillna("") + " " + g["description"].fillna("")):
            c.update(set(_CAP.findall(clean(s))))
        per[t] = c
        total.update(c)
    out = {}
    for t, c in per.items():
        cands = [w for w, n in c.items() if n >= min_count and n / total[w] >= min_share]
        out[t] = sorted(cands, key=lambda w: -c[w])[:top]
    return out


_BRAND = re.compile(r"^[A-Z][a-z]+[A-Z][A-Za-z]+$")   # masked brand names look like DreamScape, RouteWave


def load_aliases(auto_path: str | Path, manual_path: str | Path) -> dict[str, list[str]]:
    """Hand-checked aliases plus only the auto-mined ones shaped like brand names (CamelCase).
    Other auto suggestions (places, people) stay in aliases_auto.json for humans to review."""
    auto = json.load(open(auto_path)) if Path(auto_path).exists() else {}
    manual = json.load(open(manual_path)) if Path(manual_path).exists() else {}
    keys = {k for k in list(auto) + list(manual) if not k.startswith("_")}
    return {k: sorted({a for a in auto.get(k, []) if _BRAND.match(a)} | set(manual.get(k, []))) for k in keys}
