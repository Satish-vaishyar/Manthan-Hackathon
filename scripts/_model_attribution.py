"""Per-model P&L attribution: which signal (rank/news/mom) drove each trade
and whether it helped.

Method: for every FILLED BUY in the live orderbook, find the journal score
record for that ticker on the nearest sim_date <= order date with a BUY-ish
decision. Decompose the ensemble score into weighted contributions
(w_i * score_i) using the live optimized weights; the dominant contributor
"carries" the trade. FIFO-match SELLs to attribute realized P&L per round
trip to the entry's dominant signal. Also report each model's average
weighted contribution on winning vs losing trades (is it helping?).
"""
import glob
import io
import json
from collections import defaultdict

WEIGHTS = json.load(open("configs/optimized_weights.json"))["weights"]
MODELS = ["rank", "news", "mom"]

# ---- load journal scores: (ticker, sim_date) -> latest scores dict
scores_by = {}
for jf in glob.glob("runs/*/journal.jsonl"):
    try:
        for line in io.open(jf, encoding="utf-8"):
            r = json.loads(line)
            s = r.get("scores")
            t, d = r.get("ticker"), r.get("sim_date")
            if s and t and d:
                scores_by[(t, d)] = s
    except Exception:
        continue

dates_by_ticker = defaultdict(list)
for (t, d) in scores_by:
    dates_by_ticker[t].append(d)
for t in dates_by_ticker:
    dates_by_ticker[t] = sorted(set(dates_by_ticker[t]))


def entry_scores(ticker, date):
    """Scores from the closest journal day <= order date (fallback: nearest)."""
    ds = dates_by_ticker.get(ticker)
    if not ds:
        return None
    le = [d for d in ds if d <= date]
    key = le[-1] if le else ds[0]
    return scores_by.get((ticker, key))


# ---- FIFO round trips with entry attribution
ob = sorted(glob.glob("runs/orderbook_live_*.json"))[-1]
orders = json.load(io.open(ob, encoding="utf-8-sig"))
orders = orders.get("orders", orders if isinstance(orders, list) else [])
orders = [o for o in orders if o.get("status") == "FILLED"]
orders.sort(key=lambda o: o.get("simulated_date") or o.get("created_at") or "")

lots = defaultdict(list)  # ticker -> [[qty, price, dominant, contribs], ...]
# per (ticker, model): pnl, trips, wins
pnl = defaultdict(float)
trips = defaultdict(int)
wins = defaultdict(int)
# per model: contribution sums on wins vs losses
contrib_win = defaultdict(float)
contrib_loss = defaultdict(float)
n_win = defaultdict(int)
n_loss = defaultdict(int)
unattributed_qty = 0.0

for o in orders:
    t = o.get("ticker")
    act = (o.get("action") or "").upper()
    q = float(o.get("quantity") or 0)
    p = float(o.get("execution_price") or o.get("price") or 0)
    dt = (o.get("simulated_date") or o.get("created_at") or "")[:10]
    if not t or q == 0 or p == 0:
        continue
    if act == "BUY":
        s = entry_scores(t, dt)
        if s:
            contrib = {m: WEIGHTS.get(m, 0) * float(s.get(m, 0) or 0) for m in MODELS}
            dom = max(contrib, key=lambda m: abs(contrib[m]))
        else:
            contrib, dom = {m: 0.0 for m in MODELS}, "unattr"
            unattributed_qty += q
        lots[t].append([q, p, dom, contrib])
    elif act == "SELL":
        rem = q
        while rem > 1e-9 and lots[t]:
            lq, lp, dom, contrib = lots[t][0]
            take = min(lq, rem)
            r = take * (p - lp)
            key = (t, dom)
            pnl[key] += r
            trips[key] += 1
            if r > 0:
                wins[key] += 1
                for m in MODELS:
                    contrib_win[m] += contrib.get(m, 0)
                n_win[dom] += 1
            elif r < 0:
                for m in MODELS:
                    contrib_loss[m] += contrib.get(m, 0)
                n_loss[dom] += 1
            lq -= take
            rem -= take
            if lq <= 1e-9:
                lots[t].pop(0)
            else:
                lots[t][0][0] = lq

tickers = sorted({k[0] for k in trips})
print(f"weights used: {WEIGHTS}")
print(f"unattributed BUY qty (no journal scores): {unattributed_qty:,.0f}\n")

print("=== per-ticker x dominant-signal attribution (realized P&L, net of nothing) ===")
print(f'{"ticker":<8}{"signal":<9}{"realized":>13}{"trips":>7}{"win%":>7}')
grand = defaultdict(float)
gtrips = defaultdict(int)
gwins = defaultdict(int)
for t in tickers:
    for m in MODELS + ["unattr"]:
        k = (t, m)
        if trips.get(k):
            wr = wins[k] / trips[k] * 100
            print(f"{t:<8}{m:<9}{pnl[k]:>13,.0f}{trips[k]:>7}{wr:>6.0f}%")
            grand[m] += pnl[k]
            gtrips[m] += trips[k]
            gwins[m] += wins[k]

print("\n=== signal-level totals (who is carrying the P&L?) ===")
print(f'{"signal":<9}{"realized":>13}{"trips":>7}{"win%":>7}')
for m in MODELS + ["unattr"]:
    if gtrips.get(m):
        print(f"{m:<9}{grand[m]:>13,.0f}{gtrips[m]:>7}{gwins[m] / gtrips[m] * 100:>6.0f}%")

print("\n=== is each model helping? (avg weighted contribution on winning vs losing trips) ===")
tw = sum(n_win.values())
tl = sum(n_loss.values())
for m in MODELS:
    cw = contrib_win[m] / tw if tw else 0.0
    cl = contrib_loss[m] / tl if tl else 0.0
    verdict = "HELPING" if cw > cl else "HURTING" if cw < cl else "neutral"
    print(f"{m:<9} winners: {cw:+.4f}   losers: {cl:+.4f}   -> {verdict}")
