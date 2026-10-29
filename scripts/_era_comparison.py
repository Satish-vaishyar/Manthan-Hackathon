"""Compare trade quality across eras: what changed vs the profitable period."""
import glob
import io
import json
import os
import urllib.request
from collections import defaultdict

tok = os.environ["BOT_ACCESS_TOKEN"]
req = urllib.request.Request("http://192.168.65.157:3002/api/v1/orders?limit=10000",
                             headers={"Authorization": "Bearer " + tok})
d = json.load(urllib.request.urlopen(req, timeout=30))
orders = d.get("orders", d if isinstance(d, list) else [])
orders = [o for o in orders if o.get("status") == "FILLED"]
orders.sort(key=lambda o: (o.get("simulated_date") or "", o.get("ticker") or ""))

# FIFO round trips globally (position may span eras; era stamped by SELL date)
lots = defaultdict(list)
trips = []  # (sell_date, ticker, qty, buy_px, sell_px, hold_days, pnl)
for o in orders:
    t, act = o.get("ticker"), (o.get("action") or "").upper()
    q, p = float(o.get("quantity") or 0), float(o.get("execution_price") or 0)
    dt = o.get("simulated_date") or ""
    if not t or q == 0 or p == 0:
        continue
    if act == "BUY":
        lots[t].append([q, p, dt])
    elif act == "SELL":
        rem = q
        while rem > 1e-9 and lots[t]:
            lq, lp, ldt = lots[t][0]
            take = min(lq, rem)
            hold = 0
            try:
                from datetime import date
                y1, m1, d1 = map(int, ldt[:10].split("-"))
                y2, m2, d2 = map(int, dt[:10].split("-"))
                hold = (date(y2, m2, d2) - date(y1, m1, d1)).days
            except Exception:
                pass
            trips.append((dt, t, take, lp, p, hold, take * (p - lp)))
            lq -= take
            rem -= take
            if lq <= 1e-9:
                lots[t].pop(0)
            else:
                lots[t][0][0] = lq

ERAS = [("early  (<=2026-05-31)", lambda d: d <= "2026-05-31"),
        ("recent (>2026-05-31)", lambda d: d > "2026-05-31")]

for name, pred in ERAS:
    sub = [tr for tr in trips if pred(tr[0])]
    if not sub:
        continue
    pnl = sum(tr[6] for tr in sub)
    wins = [tr for tr in sub if tr[6] > 0]
    losses = [tr for tr in sub if tr[6] <= 0]
    aw = sum(t[6] for t in wins) / len(wins) if wins else 0
    al = sum(t[6] for t in losses) / len(losses) if losses else 0
    ah_w = sum(t[5] for t in wins) / len(wins) if wins else 0
    ah_l = sum(t[5] for t in losses) / len(losses) if losses else 0
    print(f"\n== {name}: {len(sub)} round trips ==")
    print(f"   net P&L Rs {pnl:,.0f}   win rate {len(wins)/len(sub)*100:.1f}%")
    print(f"   avg win Rs {aw:,.0f} (hold {ah_w:.0f}d)   avg loss Rs {al:,.0f} (hold {ah_l:.0f}d)")
    # hold-band breakdown
    for lo, hi, lbl in [(0, 3, "0-3d"), (4, 7, "4-7d"), (8, 30, "8-30d"), (31, 9999, ">30d")]:
        band = [t for t in sub if lo <= t[5] <= hi]
        if band:
            bp = sum(t[6] for t in band)
            bw = sum(1 for t in band if t[6] > 0)
            print(f"     {lbl:>5}: n={len(band):>3}  pnl={bp:>10,.0f}  win%={bw/len(band)*100:.0f}")
    # per-ticker
    pert = defaultdict(float)
    for tr in sub:
        pert[tr[1]] += tr[6]
    worst = sorted(pert.items(), key=lambda kv: kv[1])[:4]
    best = sorted(pert.items(), key=lambda kv: -kv[1])[:3]
    print("     best:", ", ".join(f"{t} {p:+,.0f}" for t, p in best))
    print("     worst:", ", ".join(f"{t} {p:+,.0f}" for t, p in worst))

print(f"\ntotal round trips analysed: {len(trips)}")
