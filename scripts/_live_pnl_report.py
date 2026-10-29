"""One-off live tradebook P&L report (server orderbook = ground truth)."""
import glob
import io
import json
from collections import defaultdict

f = sorted(glob.glob("runs/orderbook_live_*.json"))[-1]
d = json.load(io.open(f, encoding="utf-8-sig"))
orders = d.get("orders", d if isinstance(d, list) else [])
orders.sort(key=lambda o: o.get("simulated_date") or o.get("created_at") or "")

filled = [o for o in orders if o.get("status") == "FILLED"]
rejected = [o for o in orders if o.get("status") == "REJECTED"]

lots = defaultdict(list)
realized = defaultdict(float)
costs = defaultdict(float)
wins = defaultdict(int)
losses = defaultdict(int)
roundtrips = defaultdict(int)

for o in filled:
    t = o.get("ticker")
    act = (o.get("action") or "").upper()
    q = float(o.get("quantity") or 0)
    p = float(o.get("execution_price") or o.get("price") or 0)
    fee = float(o.get("brokerage") or 0) + float(o.get("taxes") or 0)
    costs[t] += fee
    if not t or q == 0 or p == 0:
        continue
    if act == "BUY":
        lots[t].append([q, p])
    elif act == "SELL":
        rem = q
        while rem > 1e-9 and lots[t]:
            lq, lp = lots[t][0]
            take = min(lq, rem)
            pnl = take * (p - lp) - fee * (take / q)
            realized[t] += pnl
            roundtrips[t] += 1
            if pnl > 0:
                wins[t] += 1
            elif pnl < 0:
                losses[t] += 1
            lq -= take
            rem -= take
            if lq <= 1e-9:
                lots[t].pop(0)
            else:
                lots[t][0][0] = lq

print("=== LIFETIME (server truth) ===")
print(f"file: {f}")
print(f"orders: {len(orders)}  filled: {len(filled)}  rejected: {len(rejected)}  "
      f"success: {len(filled) / len(orders) * 100:.1f}%")
tot_r = sum(realized.values())
tot_c = sum(costs.values())
print(f"realized P&L (net of allocated fees): Rs {tot_r:,.0f}")
print(f"total brokerage+taxes paid:           Rs {tot_c:,.0f}")
tw, tl = sum(wins.values()), sum(losses.values())
if tw + tl:
    print(f"round trips: {tw + tl}  wins: {tw}  losses: {tl}  win rate: {tw / (tw + tl) * 100:.1f}%")

hdr = f'{"ticker":<8}{"realized":>14}{"costs":>12}{"trips":>7}{"win%":>7}{"open_qty":>10}'
print("\n" + hdr)
for t in sorted(set(list(realized) + list(costs) + list(lots)),
                key=lambda x: -realized.get(x, 0)):
    tr = roundtrips.get(t, 0)
    w = wins.get(t, 0)
    open_q = sum(l[0] for l in lots.get(t, []))
    print(f"{t:<8}{realized.get(t, 0):>14,.0f}{costs.get(t, 0):>12,.0f}"
          f"{tr:>7}{(w / tr * 100 if tr else 0):>6.0f}%{open_q:>10,.0f}")

print("\n=== sample of rejected orders (last 5) ===")
for o in rejected[-5:]:
    print(o.get("simulated_date") or o.get("created_at"), o.get("ticker"),
          o.get("action"), o.get("quantity"),
          (o.get("reason") or o.get("message") or "")[:80])
