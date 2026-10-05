"""Baseline on REAL NSE prices (INR) with REAL Zerodha / Groww delivery costs.
Same moving-average rule as before, one position of Rs 1,00,000 per stock."""
import numpy as np
import pandas as pd
from data_nse import load_nse
from strategy import MovingAverageStrategy, get_positions
from backtest import daily_profits, score
from costs import trade_costs

CAPITAL = 100_000
close, volume = load_nse()
print(f"{close.shape[1]} NSE stocks, {close.shape[0]} days, {close.index[0].date()} to {close.index[-1].date()}\n")

rows = []
for broker in ["zerodha", "groww"]:
    cost = trade_costs(CAPITAL, broker)
    for name in close.columns:
        p = close[name].to_numpy()
        pos = get_positions(MovingAverageStrategy(20), p)
        s = score(daily_profits(p, pos, cost))
        h = score(daily_profits(p, np.ones(len(p), dtype=int), cost))
        rows.append({"broker": broker, "stock": name, "ma_return": s["total_return"], "hold_return": h["total_return"],
                     "trades": int(np.abs(np.diff(pos)).sum()), "invested": pos.mean()})
df = pd.DataFrame(rows)
df.to_csv("results_real.csv", index=False)
for broker, g in df.groupby("broker"):
    print(f"{broker}: round trip cost {sum(trade_costs(CAPITAL, broker))*100:.3f}%")
    print(f"  moving average : avg {g.ma_return.mean()*100:6.1f}%  profit on Rs 1 lakh: Rs {g.ma_return.mean()*CAPITAL:,.0f}  made money on {(g.ma_return>0).sum()} of {len(g)}")
    print(f"  buy and hold   : avg {g.hold_return.mean()*100:6.1f}%  profit on Rs 1 lakh: Rs {g.hold_return.mean()*CAPITAL:,.0f}  made money on {(g.hold_return>0).sum()} of {len(g)}")
    print(f"  trades per stock {g.trades.mean():.0f}, days invested {g.invested.mean()*100:.0f}%, MA beat hold on {(g.ma_return>g.hold_return).sum()} of {len(g)}\n")
