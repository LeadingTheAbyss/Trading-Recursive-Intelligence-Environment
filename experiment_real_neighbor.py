"""Shape-matching (NeighborStrategy) on REAL NSE stocks with real Zerodha costs."""
import numpy as np
import pandas as pd
from data_nse import load_nse
from strategy import NeighborStrategy, get_positions
from backtest import daily_profits
from costs import trade_costs

CAPITAL = 100_000
cost = trade_costs(CAPITAL, "zerodha")
close, volume = load_nse()
half = (len(close) - 1) // 2

configs = {                       # settings fixed in advance, not tuned on these results
    "prices only, threshold 0":        NeighborStrategy(window=5, k=20, threshold=0.0),
    "prices only, threshold 0.25%":    NeighborStrategy(window=5, k=20, threshold=0.0025),
    "prices+volume, threshold 0":      NeighborStrategy(window=5, k=20, threshold=0.0, use_volume=True),
    "prices+volume, threshold 0.25%":  NeighborStrategy(window=5, k=20, threshold=0.0025, use_volume=True),
}

def total(x):
    return np.prod(1 + x) - 1

rows = []
for name in close.columns:
    p, v = close[name].to_numpy(), volume[name].to_numpy()
    hold = daily_profits(p, np.ones(len(p), dtype=int), cost)
    rows.append({"config": "buy and hold", "stock": name, "full": total(hold), "second_half": total(hold[half:]),
                 "trades": 1, "invested": 1.0})
    for cname, strat in configs.items():
        pos = get_positions(strat, p, v)
        pr = daily_profits(p, pos, cost)
        rows.append({"config": cname, "stock": name, "full": total(pr), "second_half": total(pr[half:]),
                     "trades": int(np.abs(np.diff(pos)).sum()), "invested": pos.mean()})

df = pd.DataFrame(rows)
df.to_csv("results_real_neighbor.csv", index=False)
print(f"{close.shape[1]} stocks, {close.shape[0]} days; second half starts {close.index[half].date()}; round trip cost {sum(cost)*100:.3f}%\n")
hold_row = df[df.config == "buy and hold"].set_index("stock")
print(f"{'':32}{'avg full':>9}{'avg 2nd half':>14}{'win(full)':>11}{'win(2nd)':>10}{'trades':>8}{'invested':>10}{'beat hold(2nd)':>16}")
for cname, g in df.groupby("config", sort=False):
    g = g.set_index("stock")
    beat = (g.second_half > hold_row.second_half).sum()
    print(f"{cname:32}{g.full.mean()*100:8.1f}%{g.second_half.mean()*100:13.1f}%{(g.full>0).sum():8}/48{(g.second_half>0).sum():7}/48{g.trades.mean():8.0f}{g.invested.mean()*100:9.0f}%{beat:12}/48")
