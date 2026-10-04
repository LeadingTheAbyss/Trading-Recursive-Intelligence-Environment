"""Step A and B: run one strategy on all 100 charts, compare with buy-and-hold and never-trade."""
import numpy as np
import pandas as pd
from market import make_prices
from strategy import MovingAverageStrategy, get_positions
from backtest import daily_profits, score

WINDOW = 20
COST = 0.001
N_CHARTS = 100


def run(prices, positions):
    s = score(daily_profits(prices, positions, COST))
    trades = int(np.abs(np.diff(np.concatenate([[0], positions]))).sum())
    s.update(time_invested=positions.mean(), trades=trades, cost_paid=trades * COST)
    return s


rows = []
for seed in range(N_CHARTS):
    p = make_prices(seed=seed)
    strat = run(p, get_positions(MovingAverageStrategy(WINDOW), p))
    hold = run(p, np.ones(len(p), dtype=int))
    rows.append({"seed": seed, **strat, "hold_return": hold["total_return"], "hold_sharpe": hold["sharpe"]})

df = pd.DataFrame(rows).set_index("seed")
df.to_csv("results_ab.csv")

print(f"Moving-average strategy (window {WINDOW}) on {N_CHARTS} charts")
print(df.describe().round(3).T[["mean", "50%", "min", "max"]])
print()
print("strategy made money on:", (df.total_return > 0).sum(), "charts")
print("buy-and-hold made money on:", (df.hold_return > 0).sum(), "charts")
print("strategy beat buy-and-hold on:", (df.total_return > df.hold_return).sum(), "charts")
print("never-trade return: exactly 0 on every chart")
