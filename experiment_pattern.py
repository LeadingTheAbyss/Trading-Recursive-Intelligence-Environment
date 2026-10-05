"""How big is the hidden pattern, and can the moving-average strategy see it?"""
import numpy as np
import pandas as pd
from market import make_market
from strategy import MovingAverageStrategy, get_positions
from backtest import daily_profits, score

N_CHARTS = 100
N_DAYS = 1000
COST = 0.001

rows = []
for seed in range(N_CHARTS):
    prices, volume, signal = make_market(n_days=N_DAYS, seed=seed)
    ma = score(daily_profits(prices, get_positions(MovingAverageStrategy(20), prices), COST))
    oracle = score(daily_profits(prices, signal, COST))        # knows the signal, pays costs
    oracle_free = score(daily_profits(prices, signal, 0.0))    # knows the signal, no costs
    rows.append({"seed": seed, "fires": signal.sum(),
                 "ma_return": ma["total_return"],
                 "oracle_return": oracle["total_return"],
                 "oracle_free_return": oracle_free["total_return"]})

df = pd.DataFrame(rows)
df.to_csv("results_pattern.csv", index=False)
print("signal fires per chart (of", N_DAYS, "days):", df.fires.mean().round(1))
print("\naverage total return over", N_CHARTS, "charts")
print("  moving-average strategy (cannot see volume):", round(df.ma_return.mean() * 100, 1), "%")
print("  oracle, WITH costs   :", round(df.oracle_return.mean() * 100, 1), "%   made money on", (df.oracle_return > 0).sum(), "charts")
print("  oracle, NO costs     :", round(df.oracle_free_return.mean() * 100, 1), "%   made money on", (df.oracle_free_return > 0).sum(), "charts")
