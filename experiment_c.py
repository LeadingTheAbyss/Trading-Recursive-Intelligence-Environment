"""Step C: the luck experiment.

For each fake chart: make N random strategies (random window), pick the best on the FIRST half,
then check how that same strategy does on the SECOND half (which it never saw).
Repeat for different N to see how "searching harder" changes the gap.
"""
import numpy as np
import pandas as pd
from market import make_prices
from strategy import MovingAverageStrategy, get_positions
from backtest import daily_profits

COST = 0.001
N_CHARTS = 100
WINDOWS = np.arange(2, 201)          # the whole space of strategies: window = 2..200
SEARCH_SIZES = [1, 10, 100, 1000]    # how many random strategies we try per chart


def sharpe(profits):
    return profits.mean() / profits.std() * np.sqrt(252) if profits.std() > 0 else 0.0


rng = np.random.default_rng(123)
rows = []
for seed in range(N_CHARTS):
    prices = make_prices(seed=seed)
    # daily profits of every possible strategy, computed once on the full chart
    all_profits = np.array([daily_profits(prices, get_positions(MovingAverageStrategy(w), prices), COST)
                            for w in WINDOWS])
    half = all_profits.shape[1] // 2
    train, test = all_profits[:, :half], all_profits[:, half:]
    train_sharpe = np.array([sharpe(x) for x in train])
    test_sharpe = np.array([sharpe(x) for x in test])

    for n in SEARCH_SIZES:
        picked = rng.integers(0, len(WINDOWS), size=n)           # n random strategies
        best = picked[np.argmax(train_sharpe[picked])]           # best on train
        rows.append({"seed": seed, "n_strategies": n,
                     "train_sharpe": train_sharpe[best], "test_sharpe": test_sharpe[best],
                     "test_return": np.prod(1 + test[best]) - 1,
                     "avg_strategy_test_sharpe": test_sharpe.mean()})

df = pd.DataFrame(rows)
df.to_csv("results_c.csv", index=False)

summary = df.groupby("n_strategies")[["train_sharpe", "test_sharpe", "test_return"]].mean().round(3)
summary["gap"] = (summary.train_sharpe - summary.test_sharpe).round(3)
summary["test_wins"] = df[df.test_return > 0].groupby("n_strategies").size()
print("Average over", N_CHARTS, "charts (Sharpe on first half = train, second half = test)")
print(summary)
print("\naverage test Sharpe of a random strategy (no picking):", round(df.avg_strategy_test_sharpe.mean(), 3))
