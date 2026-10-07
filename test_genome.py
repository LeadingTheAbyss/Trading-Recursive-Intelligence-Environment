"""Checks with known answers for genome.py. Run: py test_genome.py"""
import numpy as np
from data_nse import load_nse
from genome import Market, random_genome, key, COST

close, volume = load_nse(top=30)
close, volume = close.iloc[:800], volume.iloc[:800]
m = Market(close, volume)

# no peeking: rewrite all prices and volumes after day 500, weights up to day 500 must not change
future_close, future_volume = close.copy(), volume.copy()
future_close.iloc[501:] = future_close.iloc[501:] * np.random.default_rng(0).uniform(0.5, 2, future_close.iloc[501:].shape)
future_volume.iloc[501:] = 1.0
m2 = Market(future_close, future_volume)
rng = np.random.default_rng(1)
for _ in range(40):
    g = random_genome(rng)
    assert np.allclose(m.weights(g)[:501], m2.weights(g)[:501]), key(g)

# never invested -> exactly zero profit, no trades
g = random_genome(rng)
profits, turnover = m.portfolio(np.zeros((m.T, m.N)))
assert np.all(profits == 0) and np.all(turnover == 0)

# buy and hold, no costs, equals the average of all stocks' returns
profits, _ = m.portfolio(np.full((m.T, m.N), 1 / m.N), cost=(0.0, 0.0))
assert np.allclose(profits, m.r.mean(axis=1))

# weights are never negative, never more than the exposure limit in total or 5% in one stock, and costs are charged when trading
for _ in range(40):
    g = random_genome(rng)
    W = m.weights(g)
    assert W.min() >= 0 and W.sum(axis=1).max() <= g["exposure"] + 1e-9
    assert W.max() <= 0.05 * g["exposure"] + 1e-9
free, _ = m.portfolio(W, cost=(0.0, 0.0))
paid, _ = m.portfolio(W)
assert paid.sum() <= free.sum() + 1e-12

# rebalance every 21 days -> weights only change on those days
g = random_genome(rng)
g["rebalance"] = 21
W = m.weights(g)
changes = np.nonzero(np.abs(np.diff(W, axis=0)).sum(axis=1))[0] + 1
assert np.all(changes % 21 == 0)

print("all genome checks passed")
