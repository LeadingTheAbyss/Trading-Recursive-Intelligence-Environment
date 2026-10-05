"""Checks with known answers. Run: py test_backtest.py"""
import numpy as np
from backtest import daily_profits, score
from market import make_market
from strategy import MovingAverageStrategy, VolumeRuleStrategy, get_positions

rising = 100 * 1.01 ** np.arange(50)       # +1% every day
flat = np.full(50, 100.0)

# never hold -> exactly zero profit
p = daily_profits(rising, np.zeros(50, dtype=int))
assert np.all(p == 0)

# always hold on a rising chart, no costs -> +1% per day
p = daily_profits(rising, np.ones(50, dtype=int), cost=0)
assert np.allclose(p, 0.01)

# trade every day on a flat chart -> only costs, so we lose money
alternating = np.arange(50) % 2
p = daily_profits(flat, alternating, cost=0.001)
assert score(p)["total_return"] < 0

# no peeking: changing FUTURE prices must not change today's decision
a = rising.copy()
b = rising.copy()
b[30:] = 1.0                                # rewrite the future after day 29
s = MovingAverageStrategy(5)
assert np.array_equal(get_positions(s, a)[:30], get_positions(s, b)[:30])

# the volume strategy: no peeking, and the fast version matches the slow day-by-day version
prices, volume, _ = make_market(n_days=400, seed=3)
v = VolumeRuleStrategy(mom_days=5, mom_thresh=0.0, vol_days=10, vol_thresh=0.01, volume_pct=90)
slow = get_positions(v, prices, volume)
assert slow.sum() > 0, "strategy never fires, test would prove nothing"
assert np.array_equal(slow, v.positions_fast(prices, volume))

p2, v2 = prices.copy(), volume.copy()
p2[300:] = 1.0
v2[300:] = 99.0                             # rewrite the future after day 299
assert np.array_equal(slow[:300], get_positions(v, p2, v2)[:300])

# the neighbor strategy: no peeking, and it learns a repeating pattern it was never told about
from strategy import NeighborStrategy

nb = NeighborStrategy(window=3, k=10)
cycle = np.tile([0.01, 0.02, -0.02, -0.01], 100)          # a chart that repeats every 4 days
cyc_prices = 100 * np.concatenate([[1], np.cumprod(1 + cycle)])
cyc_vol = np.ones(len(cyc_prices))
pos = get_positions(nb, cyc_prices, cyc_vol)
assert score(daily_profits(cyc_prices, pos, cost=0))["total_return"] > 1.0   # more than doubles

prices, volume, _ = make_market(n_days=300, seed=5)
nbv = NeighborStrategy(window=5, k=10, use_volume=True)
base = get_positions(nbv, prices, volume)
p2, v2 = prices.copy(), volume.copy()
p2[200:], v2[200:] = 1.0, 99.0
assert np.array_equal(base[:200], get_positions(nbv, p2, v2)[:200])

# different buy and sell costs: one buy + one sell on a flat chart costs exactly buy + sell
from costs import trade_costs
buy, sell = trade_costs(100_000, "zerodha")
pos = np.zeros(50, dtype=int); pos[10:20] = 1
p = daily_profits(flat, pos, cost=(buy, sell))
assert np.isclose(p.sum(), -(buy + sell))
assert 0.002 < buy + sell < 0.003            # about 0.24% round trip at Rs 1 lakh on Zerodha

print("all checks passed")
