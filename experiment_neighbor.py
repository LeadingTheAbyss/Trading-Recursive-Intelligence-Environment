"""Does the shape-matching strategy find the hidden pattern? (it was never told what the pattern is)"""
import sys
import numpy as np
from market import make_market
from strategy import NeighborStrategy, get_positions
from backtest import daily_profits, score

N_CHARTS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
COST = 0.001
configs = {"prices only": NeighborStrategy(window=5, k=20), "prices + volume": NeighborStrategy(window=5, k=20, use_volume=True)}

for edge in [0.5, 0.0]:
    res = {name: [] for name in configs}
    res.update({"oracle": [], "buy and hold": []})
    for seed in range(N_CHARTS):
        p, v, sig = make_market(1000, edge=edge, seed=seed)
        for name, strat in configs.items():
            res[name].append(score(daily_profits(p, get_positions(strat, p, v), COST))["total_return"])
        res["oracle"].append(score(daily_profits(p, sig, COST))["total_return"])
        res["buy and hold"].append(score(daily_profits(p, np.ones(len(p), dtype=int), COST))["total_return"])
    print(f"\npattern strength {edge}  ({N_CHARTS} charts, 1000 days, cost {COST})")
    for name, x in res.items():
        x = np.array(x)
        print(f"  {name:16} avg return {x.mean()*100:6.1f}%   made money on {(x > 0).sum()} of {N_CHARTS}")
