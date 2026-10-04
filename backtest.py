import numpy as np


def daily_profits(prices, positions, cost=0.001):
    """Daily return of the strategy, as a fraction (0.01 = +1%). Length = len(prices) - 1.

    Decide at the close of day t -> positions[t]. The gain/loss from that decision
    arrives on day t+1. So profit on day t+1 uses positions[t], never positions[t+1].
    Every change in position costs `cost` (0.001 = 0.1% of the price).
    """
    price_change = prices[1:] / prices[:-1] - 1          # change from day t to t+1
    held = positions[:-1]                                # what we held going into each day
    previous = np.concatenate([[0], positions[:-2]])     # what we held before that (start with nothing)
    trades = np.abs(held - previous)                     # 1 whenever we bought or sold
    return held * price_change - trades * cost


def score(profits, days_per_year=252):
    """Turn daily profits into three numbers."""
    equity = np.cumprod(1 + profits)                     # value of 1 unit of money over time
    total_return = equity[-1] - 1
    peak_so_far = np.maximum.accumulate(equity)
    max_drawdown = ((equity - peak_so_far) / peak_so_far).min()   # worst drop from a peak (<= 0)
    std = profits.std()
    sharpe = profits.mean() / std * np.sqrt(days_per_year) if std > 0 else 0.0
    return {"total_return": total_return, "max_drawdown": max_drawdown, "sharpe": sharpe}
