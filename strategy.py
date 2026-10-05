import numpy as np
import pandas as pd

VOLUME_LOOKBACK = 100  # "unusually high volume" = high compared with the last 100 days


class MovingAverageStrategy:
    """Hold 1 share if today's price is above the average of the last `window` days, else hold 0."""

    def __init__(self, window):
        self.window = window

    def decide(self, history, volume=None):
        """history = prices up to and including today. Returns 1 (hold) or 0 (don't hold)."""
        if len(history) < self.window:
            return 0  # not enough past data yet, stay out
        return int(history[-1] > history[-self.window:].mean())


class VolumeRuleStrategy:
    """Hold 1 share only when ALL three things are true today:
      1. momentum: the price rose more than `mom_thresh` over the last `mom_days` days
      2. calm: the std of daily returns over the last `vol_days` days is below `vol_thresh`
      3. busy: today's volume is above the `volume_pct`-th percentile of the last 100 days

    Five knobs: mom_days, mom_thresh, vol_days, vol_thresh, volume_pct.
    """

    def __init__(self, mom_days, mom_thresh, vol_days, vol_thresh, volume_pct):
        self.mom_days, self.mom_thresh = mom_days, mom_thresh
        self.vol_days, self.vol_thresh = vol_days, vol_thresh
        self.volume_pct = volume_pct

    def decide(self, history, volume):
        """history, volume = data up to and including today."""
        need = max(VOLUME_LOOKBACK, self.mom_days + 1, self.vol_days + 1)
        if len(history) < need:
            return 0
        momentum = history[-1] / history[-1 - self.mom_days] - 1
        returns = history[-self.vol_days - 1:][1:] / history[-self.vol_days - 1:][:-1] - 1
        calm = returns.std() < self.vol_thresh
        busy = volume[-1] > np.percentile(volume[-VOLUME_LOOKBACK:], self.volume_pct)
        return int(momentum > self.mom_thresh and calm and busy)

    def positions_fast(self, prices, volume):
        """Same answers as get_positions(), computed all at once (much faster).
        Every number only looks backwards in time, so there is still no peeking.
        test_backtest.py checks that this matches the slow day-by-day version."""
        p, v = pd.Series(prices), pd.Series(volume)
        momentum = p / p.shift(self.mom_days) - 1
        vol = p.pct_change().rolling(self.vol_days).std(ddof=0)
        high_volume = v.rolling(VOLUME_LOOKBACK).quantile(self.volume_pct / 100)
        hold = (momentum > self.mom_thresh) & (vol < self.vol_thresh) & (v > high_volume)
        hold &= pd.Series(np.arange(len(p)) >= max(VOLUME_LOOKBACK, self.mom_days + 1, self.vol_days + 1) - 1)
        return hold.to_numpy().astype(int)


def get_positions(strategy, prices, volume=None):
    """Ask the strategy for a decision at the close of every day.

    positions[t] = what we hold after the close of day t.
    The strategy is only handed data up to day t (prices[:t+1], volume[:t+1]),
    read-only, so it physically cannot see tomorrow.
    """
    if volume is None:
        volume = np.ones(len(prices))
    positions = np.zeros(len(prices), dtype=int)
    for t in range(len(prices)):
        history = prices[: t + 1].copy()
        history.flags.writeable = False
        vol_history = volume[: t + 1].copy()
        vol_history.flags.writeable = False
        positions[t] = strategy.decide(history, vol_history)
    return positions


class NeighborStrategy:
    """Hold if charts that looked like today's chart tended to go UP the next day.

    Every day: take the last `window` daily returns (the recent shape of the chart) and find
    the `k` most similar shapes earlier in the past. Look at what happened the day after each
    of them. Average those next-day returns, trusting closer matches more. If the average is
    above `threshold`, hold 1 share. Otherwise hold nothing.

    No pattern is built in. It only assumes "shapes that looked alike behave alike".
    With use_volume=True the shape also includes recent trading volume.
    Knobs: window, k, threshold.
    """

    def __init__(self, window=5, k=20, threshold=0.0, use_volume=False, min_history=100):
        self.window, self.k, self.threshold = window, k, threshold
        self.use_volume, self.min_history = use_volume, min_history

    def decide(self, history, volume):
        from numpy.lib.stride_tricks import sliding_window_view
        n, L = len(history), self.window
        if n < max(self.min_history, 2 * L + self.k):
            return 0
        r = history[1:] / history[:-1] - 1                      # r[i] = return into day i+1
        shapes = sliding_window_view(r, L) / r.std()            # one row per recent-shape ending at each day
        if self.use_volume:
            lv = np.log(volume[1:])
            lv = (lv - lv.mean()) / lv.std()
            shapes = np.hstack([shapes, sliding_window_view(lv, L)])
        today, past = shapes[-1], shapes[:-1]                   # past rows all have a known "next day"
        next_returns = r[L:]
        dist = np.linalg.norm(past - today, axis=1)
        nearest = np.argpartition(dist, self.k)[: self.k]
        weight = 1.0 / (dist[nearest] + 1e-6)                   # closer match -> more trust
        predicted = (weight * next_returns[nearest]).sum() / weight.sum()
        return int(predicted > self.threshold)
