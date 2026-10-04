import numpy as np


class MovingAverageStrategy:
    """Hold 1 share if today's price is above the average of the last `window` days, else hold 0."""

    def __init__(self, window):
        self.window = window

    def decide(self, history):
        """history = prices up to and including today. Returns 1 (hold) or 0 (don't hold)."""
        if len(history) < self.window:
            return 0  # not enough past data yet, stay out
        return int(history[-1] > history[-self.window:].mean())


def get_positions(strategy, prices):
    """Ask the strategy for a decision at the close of every day.

    positions[t] = what we hold after the close of day t.
    The strategy is only handed prices[:t+1], so it physically cannot see tomorrow.
    """
    positions = np.zeros(len(prices), dtype=int)
    for t in range(len(prices)):
        history = prices[: t + 1].copy()
        history.flags.writeable = False
        positions[t] = strategy.decide(history)
    return positions
