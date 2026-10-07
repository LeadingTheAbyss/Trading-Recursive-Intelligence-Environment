"""A strategy as a small, readable recipe ("genome") that can be mutated, stored, compared and backtested.

A genome says:
  conds     : 1 to 3 rules joined by AND. Each rule looks at one feature of a stock over a window and asks
              "is this stock in the top (or bottom) part of today's ranking?"  e.g. momentum(60d) rank >= 0.7
  regime    : optional market filter. Hold nothing unless the whole market is above its own N-day average.
  rebalance : positions only change every R days (this is what keeps trading costs under control).
  sizing    : "equal" weight per chosen stock, or "invvol" (calmer stocks get bigger weight).
  exposure  : the most of our money we ever put to work (a risk limit, 0.4 to 1.0).
The chosen stocks share the invested money (at most 5% each, so fewer than 20 picks leaves part in cash).

Everything is rank-based (a stock vs the other stocks that day), so thresholds mean the same thing for every
feature, and everything only looks backwards, so there is no peeking (checked in test_genome.py).
Money not invested sits in cash earning 0. Costs are the real Zerodha delivery costs, charged on every change
in a stock's weight.
"""
import json
import warnings
import numpy as np
import pandas as pd
from costs import trade_costs

FEATURES = ["mom", "vol", "volu", "trend", "dd"]
FEATURE_DOC = {"mom": "momentum", "vol": "volatility", "volu": "volume surge", "trend": "price vs average",
               "dd": "closeness to recent high"}
WINDOWS = [3, 5, 10, 20, 40, 60, 90, 120, 200]
REGIME_WINDOWS = [20, 40, 60, 90, 120, 200]
REBALANCE = [1, 5, 10, 21, 42]
SIZINGS = ["equal", "invvol"]
CAPITAL = 100_000
COST = trade_costs(CAPITAL, "zerodha")
TRAIN_FRAC, VAL_FRAC = 0.5, 0.75          # same 50% / 25% / 25% split as rsi_search.py
MAX_WEIGHT = 0.05                         # risk limit: never more than 5% of money in one stock
LAMBDA_DD, LAMBDA_TURN = 0.5, 0.02        # fitness = Sharpe - 0.5*|max drawdown| - 0.02*yearly turnover


# ---------- the recipe itself ----------

def random_genome(rng):
    n = rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
    feats = rng.choice(FEATURES, size=n, replace=False)
    return {"conds": [{"feat": str(f), "win": int(rng.choice(WINDOWS)), "sign": int(rng.choice([-1, 1])),
                       "q": round(float(rng.uniform(0.3, 0.9)), 2)} for f in feats],
            "regime": int(rng.choice(REGIME_WINDOWS)) if rng.random() < 0.4 else None,
            "rebalance": int(rng.choice(REBALANCE)), "sizing": str(rng.choice(SIZINGS)),
            "exposure": round(float(rng.uniform(0.5, 1.0)), 2)}


def key(g):
    return json.dumps(g, sort_keys=True)


def describe(g):
    parts = []
    for c in g["conds"]:
        side = f"top {round((1 - c['q']) * 100)}%" if c["sign"] > 0 else f"bottom {round((1 - c['q']) * 100)}%"
        parts.append(f"{FEATURE_DOC[c['feat']]}({c['win']}d) in {side}")
    s = " AND ".join(parts)
    if g["regime"]:
        s += f"; only when market > its {g['regime']}d average"
    return f"{s}; rebalance every {g['rebalance']}d, {g['sizing']} weights, max {g['exposure']:.0%} invested"


def niche(g):
    """What KIND of strategy this is (not how well it does). Used to keep a diverse set of elites."""
    return (g["rebalance"], g["sizing"], tuple(sorted(c["feat"] for c in g["conds"])), g["regime"] is not None)


def encode(g):
    """Fixed-length numbers describing a genome, for the dream's cheap model of 'recipe -> score'."""
    v = np.zeros(26)
    for i, f in enumerate(FEATURES):
        cs = [c for c in g["conds"] if c["feat"] == f]
        if cs:
            c = cs[0]
            v[4 * i:4 * i + 4] = [1, (c["sign"] + 1) / 2, np.log(c["win"]) / np.log(200), c["q"]]
    v[20] = g["regime"] is not None
    v[21] = np.log(g["regime"]) / np.log(200) if g["regime"] else 0
    v[22] = np.log(g["rebalance"]) / np.log(42)
    v[23] = g["sizing"] == "invvol"
    v[24] = g["exposure"]
    v[25] = len(g["conds"]) / 3
    return v


# ---------- turning a recipe into money ----------

def feature(name, win, close, volume):
    if name == "mom":
        return close / close.shift(win) - 1
    if name == "vol":
        return close.pct_change().rolling(win).std()
    if name == "volu":
        return volume / volume.rolling(win).mean()
    if name == "trend":
        return close / close.rolling(win).mean() - 1
    if name == "dd":
        return close / close.rolling(win).max() - 1
    raise ValueError(name)


class Market:
    """Prices + volumes, with a cache of the expensive indicator tables."""

    def __init__(self, close, volume):
        self.close, self.volume = close, volume
        self.P = close.to_numpy()
        self.T, self.N = self.P.shape
        self.r = self.P[1:] / self.P[:-1] - 1               # r[t] = return from day t to t+1
        self.tr_end, self.va_end = int(self.T * TRAIN_FRAC), int(self.T * VAL_FRAC)
        self._rank, self._gate, self._invvol = {}, {}, None

    def rank(self, name, win):
        k = (name, win)
        if k not in self._rank:
            self._rank[k] = feature(name, win, self.close, self.volume).rank(axis=1, pct=True).to_numpy(np.float32)
        return self._rank[k]

    def gate(self, win):
        """True on days when the whole market (equal-weight index of our stocks) is above its own average."""
        if win not in self._gate:
            level = np.concatenate([[1.0], np.cumprod(1 + self.r.mean(axis=1))])
            self._gate[win] = (level > pd.Series(level).rolling(win).mean().to_numpy())
        return self._gate[win]

    def invvol(self):
        if self._invvol is None:
            inv = 1 / feature("vol", 20, self.close, self.volume).to_numpy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")           # first rows have no volatility yet (all NaN)
                inv = inv / np.nanmean(inv, axis=1, keepdims=True)
            self._invvol = np.nan_to_num(np.clip(inv, 0, 3))
        return self._invvol

    def weights(self, g):
        """Fraction of our money in each stock after the close of each day. Shape (T, N)."""
        active = np.ones((self.T, self.N), dtype=bool)
        for c in g["conds"]:
            rk = self.rank(c["feat"], c["win"])
            active &= (rk >= c["q"]) if c["sign"] > 0 else (rk <= 1 - c["q"])
        base = active * self.invvol() if g["sizing"] == "invvol" else active.astype(float)
        total = base.sum(axis=1, keepdims=True)
        w = np.divide(base, total, out=np.zeros_like(base), where=total > 0)      # chosen stocks share the money
        w = np.minimum(w, MAX_WEIGHT) * g["exposure"]                             # no stock above 5%; the rest is cash
        if g["regime"]:
            w = w * self.gate(g["regime"])[:, None]
        rebalance_day = (np.arange(self.T) // g["rebalance"]) * g["rebalance"]
        return w[rebalance_day]                              # between rebalances, hold what we had

    def portfolio(self, W, cost=COST):
        """Daily profit of the whole portfolio (fraction of capital) and how much we traded each day."""
        buy_cost, sell_cost = cost
        held = W[:-1]
        previous = np.vstack([np.zeros((1, self.N)), W[:-2]])
        change = held - previous
        costs = (np.maximum(change, 0) * buy_cost + np.maximum(-change, 0) * sell_cost).sum(axis=1)
        return (held * self.r).sum(axis=1) - costs, np.abs(change).sum(axis=1)

    def buy_and_hold(self):
        return self.portfolio(np.full((self.T, self.N), 1 / self.N))

    # ---------- scoring ----------

    def metrics(self, profits, turnover):
        equity = np.cumprod(1 + profits)
        peak = np.maximum.accumulate(equity)
        std = profits.std()
        return {"sharpe": profits.mean() / std * np.sqrt(252) if std > 0 else 0.0,
                "ret": equity[-1] - 1, "maxdd": ((equity - peak) / peak).min(),
                "vol": std * np.sqrt(252), "turnover": turnover.sum() / (len(profits) / 252)}

    def splits(self):
        return {"train": slice(0, self.tr_end), "val": slice(self.tr_end, self.va_end), "test": slice(self.va_end, None)}

    def evaluate(self, g):
        """Backtest one recipe. Returns metrics on train / val / test, fitness, and the daily profits."""
        profits, turnover = self.portfolio(self.weights(g))
        out = {"profits": profits.astype(np.float32)}
        for name, sl in self.splits().items():
            m = self.metrics(profits[sl], turnover[sl])
            out[name] = m
            out["fit_" + name] = fitness(m)
        return out


def fitness(m):
    """Risk-adjusted score after costs: reward Sharpe, punish deep drawdowns and heavy trading."""
    return m["sharpe"] - LAMBDA_DD * abs(m["maxdd"]) - LAMBDA_TURN * m["turnover"]
