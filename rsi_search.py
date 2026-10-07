"""Recursive self-improvement (RSI) of the strategy SEARCH, vs plain random search and plain evolution.

The strategy family is VolumeRuleStrategy (5 knobs) plus two anti-overtrading knobs: a minimum holding
period (once bought, stay in at least that many days) and a cooldown (after selling, wait that many days
before buying again). A "candidate" is a point in [0,1]^7.
Every search gets the same budget of evaluations. We pick the candidate with the best TRAIN Sharpe,
then look at how it does on a validation period and on a test period it never saw (after real Zerodha costs).

  random   : every candidate is drawn uniformly at random.
  evolve   : fixed recipe - mutate one of the best candidates so far, with a fixed step size.
  rsi      : three ways to propose candidates (random / mutate an elite / sample from the elite cloud).
             After every batch it checks which way produced good candidates and shifts its budget toward
             it, and it re-tunes its own mutation step size. The search recipe itself keeps changing.
"""
import numpy as np
import pandas as pd
from data_nse import load_nse
from backtest import daily_profits, score
from costs import trade_costs

CAPITAL = 100_000
BUDGET, BATCH, N_SEEDS = 200, 20, 20
DIM = 7
N_STOCKS = 200
COST = trade_costs(CAPITAL, "zerodha")
close, volume = load_nse(top=N_STOCKS)
P, V = close.to_numpy(), volume.to_numpy()
T = len(close)
TRAIN_END, VAL_END = int(T * 0.5), int(T * 0.75)   # 50% train / 25% validation / 25% test

# cache of indicators that depend on a single knob value
_mom, _vol, _hv = {}, {}, {}


def _cached(cache, key, make):
    if key not in cache:
        cache[key] = make()
    return cache[key]


def decode(x):
    """[0,1]^7 -> the five VolumeRuleStrategy knobs + min_hold and cooldown (days)."""
    return dict(mom_days=int(round(2 + x[0] * 58)), mom_thresh=-0.02 + x[1] * 0.12,
                vol_days=int(round(5 + x[2] * 55)), vol_thresh=0.005 + x[3] * 0.035,
                volume_pct=int(round(20 + x[4] * 75)),
                min_hold=int(round(x[5] * 60)), cooldown=int(round(x[6] * 30)))


def apply_rules(signal, min_hold, cooldown):
    """Turn the raw daily signal (T x stocks, True/False) into positions that respect min-hold and cooldown.
    Only looks at today's and past state, so no peeking."""
    pos = np.zeros(signal.shape, dtype=int)
    held = np.zeros(signal.shape[1], dtype=bool)
    age = np.zeros(signal.shape[1], dtype=int)         # days held so far
    since_sell = np.full(signal.shape[1], 10**6)       # days since last sell
    for t in range(len(signal)):
        s = signal[t]
        sell = held & (age >= min_hold) & ~s
        buy = ~held & s & (since_sell >= cooldown)
        age = np.where(held, age + 1, 0)
        since_sell = np.where(held, 0, since_sell + 1)
        held = (held & ~sell) | buy
        age = np.where(buy, 1, age)
        since_sell = np.where(sell, 0, since_sell)
        pos[t] = held
    return pos


def portfolio_profits(pos, cost=COST):
    """Same maths as backtest.daily_profits, for all stocks at once; returns the equal-weight daily profit."""
    buy_cost, sell_cost = cost
    price_change = P[1:] / P[:-1] - 1
    held = pos[:-1]
    previous = np.vstack([np.zeros((1, pos.shape[1]), dtype=int), pos[:-2]])
    bought, sold = np.maximum(held - previous, 0), np.maximum(previous - held, 0)
    return (held * price_change - bought * buy_cost - sold * sell_cost).mean(axis=1)


def evaluate(x):
    """Equal-weight portfolio over all stocks. Returns Sharpe on (train, val, test) and test total return."""
    k = decode(x)
    mom = _cached(_mom, k["mom_days"], lambda: close / close.shift(k["mom_days"]) - 1)
    vol = _cached(_vol, k["vol_days"], lambda: close.pct_change().rolling(k["vol_days"]).std(ddof=0))
    hv = _cached(_hv, k["volume_pct"], lambda: volume.rolling(100).quantile(k["volume_pct"] / 100))
    hold = (mom > k["mom_thresh"]) & (vol < k["vol_thresh"]) & (volume > hv)
    hold.iloc[: max(100, k["mom_days"] + 1, k["vol_days"] + 1) - 1] = False
    pos = apply_rules(hold.to_numpy(), k["min_hold"], k["cooldown"])
    profits = portfolio_profits(pos)
    # profits[t] is the gain on day t+1; split by time (positions only used past data, so this is clean)
    tr, va, te = profits[:TRAIN_END], profits[TRAIN_END:VAL_END], profits[VAL_END:]
    return score(tr)["sharpe"], score(va)["sharpe"], score(te)["sharpe"], score(te)["total_return"]


def search(method, seed):
    rng = np.random.default_rng(seed)
    X, R = [], []                        # archive: candidates and their (train, val, test, test_return)
    sigma = 0.15                         # mutation step size (rsi tunes it, evolve keeps it fixed)
    share = np.array([1.0, 1.0, 1.0])    # rsi: how much budget each proposal method gets
    best_curve = []
    for _ in range(BUDGET // BATCH):
        n_prev = len(X)
        fit = np.array([r[0] for r in R]) if R else np.array([])
        order = np.argsort(-fit)
        if method == "random" or n_prev == 0:
            src = np.zeros(BATCH, dtype=int)
        elif method == "evolve":
            src = np.ones(BATCH, dtype=int)
        else:
            src = rng.choice(3, size=BATCH, p=share / share.sum())
        batch = []
        for s in src:
            if s == 0:
                x = rng.random(DIM)
            elif s == 1:
                parent = X[rng.choice(order[:5])]
                x = np.clip(parent + rng.normal(0, sigma, DIM), 0, 1)
            else:
                elite = np.array([X[i] for i in order[:10]])
                x = np.clip(rng.normal(elite.mean(0), np.maximum(elite.std(0), 0.03)), 0, 1)
            batch.append(x)
        results = [evaluate(x) for x in batch]
        if method == "rsi" and n_prev > 0:
            # was this proposal method any good? "good" = beat the top quarter of everything seen before
            bar = np.quantile(fit, 0.75)
            for m in range(3):
                mine = [r[0] for r, s in zip(results, src) if s == m]
                if mine:
                    hit = np.mean(np.array(mine) > bar)
                    share[m] = 0.7 * share[m] + 0.3 * (hit + 0.05)   # smoothed hit rate, never zero
            mut_hit = np.mean([r[0] > bar for r, s in zip(results, src) if s == 1] or [0])
            sigma = float(np.clip(sigma * (1.25 if mut_hit > 0.2 else 0.8), 0.02, 0.4))   # 1/5 rule
        X += batch
        R += results
        best_curve.append(max(r[0] for r in R))
    best = int(np.argmax([r[0] for r in R]))
    return R[best], best_curve


if __name__ == "__main__":
    print(f"{close.shape[1]} stocks, {T} days. train to {close.index[TRAIN_END].date()}, "
          f"val to {close.index[VAL_END].date()}, test after.  Budget {BUDGET} evals, {N_SEEDS} seeds.\n")
    # yardstick: equal-weight buy and hold on the same three periods
    hold_profits = np.mean([daily_profits(P[:, i], np.ones(T, dtype=int), COST) for i in range(P.shape[1])], axis=0)
    hold = [score(hold_profits[a:b])["sharpe"] for a, b in [(0, TRAIN_END), (TRAIN_END, VAL_END), (VAL_END, None)]]
    hold_test_ret = score(hold_profits[VAL_END:])["total_return"]
    rows, curves = [], {}
    for method in ["random", "evolve", "rsi"]:
        out = [search(method, s) for s in range(N_SEEDS)]
        curves[method] = np.mean([c for _, c in out], axis=0)
        for (tr, va, te, ter), _ in out:
            rows.append({"method": method, "train_sharpe": tr, "val_sharpe": va, "test_sharpe": te, "test_return": ter})
    df = pd.DataFrame(rows)
    df.to_csv("results_rsi.csv", index=False)
    summary = df.groupby("method", sort=False).mean().round(3)
    summary["test_beats_hold"] = df.assign(b=df.test_return > hold_test_ret).groupby("method", sort=False).b.sum()
    print("Best-on-train candidate of each search, averaged over seeds")
    print(summary)
    print(f"\nbuy and hold (equal weight): Sharpe train {hold[0]:.3f}, val {hold[1]:.3f}, test {hold[2]:.3f}; "
          f"test return {hold_test_ret*100:.1f}%")
    print("\nbest train Sharpe found so far, by number of evaluations used")
    print(pd.DataFrame({m: c for m, c in curves.items()}, index=range(BATCH, BUDGET + 1, BATCH)).round(3))
