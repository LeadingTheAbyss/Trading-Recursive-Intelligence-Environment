"""Does dreaming over accumulated discoveries improve the strategy-SEARCH process itself?

Four searches, same 300 real backtests each, same first population, same 200-stock market, same costs:
  fixed  -> online -> replay -> dream      (each step adds one learning ingredient; see discovery.py)
The strategy each search deploys is chosen WITHOUT looking at the test period.
Opponent: equal-weight buy and hold of the same 200 stocks.

Run:  py experiment_discovery.py [n_seeds]
"""
import sys
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from data_nse import load_nse
from genome import Market, describe, fitness
from discovery import search, OPS, ROUNDS, CHILDREN

MODES = ["fixed", "online", "replay", "dream"]
N_STOCKS = 200
_market = None


def _init():
    global _market
    close, volume = load_nse(top=N_STOCKS)
    _market = Market(close, volume)


def _run(job):
    mode, seed = job
    r = search(mode, seed, _market)
    e = r["chosen"]
    return {"mode": mode, "seed": seed, **{f"{s}_{k}": v for s in ("train", "val", "test") for k, v in e[s].items()},
            "test_fitness": fitness(e["test"]), "val_fitness": e["fit_val"], "train_fitness": e["fit_train"],
            "niches": r["niches"], "evals": r["evals"], "diversity": r["diversity"],
            "success": r["success"], "genome": e["g"], "policy": r["policy"], "edges": r["edges"]}


def table(df, hold):
    g = df.groupby("mode", sort=False)
    out = pd.DataFrame({
        "train fit": g.train_fitness.mean(), "val Sharpe": g.val_sharpe.mean(),
        "TEST Sharpe": g.test_sharpe.mean(), "TEST return": g.test_ret.mean(), "TEST maxDD": g.test_maxdd.mean(),
        "TEST vol": g.test_vol.mean(), "turnover/yr": g.test_turnover.mean(), "TEST fitness": g.test_fitness.mean(),
        "niches": g.niches.mean(), "diversity": g.diversity.mean()})
    out["beat hold Sharpe"] = df.assign(b=df.test_sharpe > hold["sharpe"]).groupby("mode", sort=False).b.sum()
    out["beat hold fitness"] = df.assign(b=df.test_fitness > fitness(hold)).groupby("mode", sort=False).b.sum()
    return out.round(3)


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    _init()
    hold_p, hold_t = _market.buy_and_hold()
    hold = {n: _market.metrics(hold_p[sl], hold_t[sl]) for n, sl in _market.splits().items()}
    c = _market.close
    print(f"{_market.N} stocks, {_market.T} days ({c.index[0].date()} to {c.index[-1].date()}); train to "
          f"{c.index[_market.tr_end].date()}, val to {c.index[_market.va_end].date()}, test after.")
    print(f"{n_seeds} seeds x {len(MODES)} searches x {ROUNDS * CHILDREN} backtests\n")

    jobs = [(m, s) for s in range(n_seeds) for m in MODES]
    with ProcessPoolExecutor(max_workers=6, initializer=_init) as pool:
        rows = list(pool.map(_run, jobs, chunksize=1))
    df = pd.DataFrame(rows)
    df.drop(columns=["success", "genome", "policy", "edges"]).to_csv("results_discovery.csv", index=False)

    h = hold["test"]
    print("buy and hold (same 200 stocks, same costs):")
    for n, m in hold.items():
        print(f"  {n:5} Sharpe {m['sharpe']:5.2f}  return {m['ret']*100:6.1f}%  maxDD {m['maxdd']*100:6.1f}%  "
              f"vol {m['vol']*100:4.1f}%  fitness {fitness(m):5.2f}")
    print("\nStrategy each search DEPLOYED (chosen on train+validation only), averaged over seeds:")
    print(table(df, h).to_string())

    print("\nShare of new children that beat their parent, by round (is the SEARCH getting better?):")
    succ = pd.DataFrame({m: np.mean([r["success"] for r in rows if r["mode"] == m], axis=0) for m in MODES},
                        index=range(2, ROUNDS + 1)).round(3)
    succ.index.name = "round"
    print(succ.to_string())
    print("  average first half vs second half of rounds:")
    for m in MODES:
        s = succ[m].to_numpy()
        print(f"    {m:7} {s[:len(s)//2].mean():.3f} -> {s[len(s)//2:].mean():.3f}")

    d0 = next(r for r in rows if r["mode"] == "dream" and r["seed"] == 0)
    print("\nWhat the dream learned (seed 0): modifications it favours most, by kind of parent")
    names = ["1 rule", "1 rule + mkt filter", "2 rules", "2 rules + mkt filter", "3 rules", "3 rules + mkt filter"]
    for b, nm in enumerate(names):
        top = np.argsort(-d0["policy"][b])[:3]
        print(f"  {nm:22}", ", ".join(f"{OPS[o]} {d0['policy'][b][o]*100:.0f}%" for o in top))
    print("\nStrategy the dream deployed (seed 0):\n ", describe(d0["genome"]))
    json.dump({"genome": d0["genome"]}, open("best_dream_strategy.json", "w"), indent=1)
