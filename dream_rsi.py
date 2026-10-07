"""Dream-RSI: use the history of past (real) evaluations as a cheap replay simulator of the strategy market.

Plain RSI (rsi_search.py) learns which search trick works only by spending REAL evaluations on it.
Dream-RSI keeps every past evaluation, fits a cheap "dream model" of how strategies score from that history,
and then tests many possible search policies inside the dream (free), instead of in the real backtest (0.1 s each).

Every round:
  1. fit the dream model on all real history so far (a smooth curve through candidate -> train Sharpe,
     with an uncertainty estimate: a Gaussian-process style model).
  2. dream: run each candidate search policy (random / mutate an elite with step 0.05, 0.1, 0.2, 0.4 /
     sample the elite cloud) in the dream, and score it by how good the candidates it proposes look.
  3. deploy the best policy: it proposes a big pool, and only the top BATCH go to the real backtest.
  4. the real results are added to the history, so the dream model gets better. Repeat.

Same budget as the other searches: 200 REAL evaluations. Compare against results_rsi.csv.
Run:  py dream_rsi.py
"""
import numpy as np
import pandas as pd
from rsi_search import evaluate, DIM, BUDGET, BATCH, N_SEEDS, T, TRAIN_END, VAL_END, P, COST, close
from backtest import daily_profits, score

LENGTH, NOISE, BONUS = 0.35, 1e-2, 1.0      # dream model smoothness, noise, and how much it likes unexplored spots
DREAM_POOL, DEPLOY_POOL = 200, 1000
POLICIES = ["random", "mutate0.05", "mutate0.1", "mutate0.2", "mutate0.4", "cloud"]


class DreamModel:
    """Predicts train Sharpe of an unseen candidate from past real evaluations (kernel ridge / Gaussian process)."""

    def __init__(self, X, y):
        self.X, self.mu, self.sd = np.array(X), np.mean(y), np.std(y) + 1e-9
        K = self._k(self.X, self.X) + NOISE * np.eye(len(X))
        self.Kinv = np.linalg.inv(K)
        self.alpha = self.Kinv @ ((np.array(y) - self.mu) / self.sd)

    @staticmethod
    def _k(A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * LENGTH ** 2))

    def predict(self, C):
        """Returns (expected Sharpe, uncertainty) for each row of C."""
        Ks = self._k(C, self.X)
        mean = self.mu + self.sd * (Ks @ self.alpha)
        var = np.maximum(1 - np.einsum("ij,jk,ik->i", Ks, self.Kinv, Ks), 0)
        return mean, self.sd * np.sqrt(var)


def propose(policy, n, X, order, rng):
    """n candidates from one search policy, using the real history X (best first = order)."""
    if policy == "random":
        return rng.random((n, DIM))
    if policy.startswith("mutate"):
        parents = np.array([X[i] for i in order[:5]])[rng.integers(0, 5, n)]
        return np.clip(parents + rng.normal(0, float(policy[6:]), (n, DIM)), 0, 1)
    elite = np.array([X[i] for i in order[:10]])
    return np.clip(rng.normal(elite.mean(0), np.maximum(elite.std(0), 0.03), (n, DIM)), 0, 1)


def dream_search(seed):
    rng = np.random.default_rng(seed)
    X, R, curve, chosen = [], [], [], []
    for rnd in range(BUDGET // BATCH):
        if rnd == 0:
            batch = list(rng.random((BATCH, DIM)))                         # nothing to dream from yet
        else:
            y = [r[0] for r in R]
            order = np.argsort(-np.array(y))
            model = DreamModel(X, y)
            # DREAM: try every policy inside the model, score by the quality of what it proposes
            dream_score = {}
            for pol in POLICIES:
                mean, unc = model.predict(propose(pol, DREAM_POOL, X, order, rng))
                dream_score[pol] = np.sort(mean + BONUS * unc)[-BATCH:].mean()
            best_policy = max(dream_score, key=dream_score.get)
            chosen.append(best_policy)
            # DEPLOY: the winning policy fills a big pool, only the dream's top picks cost a real evaluation
            pool = propose(best_policy, DEPLOY_POOL, X, order, rng)
            mean, unc = model.predict(pool)
            batch = list(pool[np.argsort(-(mean + BONUS * unc))[:BATCH]])
        R += [evaluate(x) for x in batch]
        X += batch
        curve.append(max(r[0] for r in R))
    return R[int(np.argmax([r[0] for r in R]))], curve, chosen


if __name__ == "__main__":
    out = [dream_search(s) for s in range(N_SEEDS)]
    df = pd.DataFrame([{"method": "dream", "train_sharpe": r[0], "val_sharpe": r[1], "test_sharpe": r[2],
                        "test_return": r[3]} for r, _, _ in out])
    df.to_csv("results_dream.csv", index=False)

    hold_profits = np.mean([daily_profits(P[:, i], np.ones(T, dtype=int), COST) for i in range(P.shape[1])], axis=0)
    hold_test_ret = score(hold_profits[VAL_END:])["total_return"]
    old = pd.read_csv("results_rsi.csv")                         # random / evolve / rsi from rsi_search.py
    both = pd.concat([old, df])
    summary = both.groupby("method", sort=False).mean().round(3)
    summary["test_beats_hold"] = both.assign(b=both.test_return > hold_test_ret).groupby("method", sort=False).b.sum()
    print("Best-on-train candidate of each search, averaged over", N_SEEDS, "seeds")
    print(summary)
    print(f"buy and hold test return {hold_test_ret*100:.1f}%\n")
    print("dream: best train Sharpe found so far, by real evaluations used")
    print(pd.Series(np.mean([c for _, c, _ in out], axis=0), index=range(BATCH, BUDGET + 1, BATCH)).round(3).to_string())
    picks = pd.Series([p for _, _, ch in out for p in ch]).value_counts()
    print("\npolicy the dream chose to deploy (all rounds, all seeds):\n", picks.to_string())
