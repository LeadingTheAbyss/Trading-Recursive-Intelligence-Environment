"""Dream-RSI style strategy discovery: improve the PROCESS that invents strategies, not just the strategies.

Every search below shares the same machinery (so the comparison is fair):
  - a population of strategy recipes (genome.py), kept as a tree: every child remembers its parent and the
    kind of modification (operator) that made it
  - a diverse archive: the best recipe of each KIND (niche) is kept, not just the single best recipe
  - the same 30 children per round, 10 rounds = 300 real backtests, parents picked from the archive

The only thing that differs is the DISCOVERY POLICY: how it picks which modification to try on a parent.

  fixed   : every modification equally likely, forever.                          (fixed strategy search)
  online  : adapts from the last round only. No memory, no context.             (ablation: no replay, no dream)
  replay  : replays ALL past parent->child edges. Learns "in this kind of parent, this modification
            has tended to help". Real history only.                              (ablation: replay, no dream)
  dream   : replay + dreaming. Fits a cheap model of recipe -> score from all history, then tries
            hypothetical modifications inside that model (free) to estimate what would help
            where real history is thin.                                          (Dream-RSI style)
"""
import numpy as np
from genome import (FEATURES, WINDOWS, REGIME_WINDOWS, REBALANCE, SIZINGS, random_genome, key, niche, encode)

OPS = ["win_small", "win_big", "thr_small", "thr_big", "add_cond", "remove_cond", "swap_feat", "flip_sign",
       "rebalance", "regime_toggle", "regime_win", "sizing", "exposure", "crossover", "fresh"]
N_BUCKETS = 6          # parent context: (number of rules 1/2/3) x (has market filter or not)
ROUNDS, CHILDREN = 10, 30


# ---------- the modifications ----------

def _nearest(grid, value):
    return min(grid, key=lambda w: abs(np.log(w) - np.log(value)))


def _step(grid, current, scale, rng):
    new = _nearest(grid, current * np.exp(rng.normal(0, scale)))
    if new == current:                                        # make sure a "change" actually changes something
        i = grid.index(current) + rng.choice([-1, 1])
        new = grid[min(max(i, 0), len(grid) - 1)]
    return new


def applicable(g):
    return [not (op == "add_cond" and len(g["conds"]) >= 3) and not (op == "remove_cond" and len(g["conds"]) <= 1)
            for op in OPS]


def apply_op(op, g, rng, other=None):
    """Return a modified copy of recipe g. `other` is a second recipe, used only by crossover."""
    g = {**g, "conds": [dict(c) for c in g["conds"]]}
    c = g["conds"][rng.integers(len(g["conds"]))]
    if op in ("win_small", "win_big"):
        c["win"] = _step(WINDOWS, c["win"], 0.3 if op == "win_small" else 0.9, rng)
    elif op in ("thr_small", "thr_big"):
        c["q"] = round(float(np.clip(c["q"] + rng.normal(0, 0.05 if op == "thr_small" else 0.2), 0.3, 0.95)), 2)
    elif op == "add_cond":
        free = [f for f in FEATURES if f not in {x["feat"] for x in g["conds"]}]
        g["conds"].append({"feat": str(rng.choice(free)), "win": int(rng.choice(WINDOWS)),
                           "sign": int(rng.choice([-1, 1])), "q": round(float(rng.uniform(0.3, 0.9)), 2)})
    elif op == "remove_cond":
        g["conds"].pop(rng.integers(len(g["conds"])))
    elif op == "swap_feat":
        free = [f for f in FEATURES if f not in {x["feat"] for x in g["conds"]}]
        if free:
            c["feat"] = str(rng.choice(free))
    elif op == "flip_sign":
        c["sign"] = -c["sign"]
    elif op == "rebalance":
        g["rebalance"] = int(_step(REBALANCE, g["rebalance"], 0.8, rng))
    elif op == "regime_toggle":
        g["regime"] = None if g["regime"] else int(rng.choice(REGIME_WINDOWS))
    elif op == "regime_win":
        g["regime"] = int(_step(REGIME_WINDOWS, g["regime"], 0.6, rng)) if g["regime"] else int(rng.choice(REGIME_WINDOWS))
    elif op == "sizing":
        g["sizing"] = [s for s in SIZINGS if s != g["sizing"]][0]
    elif op == "exposure":
        g["exposure"] = round(float(np.clip(g["exposure"] + rng.normal(0, 0.15), 0.4, 1.0)), 2)
    elif op == "crossover":
        o = other if other is not None else random_genome(rng)
        take = [dict(x) for x in o["conds"]]
        g["conds"] = take if rng.random() < 0.5 else g["conds"][:1] + [x for x in take if x["feat"] != g["conds"][0]["feat"]][:2]
        for field in ("regime", "rebalance", "sizing", "exposure"):
            if rng.random() < 0.5:
                g[field] = o[field]
    elif op == "fresh":
        return random_genome(rng)
    return g


def bucket(g):
    return (len(g["conds"]) - 1) * 2 + (g["regime"] is not None)


# ---------- the dream: a cheap model of "recipe -> score", built from everything tried so far ----------

class DreamModel:
    LENGTH, NOISE = 1.5, 0.05

    def __init__(self, genomes, fits):
        self.X = np.array([encode(g) for g in genomes])
        y = np.array(fits)
        self.mu, self.sd = y.mean(), y.std() + 1e-9
        K = self._k(self.X, self.X) + self.NOISE * np.eye(len(y))
        self.Kinv = np.linalg.inv(K)
        self.alpha = self.Kinv @ ((y - self.mu) / self.sd)

    @classmethod
    def _k(cls, A, B):
        return np.exp(-((A[:, None, :] - B[None, :, :]) ** 2).sum(-1) / (2 * cls.LENGTH ** 2))

    def predict(self, genomes):
        Ks = self._k(np.array([encode(g) for g in genomes]), self.X)
        mean = self.mu + self.sd * (Ks @ self.alpha)
        unc = self.sd * np.sqrt(np.maximum(1 - np.einsum("ij,jk,ik->i", Ks, self.Kinv, Ks), 0))
        return mean, unc


# ---------- discovery policies ----------

class Policy:
    """Holds a table probs[bucket][op]. Rebuilt after every round from whatever this mode is allowed to learn from."""
    SHRINK, EXPLORE, FLOOR = 3.0, 0.5, 0.15

    def __init__(self, mode):
        self.mode = mode
        self.table = np.full((N_BUCKETS, len(OPS)), 1 / len(OPS))
        self.online_score = np.full(len(OPS), 0.25)

    def update(self, edges, archive, entries, rng):
        """edges: list of (bucket, op, parent_fit, child_fit) for the WHOLE history; the last round is edges[-CHILDREN:]."""
        if self.mode == "fixed" or not edges:
            return
        if self.mode == "online":                             # only the latest round, no context, no history
            last = edges[-CHILDREN:]
            for o in range(len(OPS)):
                hits = [c > p for (_, op, p, c) in last if op == o]
                if hits:
                    self.online_score[o] = 0.6 * self.online_score[o] + 0.4 * np.mean(hits)
            self.table[:] = (self.online_score + 0.05) / (self.online_score + 0.05).sum()
            return

        n = np.zeros((N_BUCKETS, len(OPS)))
        total = np.zeros((N_BUCKETS, len(OPS)))
        for b, o, p, c in edges:
            n[b, o] += 1
            total[b, o] += np.clip(c - p, -1, 1)
        scale = max(np.std([np.clip(c - p, -1, 1) for _, _, p, c in edges]), 1e-3)
        prior = np.zeros((N_BUCKETS, len(OPS)))               # what we believe before seeing real edges
        if self.mode == "dream":
            prior = self._dream(entries, archive, rng)
        mean = (total + self.SHRINK * prior) / (n + self.SHRINK)
        bonus = self.EXPLORE * scale * np.sqrt(np.log(n.sum(1, keepdims=True) + 2) / (n + 1))
        z = (mean + bonus) / (0.5 * scale)
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        self.table = (1 - self.FLOOR) * p + self.FLOOR / len(OPS)

    def _dream(self, entries, archive, rng, per_cell=12):
        """Dream: fit the cheap model on all history, then for every (parent kind, modification) imagine
        `per_cell` hypothetical children and record how much better the model thinks they would score."""
        model = DreamModel([e["g"] for e in entries], [e["fit_train"] for e in entries])
        elites = sorted(archive.values(), key=lambda e: -e["fit_train"])
        H = np.zeros((N_BUCKETS, len(OPS)))
        for b in range(N_BUCKETS):
            pool = [e for e in elites if bucket(e["g"]) == b][:10] or elites[:10]
            for o, op in enumerate(OPS):
                kids, parents = [], []
                for _ in range(per_cell):
                    parent = pool[rng.integers(len(pool))]
                    if not applicable(parent["g"])[o]:
                        continue
                    kids.append(apply_op(op, parent["g"], rng, other=elites[rng.integers(len(elites))]["g"]))
                    parents.append(parent["fit_train"])
                if kids:
                    mean, unc = model.predict(kids)
                    H[b, o] = np.mean(np.clip(mean + 0.5 * unc - np.array(parents), -1, 1))
        return H


# ---------- the search ----------

def search(mode, seed, market):
    init_rng = np.random.default_rng(seed)                    # same first population for every mode
    rng = np.random.default_rng(seed + 10_000)
    policy = Policy(mode)
    entries, edges, seen, archive = [], [], {}, {}            # archive: niche -> best entry
    success_by_round, table_by_round = [], []

    def evaluate(g, parent, op):
        k = key(g)
        if k not in seen:
            r = market.evaluate(g)
            entries.append({"g": g, "fit_train": r["fit_train"], "fit_val": r["fit_val"], "test": r["test"],
                            "val": r["val"], "train": r["train"], "parent": parent, "op": op,
                            "profits": r["profits"][:market.tr_end]})
            seen[k] = len(entries) - 1
            e = entries[-1]
            n = niche(g)
            if n not in archive or e["fit_train"] > archive[n]["fit_train"]:
                archive[n] = e
        return entries[seen[k]]

    for rnd in range(ROUNDS):
        wins = []
        for _ in range(CHILDREN):
            if rnd == 0:
                evaluate(random_genome(init_rng), None, None)
                continue
            elites = sorted(archive.values(), key=lambda e: -e["fit_train"])
            weights = 1 / (np.arange(len(elites)) + 3)
            parent = elites[rng.choice(len(elites), p=weights / weights.sum())]
            b = bucket(parent["g"])
            p = policy.table[b] * np.array(applicable(parent["g"]))
            o = rng.choice(len(OPS), p=p / p.sum())
            other = elites[rng.choice(len(elites), p=weights / weights.sum())]["g"]
            child = evaluate(apply_op(OPS[o], parent["g"], rng, other), parent, o)
            edges.append((b, o, parent["fit_train"], child["fit_train"]))
            wins.append(child["fit_train"] > parent["fit_train"])
        if rnd > 0:
            success_by_round.append(float(np.mean(wins)))
        policy.update(edges, archive, entries, rng)
        table_by_round.append(policy.table.copy())

    # pick what to deploy: the 10 best-on-train niche champions, then the best of those on VALIDATION
    champions = sorted(archive.values(), key=lambda e: -e["fit_train"])[:10]
    chosen = max(champions, key=lambda e: e["fit_val"])
    corr = np.corrcoef(np.array([e["profits"] for e in champions]))
    return {"chosen": chosen, "niches": len(archive), "evals": len(entries), "success": success_by_round,
            "diversity": float(1 - corr[np.triu_indices(len(champions), 1)].mean()),
            "policy": policy.table, "edges": edges}
