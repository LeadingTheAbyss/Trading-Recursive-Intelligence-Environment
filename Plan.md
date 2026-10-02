# TRIE: Trading Recursive Intelligence Environment

I am building a project called **TRIE: Trading Recursive Intelligence Environment**.

The core idea is to build a simulated trading environment where a population of agents/strategies interacts with the market and recursively improves how it searches for trading strategies.

## Current plan

Start with a fixed historical market environment and a population of around 100 agents/strategies.

Each agent can use a different trading strategy or strategy generated/mutated from other strategies. The agents observe the current market state, make trading decisions, and receive objective feedback such as:

- profit/loss
- risk-adjusted return
- drawdown
- Sharpe-like metrics
- survival over a time period

The important part is that this is **not primarily a stock prediction project**.

The main research focus is:

> Can an AI system recursively improve the process it uses to discover and explore trading strategies?

The first version should therefore establish a simple baseline where agents use fixed or relatively simple exploration strategies.

Then add an RSI layer where successful strategies/trajectories are stored and used to improve future strategy exploration.

Conceptually:

market environment
→ population of strategies
→ trading trajectories
→ objective evaluation
→ strategy selection/mutation
→ improved population
→ repeat

I want to eventually study whether recursively improving the strategy-search process leads to better and more robust strategies than fixed/random exploration.

## Important design principle

Keep the underlying model/LLM relatively fixed.

I do NOT want the project to become:

"LLM predicts whether a stock will go up."

The interesting part should be the **meta-level search and self-improvement mechanism**.

The system should be evaluated on unseen periods rather than only the data it evolved on, so that I can test whether the discovered strategies actually generalize instead of simply overfitting/backtesting well.

## Future improvements I want to explore

### 1. Dream-RSI style replay

Use the history of previous trading trajectories as a cheap replay simulator.

Instead of repeatedly running expensive real experiments, store previous market states, decisions, branches, outcomes, and strategy trajectories.

Then use this replay environment to cheaply evaluate many candidate exploration policies.

Conceptually:

past trading histories
→ replay simulator
→ test many possible search policies
→ identify promising policies
→ deploy improved policy
→ collect new history
→ update replay simulator
→ repeat

This is inspired by the core idea of Dream-RSI, but I do NOT want to simply reproduce the paper. I want to use the principle as a foundation and develop my own mechanism.

### 2. Attention-like strategy allocation

I want to investigate whether the system can learn to allocate its search/computation budget across its previous experience.

For example, given many strategy histories, branches, and market regimes, the system could learn which ones deserve more attention.

Something conceptually similar to:

strategy/history A → low priority
strategy/history B → high priority
strategy/history C → medium priority

The goal would be to learn:

> Which parts of the previous search are worth exploring further?

### 3. DP/value-estimation style reasoning

I also want to explore a more structured alternative to LLM heuristics.

Represent a search state such as:

current strategy state + market regime + remaining exploration budget

and estimate the value of continuing from that state.

This could allow useful search states to be reused instead of repeatedly exploring the same regions.

I want to investigate whether a DP/value-based approach can make the RSI/search process more efficient.

### 4. Market regime changes

Introduce sudden changes in the environment rather than having a stationary market.

Examples:

- bull/bear transitions
- volatility spikes
- crashes
- liquidity changes
- sector rotations
- macroeconomic shocks

Then test whether recursively improved strategies adapt to regimes that were not present during their original evolution.

### 5. News and external information

Later, add news events to the environment.

Different agents could receive:

- news immediately
- delayed news
- incomplete information
- noisy or ambiguous interpretations

This would introduce information asymmetry and decision-making under uncertainty.

The environment could generate events such as:

- interest-rate changes
- earnings surprises
- geopolitical events
- supply shocks
- market-wide crashes
- sector-specific events

I want the news/event generator to be stochastic enough that agents cannot simply memorize fixed responses.

### 6. Co-evolution

Eventually, I may have multiple populations or strategy families competing:

- trend following
- mean reversion
- momentum
- volatility strategies
- event/news strategies
- contrarian strategies
- market-making-like behavior

The interesting question would become whether useful strategy diversity emerges and whether the RSI system learns when to explore entirely new strategy families rather than continuously optimizing the current best strategy.

## What I want from you

Help me develop this as a serious AI research/project idea rather than a generic trading application.

Be critical.

Do not automatically agree with my ideas. Point out when something is already common, when an idea is poorly defined, when an experiment would be invalid, or when I am adding complexity without research value.

Prioritize:

1. a clear research question
2. a minimal working environment
3. objective evaluation
4. strong baselines
5. unseen-period/generalization testing
6. ablation studies
7. computational efficiency
8. genuine novelty in the RSI/search mechanism

I want the final project to demonstrate **recursive improvement of strategy exploration**, not simply a collection of LLM trading agents.

---

## Working agreement (how Claude should help)

- Maximize my learning and first-principles thinking. Do NOT build files/code on my behalf unprompted.
- Prefer asking guiding questions, explaining concepts, critiquing my reasoning, and suggesting exercises over handing over finished implementations.
- Be critical; push back on weak ideas, invalid experiments, and unneeded complexity.
