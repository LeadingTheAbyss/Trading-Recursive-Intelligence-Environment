# How the project works now

The market has a **real hidden pattern**, and the strategy can finally **see** what the pattern is made of. Here is how each part fits together.

---

## 1. The market now has three things per day

```
day:      1    2    3    4    5    6    7   ...
price:   100  101  100  102  103  104  103
volume:   0.9  1.1  0.8  2.3  1.0  0.7  1.2     <- NEW: how much trading happened
signal:   0    0    0    0    1    0    0       <- NEW: hidden. Only the market knows it.
```

- **Price** and **volume** are what a trader can see.
- **Signal** is the secret. It marks days when the pattern is "on". A strategy must never see it. We keep it only to measure how big the pattern is.

---

## 2. The hidden pattern (what the market is hiding)

The signal turns on at the end of day t when **all three** are true:

```
  [1] MOMENTUM        [2] CALM              [3] BUSY
  price rose over     prices were not       volume is in the
  the last 5 days     jumping around        top 10% of its range
  (last 10 days)
        \                  |                    /
         \                 |                   /
          +------- all three at once ---------+
                           |
                           v
        TOMORROW's return gets a small push upward:  +edge
        (otherwise: pure random noise, average 0)
```

Because all three have to line up, it happens only about 3 days in 100.

```
tomorrow's return =  [ small push, only on signal days ]  +  [ random noise every day ]
                       edge x 1%                              size about 1%
```

The push is small compared with the noise. That is realistic, and it is why finding the pattern is hard.

---

## 3. The strategy: same three questions, but with adjustable answers

```
VolumeRuleStrategy asks, every day:

  [1] did the price rise more than ____ over the last ____ days?     (2 knobs)
  [2] is the daily wobble over the last ____ days below ____ ?       (2 knobs)
  [3] is today's volume above the ____th percentile of 100 days?     (1 knob)

  all yes -> hold 1 share      any no -> hold nothing
```

Five knobs. The right knobs match the pattern. Wrong knobs mean it never fires:

```
 knobs match the pattern       knobs are off
 ----------------------        ----------------------
 fires on the real days        fires on random days (loses to costs)
 makes money                   or never fires: exactly 0
```

**What a "search" means now:** find good values for the five knobs without being told the answer.

```
 5 knobs, each with many values  ->  millions of combinations
 testing every one costs time    ->  we can only test a few thousand
 so HOW we pick which to test matters   <- this is what your project is about
```

---

## 4. Why we can't just cheat

Two rules protect every result:

```
RULE 1: no peeking              RULE 2: judge on data the search never saw

 on day t, the strategy          |=== first half: search here ===|=== second half: judge here ===|
 gets prices[0..t] only.
 never day t+1.
 (tested: rewrite the future,
  today's decision doesn't change)
```

Rule 2 matters because of the luck experiment: try enough strategies and the best one looks great on the past by pure chance.

---

## 5. The yardstick: the oracle

The **oracle** is a fake trader who is told the secret signal and holds a share only after it fires.

```
 best possible        <- oracle (knows the secret)
     |
 strategy with the right knobs   about 70-75% of the oracle
     |
 strategy with wrong knobs       about 0%
     |
 moving average (blind to volume)   loses money to costs
```

Any search method you build gets scored between these. "Good search" means getting close to the right knobs using few tests.

---

## 6. Pattern strength (the `edge`) sets how hard the game is

```
 edge   oracle's return   right-knobs strategy
 0.15      -1.5%               -2.0%      too weak: even perfect knowledge loses to costs
 0.3       +3.5%               +2.0%
 0.5      +10.9%               +7.8%      <- a good level: clearly findable but not trivial
 1.0      +31.0%              +23.4%      too easy: almost anything near right wins
```

---

## 7. The files

| File | What it does |
|---|---|
| `market.py` | makes the fake market (`make_market` has the hidden pattern) |
| `strategy.py` | the strategies; `get_positions` enforces no peeking |
| `backtest.py` | turns decisions into profit, costs, and 3 scores |
| `test_backtest.py` | checks with known answers, run `py test_backtest.py` |
| `experiment_*.py` | the experiments (A/B, luck, pattern strength) |

---

## What's next

```
 [done] market with a real pattern + strategy that can see it
   next: plain search (random vs breeding) on the 5 knobs, judged on the second half
   then: the layer that learns HOW to search better
```
