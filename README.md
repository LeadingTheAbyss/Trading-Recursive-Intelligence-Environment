# TRIE: Trading Recursive Intelligence Environment

See [Plan.md](Plan.md) for the full project idea.

## Learning Process

1. We built a 100 stocks ad hoc data to work on a baseline model.

   ![100 fake price charts](images/100_charts.png)

2. We're using the 0/1 DP logic of either taking a share or not, to build the baseline model. For now it's only 1 share for each stock, and we haven't considered how many shares to buy from each stock.

3. We ran the baseline on all 100 fake charts ([experiment_ab.py](experiment_ab.py), raw numbers in `results_ab.csv`). The rule: hold the stock if today's price is above its 20-day average, otherwise hold nothing. Each buy or sell costs 0.1%. We compared it to two yardsticks: buy-and-hold, and never trading (always 0 profit).

   | Average over 100 charts | Moving-average strategy | Buy and hold |
   |---|---|---|
   | Total return | -12.2% | -0.1% |
   | Charts that made money | 24 of 100 | 44 of 100 |
   | Sharpe | -0.32 | -0.01 |

   Other numbers for the strategy: it was invested about 48% of the days, made about 126 trades per chart, and paid about 12.6% of its money in trading costs. It beat buy-and-hold on 37 of 100 charts.

   What we learned:
   - The charts are pure randomness, so there is nothing real to find, and the strategy lost money on average.
   - The loss is almost exactly the trading costs (12.6% paid vs 12.2% lost). The rule found no edge, and the costs ate the money.
   - Results vary hugely between charts (from -51% to +130%). One chart tells you almost nothing, so always look at many.
