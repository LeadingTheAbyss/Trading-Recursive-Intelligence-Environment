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

4. We moved to real data: 48 big NSE stocks in INR (Nov 2017 to Oct 2026), with real Zerodha/Groww delivery costs (about 0.24% per buy+sell round trip, mostly government tax). We tried two simple strategies on it:
   - **Moving average:** hold if today's price is above its 20-day average.
   - **Shape-matching:** hold if past charts that looked like the last 5 days tended to go up the next day.

   Both lost to simply buying and holding. They trade too often (250 to 1,000 trades per stock), so costs eat the profit, and the market rose a lot in this period. Caveats: I picked today's big companies (survivorship bias), and there is no train/test split yet.

   What we learned:
   - **Buy and hold is the yardstick to beat,** not the model we built. A strategy only counts if it beats holding on data it never saw, after costs.
   - **RSI here means recursive self-improvement,** not the stock indicator. It will be added after a plain search baseline (random search vs breeding over the strategy settings), judged on unseen periods.
