import numpy as np


def make_prices(n_days=1000, start_price=100.0, daily_change=0.01, seed=0):
    """Make a fake price chart.

    Each day, the price is yesterday's price times a small random change.
    daily_change is how big a typical change is (0.01 = about 1%).
    The seed fixes the randomness, so the same seed gives the same chart.
    """
    rng = np.random.default_rng(seed)
    changes = rng.normal(loc=0.0, scale=daily_change, size=n_days - 1)
    prices = start_price * np.cumprod(1 + changes)
    return np.concatenate([[start_price], prices])


def make_market(n_days=1000, start_price=100.0, daily_change=0.01, edge=0.15, seed=0):
    """Fake market with a hidden pattern. Returns (prices, volume, signal).

    Signal on day t fires when ALL of these hold (using only data up to day t):
      1. the last 5 days' total return is positive
      2. the last 10 days' volatility is below the market's normal volatility
      3. today's volume is above its 90th percentile
    When it fires, the NEXT day's average return is +edge * daily_change (0.15 = 0.15 sigma).
    Otherwise the average is 0. Every day also gets normal noise.

    `signal` is returned only so we can measure the pattern. Strategies must never see it.
    """
    rng = np.random.default_rng(seed)
    sigma = daily_change
    # volume: random, always positive. 90th percentile of this lognormal is known exactly.
    volume = rng.lognormal(mean=0.0, sigma=0.5, size=n_days)
    volume_90 = np.exp(0.5 * 1.2815515655446004)

    noise = rng.normal(0.0, sigma, size=n_days)
    returns = np.zeros(n_days)      # returns[t] = change from day t-1 to day t
    signal = np.zeros(n_days, dtype=int)
    for t in range(n_days):
        # the effect of yesterday's signal lands on today's return
        returns[t] = noise[t] + (edge * sigma if t > 0 and signal[t - 1] else 0.0)
        if t >= 10:
            momentum_up = returns[t - 4 : t + 1].sum() > 0
            calm = returns[t - 9 : t + 1].std() < sigma
            busy = volume[t] > volume_90
            signal[t] = int(momentum_up and calm and busy)

    returns[0] = 0.0
    prices = start_price * np.cumprod(1 + returns)
    return prices, volume, signal


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    a = make_prices(seed=1)
    b = make_prices(seed=1)
    c = make_prices(seed=2)
    print("same seed gives same chart:", np.array_equal(a, b))
    print("different seed gives different chart:", not np.array_equal(a, c))

    plt.plot(a, label="seed 1")
    plt.plot(c, label="seed 2")
    plt.xlabel("day")
    plt.ylabel("price")
    plt.legend()
    plt.show()
