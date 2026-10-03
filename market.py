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
