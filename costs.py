"""Real Indian delivery-trade costs for Zerodha and Groww (rates read from their pricing pages, Oct 2026).

We hold shares overnight, so these are EQUITY DELIVERY charges.
A trade of `amount` rupees costs some rupees; we return that as a fraction of the amount,
so it plugs straight into backtest.daily_profits(cost=(buy_cost, sell_cost)).
"""

# charges that are the same at every broker (government / exchange)
STT = 0.001            # securities transaction tax: 0.1% on buy AND on sell (delivery)
EXCHANGE = 0.0000307   # NSE transaction charge: 0.00307%
SEBI = 10 / 1e7        # Rs 10 per crore
STAMP_BUY = 0.00015    # stamp duty: 0.015%, buy side only
GST = 0.18             # on (brokerage + exchange + SEBI [+ DP charges at Groww])

BROKERS = {
    # brokerage: Rs 0 for delivery. DP charge (to take shares out of your account) on every sell.
    "zerodha": {"brokerage": lambda amount: 0.0, "dp_sell": 15.34},
    # brokerage: Rs 20 or 0.1% whichever is lower (min Rs 5). DP: Rs 3.5 + Rs 16.5, GST on top.
    "groww": {"brokerage": lambda amount: max(5.0, min(20.0, 0.001 * amount)), "dp_sell": (3.5 + 16.5) * 1.18},
}


def trade_costs(amount, broker="zerodha"):
    """Return (buy_cost, sell_cost) as fractions of `amount` rupees traded."""
    b = BROKERS[broker]
    brokerage = b["brokerage"](amount)
    taxed_fees = brokerage + EXCHANGE * amount + SEBI * amount
    common = STT * amount + taxed_fees + GST * taxed_fees
    buy = common + STAMP_BUY * amount
    sell = common + b["dp_sell"]
    return buy / amount, sell / amount


if __name__ == "__main__":
    for amount in [10_000, 100_000, 1_000_000]:
        for broker in BROKERS:
            buy, sell = trade_costs(amount, broker)
            print(f"Rs {amount:>9,}  {broker:8}  buy {buy*100:.3f}%   sell {sell*100:.3f}%   round trip {(buy+sell)*100:.3f}%")
