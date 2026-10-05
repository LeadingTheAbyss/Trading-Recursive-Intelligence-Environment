"""Download real daily prices (in INR) and volumes for big NSE stocks and save them in data/.

Run once:  py data_nse.py
Prices are adjusted for splits and dividends, so daily changes are real returns.
"""
import os
import pandas as pd
import yfinance as yf

START = "2015-01-01"
MIN_DAYS = 2000   # drop stocks with too little history

# large NSE companies (mostly the Nifty 50). Today's big companies only -> see "survivorship" note in README.
SYMBOLS = """RELIANCE TCS HDFCBANK INFY ICICIBANK HINDUNILVR ITC SBIN BHARTIARTL KOTAKBANK LT AXISBANK
ASIANPAINT MARUTI SUNPHARMA TITAN ULTRACEMCO BAJFINANCE NESTLEIND WIPRO HCLTECH ONGC NTPC POWERGRID
TATASTEEL JSWSTEEL M&M ADANIENT ADANIPORTS COALINDIA BAJAJFINSV TECHM INDUSINDBK GRASIM HINDALCO
DRREDDY CIPLA EICHERMOT HEROMOTOCO BRITANNIA APOLLOHOSP DIVISLAB BPCL TATACONSUM SBILIFE HDFCLIFE
BAJAJ-AUTO SHREECEM""".split()


def load_nse(folder="data"):
    """Return (close, volume): tables with one column per stock, one row per trading day."""
    close = pd.read_csv(os.path.join(folder, "nse_close.csv"), index_col=0, parse_dates=True)
    volume = pd.read_csv(os.path.join(folder, "nse_volume.csv"), index_col=0, parse_dates=True)
    return close, volume


if __name__ == "__main__":
    raw = yf.download([s + ".NS" for s in SYMBOLS], start=START, progress=False, auto_adjust=True)
    close, volume = raw["Close"], raw["Volume"]
    close.columns = close.columns.str.replace(".NS", "", regex=False)
    volume.columns = volume.columns.str.replace(".NS", "", regex=False)

    # keep stocks with enough history; fill the odd missing day with the last known value
    enough = close.notna().sum() >= MIN_DAYS
    dropped = list(close.columns[~enough])
    close, volume = close.loc[:, enough].ffill().dropna(), volume.loc[:, enough].fillna(0)
    volume = volume.loc[close.index]
    volume = volume.where(volume > 0, 1)          # zero volume breaks log(); treat as 1

    os.makedirs("data", exist_ok=True)
    close.to_csv("data/nse_close.csv")
    volume.to_csv("data/nse_volume.csv")
    print(f"saved {close.shape[1]} stocks x {close.shape[0]} days  ({close.index[0].date()} to {close.index[-1].date()})")
    if dropped:
        print("dropped (not enough history or no data):", dropped)
