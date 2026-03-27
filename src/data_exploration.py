import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from webull.core.client import ApiClient
from webull.data.data_client import DataClient

# --- Configuration ---
APP_KEY = "eecbf4489f460ad2f7aecef37b267618"
APP_SECRET = "8abf920a9cc3cb7af3ea5e9e03850692"
REGION_ID = "us"
API_ENDPOINT = "us-openapi-alb.uat.webullbroker.com"

SYMBOLS = ["AAPL", "XOM", "NVDA", "NFLX"]
CATEGORY = "US_STOCK"
TARGET_YEAR = 2022
TIMESPAN = "D"
MAX_BARS = "1200"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")


# --- Client ---
def create_client():
    api_client = ApiClient(APP_KEY, APP_SECRET, REGION_ID)
    api_client.add_endpoint(REGION_ID, API_ENDPOINT)
    return DataClient(api_client)


# --- Data Fetching ---
def fetch_historical_bars(data_client, symbols, timespan=TIMESPAN, count=MAX_BARS):
    res = data_client.market_data.get_batch_history_bar(
        symbols=symbols,
        category=CATEGORY,
        timespan=timespan,
        count=count,
        real_time_required=False
    )
    if res.status_code != 200:
        raise RuntimeError(f"API error {res.status_code}: {res.text}")
    return res.json()


# --- Parsing ---
def parse_bars(raw_data):
    frames = {}
    for stock in raw_data.get("result", []):
        symbol = stock["symbol"]
        bars = stock.get("result", [])
        if not bars:
            print(f"Warning: no data returned for {symbol}")
            continue
        df = pd.DataFrame(bars)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.rename(columns={"time": "date"})
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
        df["volume"] = df["volume"].astype(float).astype(int)
        df = df.sort_values("date").reset_index(drop=True)
        frames[symbol] = df
    return frames


# --- Filtering ---
def filter_by_year(frames, year):
    return {
        symbol: df[df["date"].dt.year == year].reset_index(drop=True)
        for symbol, df in frames.items()
    }


# --- yfinance Fetching ---
def fetch_yfinance(symbols, year):
    frames = {}
    for symbol in symbols:
        df = yf.download(symbol, start=f"{year}-01-01", end=f"{year+1}-01-01", auto_adjust=True, progress=False)
        if df.empty:
            print(f"Warning: no data returned for {symbol}")
            continue
        df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df["volume"] = df["volume"].astype(int)
        frames[symbol] = df
        print(f"Fetched {len(df)} rows for {symbol}")
    return frames


# --- Storage ---
def save_to_csv(frames, output_dir, year=TARGET_YEAR):
    os.makedirs(output_dir, exist_ok=True)
    for symbol, df in frames.items():
        path = os.path.join(output_dir, f"{symbol}_{year}.csv")
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} rows for {symbol} -> {path}")


# --- CSV Loading ---
def load_csv(symbol, year=TARGET_YEAR, data_dir=OUTPUT_DIR):
    path = os.path.join(data_dir, f"{symbol}_{year}.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    records = df.to_dict(orient="records")
    return records


# --- Indicators (imported from indicators.py) ---
from indicators import Indicators

def calculate_rsi(closes, window=14):
    df = pd.DataFrame({"close": closes})
    return Indicators(df).add_rsi(window).df["rsi"]

def calculate_macd(closes):
    df = pd.DataFrame({"close": closes})
    ind = Indicators(df).add_macd().df
    return ind["macd"], ind["signal"], ind["histogram"]


# --- Graphing ---
def graph_data(ax, records, symbol):
    dates = [r["date"] for r in records]
    closes = pd.Series([r["close"] for r in records])
    sma20 = closes.rolling(window=20).mean()
    sma50 = closes.rolling(window=50).mean()
    ax.plot(dates, closes, label="Close")
    ax.plot(dates, sma20, label="SMA 20")
    ax.plot(dates, sma50, label="SMA 50")
    ax.set_title(symbol)
    ax.legend()


def plot_symbol_detail(symbol, year=TARGET_YEAR):
    records = load_csv(symbol, year)
    dates = [r["date"] for r in records]
    closes = pd.Series([r["close"] for r in records])
    volumes = [r["volume"] for r in records]
    sma20 = closes.rolling(window=20).mean()
    sma50 = closes.rolling(window=50).mean()

    rsi = calculate_rsi(closes)
    macd, signal, histogram = calculate_macd(closes)

    fig, (ax_price, ax_vol, ax_rsi, ax_macd) = plt.subplots(4, 1, figsize=(14, 14), gridspec_kw={"height_ratios": [3, 1, 1, 1]})
    fig.suptitle(symbol)

    ax_price.plot(dates, closes, label="Close")
    ax_price.plot(dates, sma20, label="SMA 20")
    ax_price.plot(dates, sma50, label="SMA 50")
    ax_price.set_ylabel("Price ($)")
    ax_price.legend()

    ax_vol.bar(dates, volumes, width=1)
    ax_vol.set_ylabel("Volume")

    ax_rsi.plot(dates, rsi, label="RSI 14", color="purple")
    ax_rsi.axhline(70, color="red", linestyle="--", linewidth=0.8)
    ax_rsi.axhline(30, color="green", linestyle="--", linewidth=0.8)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.legend()

    ax_macd.plot(dates, macd, label="MACD", color="blue")
    ax_macd.plot(dates, signal, label="Signal", color="orange")
    ax_macd.bar(dates, histogram, label="Histogram", color="grey", width=1)
    ax_macd.axhline(0, color="black", linewidth=0.8)
    ax_macd.set_ylabel("MACD")
    ax_macd.legend()

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_symbols(symbols, year=TARGET_YEAR):
    n = len(symbols)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols  # ceiling division
    fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 5 * rows))
    axes = [axes] if n == 1 else axes.flatten()
    for i, symbol in enumerate(symbols):
        records = load_csv(symbol, year)
        graph_data(axes[i], records, symbol)
    for j in range(i + 1, len(axes)):  # hide unused subplots (e.g. odd number of symbols)
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()
    plt.close()


# --- Main ---
if __name__ == "__main__":
    for symbol in SYMBOLS:
        plot_symbol_detail(symbol)
