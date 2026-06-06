"""
Data pipeline for precomputing and storing indicators.

Storage layout:
  src/data/indicators/{SYMBOL}.parquet
    - One file per symbol, all years
    - Contains OHLCV + all technical indicators
    - No target column (varies by model) — added at training time
    - No cross-sectional ranks — computed at training time (needs all symbols)

CLI usage:
  python data_pipeline.py --build                        # build all S&P 500 symbols, 2005-2025
  python data_pipeline.py --build --symbols AAPL XOM     # specific symbols only
  python data_pipeline.py --update                       # append latest bar for all symbols
  python data_pipeline.py --update --symbols AAPL        # update one symbol

Importing in model files:
  from data_pipeline import load_all_symbols
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import yfinance as yf

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _d in (_src, os.path.join(_src, "pipeline")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

INDICATORS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "indicators")
START_YEAR     = 2005
END_YEAR       = pd.Timestamp.today().year
RANK_FEATURES  = ["rsi", "volume", "momentum", "zscore20", "volatility10"]

FEATURES = [
    "close", "volume",
    "sma20", "sma50",
    "rsi",
    "macd", "signal", "histogram",
    "return_lag1", "return_lag2", "return_lag3",
    "volatility10",
    "price_range",
    "close_vs_sma20", "close_vs_sma50",
    "zscore20",
    "momentum",
    "vix",
    "rsi_rank", "volume_rank", "momentum_rank", "zscore20_rank", "volatility10_rank",
]


ETF_SYMBOLS = ["SPY", "QQQ", "XLP", "XLV", "XLU", "GLD", "XLI", "IWM"]


# --- S&P 500 symbols ---

def fetch_sp500_symbols():
    """Fetch current S&P 500 constituents from Wikipedia."""
    import requests, io
    url     = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    try:
        resp    = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        table   = pd.read_html(io.StringIO(resp.text))[0]
        symbols = table["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"Fetched {len(symbols)} S&P 500 symbols from Wikipedia.")
        return symbols
    except Exception as e:
        print(f"Could not fetch S&P 500 list ({e}). Using default symbols.")
        return ["AAPL", "XOM", "NVDA", "NFLX"]


# --- Fetch OHLCV from yfinance ---

def fetch_ohlcv(symbol, start, end):
    """Download OHLCV data from yfinance. Returns a clean DataFrame or None on failure."""
    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            return None
        df = df.reset_index()
        # yfinance may return MultiIndex columns when downloading single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [col.lower() for col in df.columns]
        df = df.rename(columns={"date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "open", "high", "low", "close", "volume"]].copy()
    except Exception as e:
        print(f"  Failed to fetch {symbol}: {e}")
        return None


# --- Fetch VIX ---

def fetch_vix(start, end):
    """
    Fetch VIX daily close for the given date range.
    Returns a DataFrame with columns [date, vix], or None on failure.
    """
    df = fetch_ohlcv("^VIX", start, end)
    if df is None:
        print("  Warning: could not fetch VIX data — vix feature will be NaN.")
        return None
    return df[["date", "close"]].rename(columns={"close": "vix"})


# --- Compute indicators ---

def compute_indicators(df, vix_df=None):
    """
    Compute all indicators for a single symbol DataFrame.
    Returns DataFrame with indicators, no target column.
    vix_df : optional DataFrame with [date, vix] columns.
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from indicators import Indicators
    return Indicators(df).build(include_target=False, vix_df=vix_df)


# --- Parquet I/O ---

def parquet_path(symbol):
    return os.path.join(INDICATORS_DIR, f"{symbol}.parquet")


def save_symbol(symbol, df):
    os.makedirs(INDICATORS_DIR, exist_ok=True)
    df.to_parquet(parquet_path(symbol), index=False)


def load_symbol(symbol):
    """Load precomputed indicators for one symbol. Returns None if file missing."""
    path = parquet_path(symbol)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# --- Build / update ---

def build_symbol(symbol, start_year=START_YEAR, end_year=END_YEAR, vix_df=None):
    """Fetch full history, compute indicators, save to Parquet."""
    start = f"{start_year}-01-01"
    end   = f"{end_year}-12-31"
    df = fetch_ohlcv(symbol, start, end)
    if df is None or len(df) < 300:   # need at least ~1 year for momentum lookback
        print(f"  {symbol}: insufficient data, skipping.")
        return False
    df = compute_indicators(df, vix_df=vix_df)
    df["symbol"] = symbol
    save_symbol(symbol, df)
    print(f"  {symbol}: {len(df)} rows saved.")
    return True


def update_symbol(symbol):
    """
    Fetch only the latest bars (last 30 days for safety), append any new rows,
    recompute the last row's indicators, and save.
    """
    existing = load_symbol(symbol)
    if existing is None:
        print(f"  {symbol}: no existing data, running full build.")
        return build_symbol(symbol)

    last_date = existing["date"].max()
    start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end   = pd.Timestamp.today().strftime("%Y-%m-%d")

    new_ohlcv = fetch_ohlcv(symbol, start=start, end=end)
    if new_ohlcv is None or new_ohlcv.empty:
        print(f"  {symbol}: already up to date.")
        return True

    # Fetch VIX for the update window (lookback + new rows)
    lookback_start = (last_date - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    vix_df = fetch_vix(start=lookback_start, end=end)

    # We need some history before the new rows to recompute rolling indicators
    # Pull last 300 rows of existing OHLCV and append
    lookback_ohlcv = existing[["date", "open", "high", "low", "close", "volume"]].tail(300)
    combined_ohlcv = pd.concat([lookback_ohlcv, new_ohlcv], ignore_index=True)
    combined_ohlcv = combined_ohlcv.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    recomputed = compute_indicators(combined_ohlcv, vix_df=vix_df)
    recomputed["symbol"] = symbol

    # Keep existing rows up to last_date, add newly computed rows after
    old_rows = existing[existing["date"] <= last_date]
    new_rows = recomputed[recomputed["date"] > last_date]
    updated = pd.concat([old_rows, new_rows], ignore_index=True)
    save_symbol(symbol, updated)
    print(f"  {symbol}: +{len(new_rows)} new rows (total {len(updated)}).")
    return True


# --- Load for model training ---

def add_target(df, horizon=5, target_type="classification"):
    """Add target column to a DataFrame (applied at training time)."""
    df = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    future_close = df.groupby("symbol")["close"].shift(-horizon)
    if target_type == "classification":
        df["target"] = (future_close > df["close"]).astype(int)
    elif target_type == "regression":
        df["target"] = (future_close - df["close"]) / df["close"]
    elif target_type == "risk_adjusted":
        raw_return = (future_close - df["close"]) / df["close"]
        recent_vol = df.groupby("symbol")["close"].pct_change().rolling(20).std().reset_index(0, drop=True)
        df["target"] = raw_return / recent_vol.replace(0, np.nan)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    # Drop last horizon rows per symbol (no future close available) using a mask
    rows_from_end = df.groupby("symbol").cumcount(ascending=False)
    df = df[rows_from_end >= horizon]
    return df.dropna().reset_index(drop=True)


def cross_sectional_rank(df, features=RANK_FEATURES):
    """Rank each feature across all symbols on the same date (percentile 0-1)."""
    from indicators import cross_sectional_rank as _rank
    return _rank(df, features)


def _merge_sentiment(combined, symbols):
    """
    Merge daily sentiment scores into combined DataFrame.
    Fills 0.0 for symbols/dates with no sentiment data (graceful degradation).
    """
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from sentiment import load_sentiment
    except ImportError:
        for col in ["sentiment_1d", "sentiment_3d", "sentiment_7d"]:
            combined[col] = 0.0
        return combined

    frames = [load_sentiment(sym) for sym in symbols]
    frames = [f for f in frames if f is not None]

    if not frames:
        print("No sentiment data found — using 0.0. Run `python sentiment.py --build` to generate.")
        for col in ["sentiment_1d", "sentiment_3d", "sentiment_7d"]:
            combined[col] = 0.0
        return combined

    sentiment_all = pd.concat(frames, ignore_index=True)
    combined = combined.merge(sentiment_all, on=["date", "symbol"], how="left")
    for col in ["sentiment_1d", "sentiment_3d", "sentiment_7d"]:
        combined[col] = combined[col].fillna(0.0)
    return combined


def load_all_symbols(symbols, target_type="classification", horizon=1):
    """
    Load precomputed indicators for all symbols, add target and cross-sectional ranks.
    Used by model files instead of building indicators from scratch each time.

    horizon : int — prediction horizon in trading days (default 1 for daily re-evaluation)
    """
    frames = []
    missing = []
    loaded = []
    for symbol in symbols:
        df = load_symbol(symbol)
        if df is None:
            missing.append(symbol)
        else:
            frames.append(df)
            loaded.append(symbol)

    if missing:
        print(f"Warning: {len(missing)} symbols have no precomputed data: {missing[:5]}{'...' if len(missing)>5 else ''}")
        print("Run `python data_pipeline.py --build` to precompute indicators.")

    if not frames:
        raise RuntimeError("No symbol data found. Run the pipeline first.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)
    combined = cross_sectional_rank(combined)
    combined = _merge_sentiment(combined, loaded)
    combined = add_target(combined, horizon=horizon, target_type=target_type)
    return combined


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="Data pipeline for precomputing indicators.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build",  action="store_true", help="Full build from scratch")
    group.add_argument("--update", action="store_true", help="Append latest bars only")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols to process (default: S&P 500)")
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year",   type=int, default=END_YEAR)
    args = parser.parse_args()

    symbols = list(set(args.symbols or fetch_sp500_symbols()) | set(ETF_SYMBOLS))
    print(f"Processing {len(symbols)} symbols (includes {len(ETF_SYMBOLS)} ETFs)...\n")

    # Fetch VIX once for all symbols (avoid 500 redundant downloads)
    vix_df = None
    if args.build:
        start = f"{args.start_year}-01-01"
        end   = f"{args.end_year}-12-31"
        print("Fetching VIX data...")
        vix_df = fetch_vix(start, end)

    success, failed = 0, []
    for symbol in symbols:
        if args.build:
            ok = build_symbol(symbol, args.start_year, args.end_year, vix_df=vix_df)
        else:
            ok = update_symbol(symbol)
        if ok:
            success += 1
        else:
            failed.append(symbol)

    print(f"\nDone. {success}/{len(symbols)} succeeded.")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
