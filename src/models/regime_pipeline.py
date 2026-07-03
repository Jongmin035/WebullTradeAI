"""
Regime detection for SPY.

Detects daily market regime from SPY using three signals:
  1. SMA crossover (SMA50 vs SMA200 + 0.5% buffer) — structural trend direction
  2. 21-day SPY rolling return — short-term momentum confirmation
  3. VIX level — market fear / uncertainty modifier
  ADX is used only for bear confirmation (avoids calling bear on short-term dips).

Final labels:
  Bull     : SMA50 > SMA200 + 21d return > 0, AND VIX < 30
  Bear     : (bearish SMA + ADX > 25 + 21d return < 0) OR VIX >= 40
  Sideways : everything else (includes "structurally bullish but VIX >= 30")
"""

import sys
import os
import pandas as pd
import yfinance as yf

SMA_FAST            = 50
SMA_SLOW            = 200
SMA_BUFFER          = 0.005
ADX_WINDOW          = 14
ADX_THRESHOLD       = 25
SHORT_RETURN_WINDOW = 21
VIX_DAMPENING       = 30
VIX_PANIC           = 40

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
    "rsi_rank", "volume_rank", "momentum_rank", "zscore20_rank", "volatility10_rank",
    "sentiment_1d", "sentiment_3d", "sentiment_7d",
]


def fetch_spy_regimes(start, end):
    """
    Returns a Series indexed by date with values: 'bull', 'bear', 'sideways'.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
    from indicators import Indicators

    extended_start = (pd.Timestamp(start) - pd.DateOffset(days=300)).strftime("%Y-%m-%d")
    spy = yf.download("SPY", start=extended_start, end=end, auto_adjust=True, progress=False)

    spy_df = spy[["Open", "High", "Low", "Close", "Volume"]].copy()
    spy_df.columns = ["open", "high", "low", "close", "volume"]
    spy_df.index = pd.to_datetime(spy_df.index)
    spy_df = spy_df.sort_index().reset_index().rename(columns={"index": "date", "Date": "date"})

    ind = Indicators(spy_df)
    ind.add_sma(SMA_FAST).add_sma(SMA_SLOW).add_adx(ADX_WINDOW)
    spy_feat = ind.df.set_index("date")

    sma_fast = spy_feat[f"sma{SMA_FAST}"]
    sma_slow = spy_feat[f"sma{SMA_SLOW}"]
    adx      = spy_feat[f"adx{ADX_WINDOW}"]

    bullish  = sma_fast > sma_slow * (1 + SMA_BUFFER)
    bearish  = sma_fast < sma_slow * (1 - SMA_BUFFER)
    trending = adx > ADX_THRESHOLD

    short_return = spy_feat["close"].pct_change(SHORT_RETURN_WINDOW)
    short_up = short_return > 0
    short_dn = short_return < 0

    vix_raw = yf.download("^VIX", start=extended_start, end=end, auto_adjust=True, progress=False)
    vix = vix_raw["Close"].squeeze()
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix = vix.reindex(spy_feat.index).ffill()

    regime = pd.Series("sideways", index=spy_feat.index, name="regime")
    regime[bullish & short_up] = "bull"               # ADX not required — direction + momentum is enough
    regime[bearish & trending & short_dn] = "bear"    # ADX still required — avoids calling bear on a dip
    regime[(regime == "bull") & (vix >= VIX_DAMPENING)] = "sideways"
    regime[vix >= VIX_PANIC] = "bear"

    return regime[regime.index >= pd.Timestamp(start)]


def label_regimes(df, spy_regimes):
    """Add regime column to df by joining on date."""
    df = df.copy()
    df["regime"] = df["date"].map(spy_regimes)
    df["regime"] = df["regime"].fillna("sideways")
    return df