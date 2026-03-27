import pandas as pd
import numpy as np


class Indicators:
    """
    Calculates technical indicators and ML features given a OHLCV DataFrame.

    Usage:
        df = pd.read_csv(...)
        ind = Indicators(df)
        df_with_features = ind.build()
    """

    def __init__(self, df):
        # Work on a copy so the original is never modified
        self.df = df.copy().sort_values("date").reset_index(drop=True)

    # --- Core indicators ---

    def add_sma(self, window):
        self.df[f"sma{window}"] = self.df["close"].rolling(window=window).mean()
        return self

    def add_rsi(self, window=14):
        delta = self.df["close"].diff()
        gains  = pd.Series(np.where(delta > 0,  delta, 0), index=self.df.index)
        losses = pd.Series(np.where(delta < 0, -delta, 0), index=self.df.index)
        avg_gain = gains.rolling(window=window).mean()
        avg_loss = losses.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        self.df["rsi"] = 100 - (100 / (1 + rs))
        return self

    def add_macd(self, fast=12, slow=26, signal=9):
        ema_fast = self.df["close"].ewm(span=fast).mean()
        ema_slow = self.df["close"].ewm(span=slow).mean()
        self.df["macd"]      = ema_fast - ema_slow
        self.df["signal"]    = self.df["macd"].ewm(span=signal).mean()
        self.df["histogram"] = self.df["macd"] - self.df["signal"]
        return self

    # --- Additional ML features ---

    def add_lag_features(self, lags=(1, 2, 3)):
        """Daily return lagged by N days."""
        daily_return = self.df["close"].pct_change()
        for lag in lags:
            self.df[f"return_lag{lag}"] = daily_return.shift(lag)
        return self

    def add_volatility(self, window=10):
        """Rolling standard deviation of daily returns."""
        daily_return = self.df["close"].pct_change()
        self.df[f"volatility{window}"] = daily_return.rolling(window=window).std()
        return self

    def add_price_range(self):
        """Where does close sit within the day's high-low range (0=low, 1=high)."""
        hl_range = self.df["high"] - self.df["low"]
        self.df["price_range"] = (self.df["close"] - self.df["low"]) / hl_range.replace(0, np.nan)
        return self

    def add_price_vs_sma(self, window):
        """Close price as a ratio to SMA (e.g. 1.02 means 2% above SMA)."""
        sma_col = f"sma{window}"
        if sma_col not in self.df.columns:
            self.add_sma(window)
        self.df[f"close_vs_sma{window}"] = self.df["close"] / self.df[sma_col]
        return self

    # --- Target ---

    def add_target(self, horizon=5, target_type="classification"):
        """
        Add target column.
        target_type='classification' : 1 if price is higher after horizon days, else 0
        target_type='regression'     : actual % return after horizon days
        """
        future_close = self.df["close"].shift(-horizon)
        if target_type == "classification":
            self.df["target"] = (future_close > self.df["close"]).astype(int)
        elif target_type == "regression":
            self.df["target"] = (future_close - self.df["close"]) / self.df["close"]
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        return self

    # --- Build all ---

    def build(self, horizon=5, target_type="classification"):
        """Add all indicators and target, drop NaN rows and the last N rows (no target)."""
        (self
            .add_sma(20)
            .add_sma(50)
            .add_rsi()
            .add_macd()
            .add_lag_features()
            .add_volatility()
            .add_price_range()
            .add_price_vs_sma(20)
            .add_price_vs_sma(50)
            .add_target(horizon, target_type)
        )
        self.df = self.df.iloc[:-horizon].dropna().reset_index(drop=True)
        return self.df
