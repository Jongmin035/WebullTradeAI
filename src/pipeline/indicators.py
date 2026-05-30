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

    def add_zscore(self, window=20):
        """
        Z-score of close relative to its rolling mean.
        Measures how many standard deviations the price is above/below recent average.
        High z-score = overbought; low = oversold.
        """
        rolling_mean = self.df["close"].rolling(window=window).mean()
        rolling_std  = self.df["close"].rolling(window=window).std()
        self.df[f"zscore{window}"] = (self.df["close"] - rolling_mean) / rolling_std.replace(0, np.nan)
        return self

    def add_adx(self, window=14):
        """
        Average Directional Index (ADX) using Wilder's smoothing.
        Measures trend strength regardless of direction (0-100 scale).
          > 25 : strong trend (reliable bull or bear signal)
          < 25 : weak trend / sideways market
        Requires high, low, close columns.
        """
        high  = self.df["high"]
        low   = self.df["low"]
        close = self.df["close"]

        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        # Directional Movement
        up   = high - high.shift(1)
        down = low.shift(1) - low
        plus_dm  = pd.Series(np.where((up > down) & (up > 0),  up,   0.0), index=self.df.index)
        minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=self.df.index)

        # Wilder's smoothing: alpha = 1/window
        alpha = 1.0 / window
        atr        = tr.ewm(alpha=alpha,        adjust=False).mean()
        plus_di    = 100 * plus_dm.ewm(alpha=alpha,  adjust=False).mean() / atr.replace(0, np.nan)
        minus_di   = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)

        dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        self.df[f"adx{window}"] = dx.ewm(alpha=alpha, adjust=False).mean()
        return self

    def add_vix(self, vix_df):
        """
        Merge market-wide VIX (fear index) into each row by date.
        vix_df must have columns: date, vix  (daily VIX close values).
        High VIX = high market fear / volatility.
        """
        vix = vix_df[["date", "vix"]].copy()
        vix["date"] = pd.to_datetime(vix["date"])
        self.df = self.df.merge(vix, on="date", how="left")
        return self

    def add_momentum(self, long_window=252, short_window=21):
        """
        12-1 month momentum: return from 12 months ago to 1 month ago.
        Skips the most recent month to avoid short-term reversal contamination.
        One of the most replicated factors in academic finance.
        """
        self.df["momentum"] = (
            self.df["close"].shift(short_window) / self.df["close"].shift(long_window) - 1
        )
        return self

    # --- Target ---

    def add_target(self, horizon=5, target_type="classification"):
        """
        Add target column.
        target_type='classification'  : 1 if price is higher after horizon days, else 0
        target_type='regression'      : actual % return after horizon days
        target_type='risk_adjusted'   : % return / recent volatility (short-term Sharpe)
                                        rewards reliable moves, penalises noisy ones
        """
        future_close = self.df["close"].shift(-horizon)
        if target_type == "classification":
            self.df["target"] = (future_close > self.df["close"]).astype(int)
        elif target_type == "regression":
            self.df["target"] = (future_close - self.df["close"]) / self.df["close"]
        elif target_type == "risk_adjusted":
            raw_return  = (future_close - self.df["close"]) / self.df["close"]
            recent_vol  = self.df["close"].pct_change().rolling(window=20).std()
            self.df["target"] = raw_return / recent_vol.replace(0, np.nan)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        return self

    # --- Build all ---

    def build(self, horizon=5, target_type="classification", include_target=True, vix_df=None):
        """
        Add all indicators and optionally the target.
        include_target=False is used by the data pipeline to store features only —
        targets are added at training time since they vary by model type.
        vix_df : optional DataFrame with columns [date, vix] to add market fear index.
        """
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
            .add_zscore()
            .add_momentum()
        )
        if vix_df is not None:
            self.add_vix(vix_df)
        if include_target:
            self.add_target(horizon, target_type)
            self.df = self.df.iloc[:-horizon].dropna().reset_index(drop=True)
        else:
            self.df = self.df.dropna().reset_index(drop=True)
        return self.df


# --- Cross-sectional ranking ---

def cross_sectional_rank(df, features):
    """
    For each date, rank each feature across all stocks as a percentile (0=lowest, 1=highest).
    Adds new columns with '_rank' suffix.

    Example: rsi_rank=0.9 means this stock has higher RSI than 90% of stocks today.
    This converts absolute values into relative signals, which is how most quant
    funds construct factors.
    """
    rank_cols = {
        f"{feat}_rank": df.groupby("date")[feat].rank(pct=True)
        for feat in features
    }
    return df.assign(**rank_cols)
