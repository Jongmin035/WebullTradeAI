"""
Daily prediction engine.

Loads the winning model artifacts from disk (downloading from S3 if needed),
fetches the latest market data for all symbols, and returns a predictions
DataFrame ready for trader.rebalance().

Returns:
    DataFrame with columns: symbol, clf_prob, reg_pred
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for d in (_src, os.path.join(_src, "core"), os.path.join(_src, "pipeline"), os.path.join(_src, "models")):
    if d not in sys.path:
        sys.path.insert(0, d)

from data_pipeline import load_all_symbols, INDICATORS_DIR
from indicators import Indicators
from model_store import load_artifacts, load_metadata
from regime_pipeline import fetch_spy_regimes, FEATURES

log = logging.getLogger(__name__)

VERSION = os.environ.get("GIT_SHA", "dev")

SEQ_LEN          = 20   # days of history needed per LSTM sequence
FEATURE_BUFFER   = 250  # extra history days to load so indicators warm up correctly
RANK_FEATURES    = ["rsi_rank", "volume_rank", "momentum_rank", "zscore20_rank", "volatility10_rank",
                    "pct_from_high_20d_rank", "recovery_slope_rank"]
RANK_SOURCE_COLS = {"rsi_rank": "rsi", "volume_rank": "volume",
                    "momentum_rank": "momentum", "zscore20_rank": "zscore20",
                    "volatility10_rank": "volatility10",
                    "pct_from_high_20d_rank": "pct_from_high_20d",
                    "recovery_slope_rank": "recovery_slope"}


# --- Data helpers ---

def _fetch_missing_days(symbols, from_date, to_date):
    """
    Batch-download OHLCV for all symbols for days in [from_date, to_date].
    Returns a dict {symbol: DataFrame} with columns: date, open, high, low, close, volume.
    Returns empty dict if no data needed.
    """
    import yfinance as yf
    start_str = pd.Timestamp(from_date).strftime("%Y-%m-%d")
    end_str   = (pd.Timestamp(to_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    log.info(f"Fetching {len(symbols)} symbols from yfinance ({start_str} → {end_str})...")
    raw = yf.download(
        symbols, start=start_str, end=end_str,
        auto_adjust=True, progress=False, group_by="ticker",
    )
    if raw.empty:
        return {}

    result = {}
    for sym in symbols:
        try:
            df = raw[sym][["Open", "High", "Low", "Close", "Volume"]].copy()
            df = df.dropna()
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index = pd.to_datetime(df.index)
            df = df.reset_index().rename(columns={"index": "date", "Date": "date"})
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df["symbol"] = sym
            result[sym] = df
        except Exception:
            pass
    return result


def _compute_features_for_new_rows(sym, hist_df, new_rows_df):
    """
    Append new_rows_df to hist_df (for buffer), recompute indicators, return
    only the newly computed rows with all feature columns.
    """
    combined = pd.concat([hist_df, new_rows_df]).sort_values("date").reset_index(drop=True)
    combined["symbol"] = sym

    ind = Indicators(combined)
    ind.add_sma(20).add_sma(50).add_rsi().add_macd()
    feat_df = ind.df.copy()

    feat_df["return_lag1"]     = feat_df["close"].pct_change(1)
    feat_df["return_lag2"]     = feat_df["close"].pct_change(2)
    feat_df["return_lag3"]     = feat_df["close"].pct_change(3)
    feat_df["volatility10"]    = feat_df["return_lag1"].rolling(10).std()
    feat_df["price_range"]     = (feat_df["high"] - feat_df["low"]) / feat_df["close"]
    feat_df["close_vs_sma20"]  = feat_df["close"] / feat_df["sma20"] - 1
    feat_df["close_vs_sma50"]  = feat_df["close"] / feat_df["sma50"] - 1
    feat_df["zscore20"]        = (
        (feat_df["close"] - feat_df["sma20"]) /
        (feat_df["close"].rolling(20).std() + 1e-9)
    )
    feat_df["momentum"]        = feat_df["close"].pct_change(20)

    # New features (added alongside 5-day label retrain)
    feat_df["gap"]             = (feat_df["open"] - feat_df["close"].shift(1)) / feat_df["close"].shift(1).replace(0, np.nan)
    feat_df["recovery_slope"]  = feat_df["close"].pct_change(5)
    _rolling_high_20           = feat_df["close"].rolling(20).max()
    feat_df["pct_from_high_20d"] = (_rolling_high_20 - feat_df["close"]) / _rolling_high_20.replace(0, np.nan)
    feat_df["range_tightness"] = (feat_df["close"].rolling(5).max() - feat_df["close"].rolling(5).min()) / feat_df["close"].replace(0, np.nan)
    _sma200                    = feat_df["close"].rolling(200).mean()
    feat_df["sma50_vs_sma200"] = ((feat_df["sma50"] - _sma200) / _sma200.replace(0, np.nan)).fillna(0.0)
    _don_high                  = feat_df["close"].rolling(55).max()
    _don_low                   = feat_df["close"].rolling(55).min()
    feat_df["donchian_55_pos"] = (feat_df["close"] - _don_low) / (_don_high - _don_low).replace(0, np.nan)
    _price_chg                 = feat_df["close"].diff()
    _direction                 = np.where(_price_chg > 0, 1, np.where(_price_chg < 0, -1, 0))
    _obv                       = (feat_df["volume"] * _direction).cumsum()
    _obv_std                   = _obv.rolling(20).std().replace(0, np.nan)
    feat_df["obv_zscore"]      = (_obv - _obv.rolling(20).mean()) / _obv_std
    _tr                        = pd.concat([
        feat_df["high"] - feat_df["low"],
        (feat_df["high"] - feat_df["close"].shift(1)).abs(),
        (feat_df["low"]  - feat_df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    feat_df["atr14_pct"]       = _tr.rolling(14).mean() / feat_df["close"].replace(0, np.nan)

    # Keep only new rows
    cutoff  = new_rows_df["date"].min()
    new_computed = feat_df[feat_df["date"] >= cutoff].copy()
    return new_computed


def _update_local_parquets(new_rows):
    """
    Append newly computed rows to their existing local parquets.
    Called after fetching missing days so the next daily run only
    needs to fetch 1 day instead of re-downloading everything.
    Only writes columns that already exist in the stored parquet
    to prevent schema drift from leaking into training data.
    """
    from data_pipeline import fetch_vix
    vix_cache = {}

    for sym, grp in new_rows.groupby("symbol"):
        fpath = os.path.join(INDICATORS_DIR, f"{sym}.parquet")
        if not os.path.exists(fpath):
            continue
        try:
            existing = pd.read_parquet(fpath)
            existing["date"] = pd.to_datetime(existing["date"]).dt.normalize()

            # Restrict new rows to only columns in existing schema
            grp = grp[[c for c in existing.columns if c in grp.columns]].copy()

            # Fill VIX for new rows using cached fetch
            if "vix" in existing.columns and "vix" not in grp.columns:
                grp["vix"] = float("nan")
            if "vix" in grp.columns and grp["vix"].isna().any():
                dates = grp["date"].dropna()
                if not dates.empty:
                    key = (dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d"))
                    if key not in vix_cache:
                        vix_cache[key] = fetch_vix(key[0], key[1])
                    vdf = vix_cache[key]
                    if vdf is not None:
                        vdf = vdf.copy()
                        vdf["date"] = pd.to_datetime(vdf["date"])
                        grp = grp.merge(vdf.rename(columns={"vix": "_vf"}), on="date", how="left")
                        grp["vix"] = grp["vix"].fillna(grp["_vf"])
                        grp = grp.drop(columns=["_vf"])

            updated = (
                pd.concat([existing, grp], ignore_index=True)
                  .drop_duplicates("date")
                  .sort_values("date")
                  .reset_index(drop=True)
            )
            updated.to_parquet(fpath, index=False)
        except Exception as e:
            log.warning(f"Could not update local parquet for {sym}: {e}")
    log.info(f"Updated {new_rows['symbol'].nunique()} local parquets with new data.")


def _merge_sentiment(combined, symbols):
    """Merge sentiment parquets into combined features. Fills 0.0 when files are missing."""
    try:
        sys.path.insert(0, os.path.join(_src, "pipeline"))
        from sentiment import load_sentiment, SENTIMENT_DIR
        if not os.path.exists(SENTIMENT_DIR):
            raise FileNotFoundError
    except Exception:
        for col in ["sentiment_1d", "sentiment_3d", "sentiment_7d"]:
            combined[col] = 0.0
        return combined

    frames = [load_sentiment(sym) for sym in symbols]
    frames = [f for f in frames if f is not None]

    if not frames:
        for col in ["sentiment_1d", "sentiment_3d", "sentiment_7d"]:
            combined[col] = 0.0
        return combined

    sent = pd.concat(frames, ignore_index=True)
    combined = combined.merge(sent, on=["date", "symbol"], how="left")
    for col in ["sentiment_1d", "sentiment_3d", "sentiment_7d"]:
        combined[col] = combined[col].fillna(0.0)
    return combined


def _add_rank_features(df):
    """Compute cross-symbol rank features (0–1) for each date group."""
    df = df.copy()
    for rank_col, src_col in RANK_SOURCE_COLS.items():
        if src_col in df.columns:
            df[rank_col] = (
                df.groupby("date")[src_col]
                  .rank(pct=True)
            )
        else:
            df[rank_col] = 0.5
    return df


def load_latest_features(symbols, window_days=None):
    """
    Load the most recent market features for all symbols.

    1. Load the last max(SEQ_LEN + FEATURE_BUFFER, window_days) rows from each parquet.
    2. Check if today's data is present; if not, fetch missing days via yfinance.
    3. Recompute indicators for the appended rows.
    4. Add cross-symbol rank features.
    5. Return a DataFrame with the latest rows (one per symbol, or SEQ_LEN rows for LSTM).

    Parameters
    ----------
    window_days : int or None
        How many recent days to return per symbol (None = just the latest row).
    """
    need_days = max(SEQ_LEN + FEATURE_BUFFER, (window_days or 0) + FEATURE_BUFFER)
    today     = pd.Timestamp.today().normalize()
    cutoff    = today - pd.Timedelta(days=need_days)

    # Load from parquets
    all_dfs = []
    missing_sym = []
    for sym in symbols:
        fpath = os.path.join(INDICATORS_DIR, f"{sym}.parquet")
        if not os.path.exists(fpath):
            missing_sym.append(sym)
            continue
        df = pd.read_parquet(fpath)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df[df["date"] >= cutoff].copy()
        df["symbol"] = sym
        all_dfs.append(df)

    if missing_sym:
        log.warning(f"{len(missing_sym)} symbols have no local parquet — skipping: {missing_sym[:5]}...")

    if not all_dfs:
        raise RuntimeError("No parquet data found locally. Run retrain.py first to download data.")

    combined = pd.concat(all_dfs, ignore_index=True)

    # Check if recent data is missing (parquets updated weekly)
    latest_parquet_date = combined["date"].max()
    yesterday = (today - pd.Timedelta(days=1)).normalize()

    if latest_parquet_date < yesterday:
        log.info(f"Parquet data ends {latest_parquet_date.date()} — fetching missing days from yfinance...")
        new_data = _fetch_missing_days(symbols, latest_parquet_date + pd.Timedelta(days=1), today)

        new_rows_list = []
        for sym in symbols:
            if sym not in new_data or new_data[sym].empty:
                continue
            hist = combined[combined["symbol"] == sym].copy()
            new_computed = _compute_features_for_new_rows(sym, hist, new_data[sym])
            new_rows_list.append(new_computed)

        if new_rows_list:
            new_all = pd.concat(new_rows_list, ignore_index=True)
            _update_local_parquets(new_all)
            combined = pd.concat([combined, new_all], ignore_index=True)

    combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)

    # VIX is market-wide and stored in parquets; today's new row may be missing it.
    # Forward-fill per symbol so the most recent day uses yesterday's VIX.
    if "vix" in combined.columns:
        combined["vix"] = combined.groupby("symbol")["vix"].ffill()

    # Always recompute rank features on the full combined so old and new rows
    # are consistent. Rank features are cross-sectional and not stored in parquets.
    combined = _add_rank_features(combined)

    # Merge sentiment features from local parquets (fill 0.0 if missing)
    combined = _merge_sentiment(combined, symbols)

    return combined


# --- Prediction functions ---

def _build_lstm_sequences(features_df, scaler):
    """Build the last SEQ_LEN rows per symbol into (n_symbols, SEQ_LEN, n_features) sequences."""
    import torch
    feat_cols = [f for f in FEATURES if f in features_df.columns]
    seqs, syms = [], []

    for sym, grp in features_df.groupby("symbol"):
        grp = grp.sort_values("date").dropna(subset=feat_cols)
        if len(grp) < SEQ_LEN:
            continue
        vals = grp[feat_cols].values[-SEQ_LEN:]   # shape: (SEQ_LEN, n_features)
        seqs.append(vals)
        syms.append(sym)

    if not seqs:
        return None, []

    X = np.stack(seqs).astype(np.float32)   # (n, SEQ_LEN, n_features)
    n, s, f = X.shape
    X_scaled = scaler.transform(X.reshape(-1, f)).reshape(n, s, f).astype(np.float32)
    return torch.tensor(X_scaled), syms


def _predict_lstm(artifacts, features_df):
    """Generate predictions using saved LSTM models."""
    import torch

    clf_model_cfg = artifacts.get("model_config", {"input_dim": len(FEATURES), "hidden_size": 32, "num_layers": 2, "dropout": 0.0})

    # Rebuild LSTM model from state dict
    class LSTMModel(torch.nn.Module):
        def __init__(self, cfg, task):
            super().__init__()
            self.task = task
            self.lstm = torch.nn.LSTM(
                input_size=cfg["input_dim"], hidden_size=cfg["hidden_size"],
                num_layers=cfg["num_layers"], batch_first=True,
                dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0,
            )
            self.drop = torch.nn.Dropout(cfg["dropout"])
            self.fc   = torch.nn.Linear(cfg["hidden_size"], 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            logit  = self.fc(self.drop(out[:, -1, :])).squeeze(1)
            return torch.sigmoid(logit) if self.task == "clf" else logit

    clf_model = LSTMModel(clf_model_cfg, "clf")
    clf_model.load_state_dict(artifacts["clf_state_dict"])
    clf_model.eval()

    reg_model = LSTMModel(clf_model_cfg, "reg")
    reg_model.load_state_dict(artifacts["reg_state_dict"])
    reg_model.eval()

    X_clf, syms = _build_lstm_sequences(features_df, artifacts["clf_scaler"])
    X_reg, _    = _build_lstm_sequences(features_df, artifacts["reg_scaler"])

    if X_clf is None or len(syms) == 0:
        return pd.DataFrame(columns=["symbol", "clf_prob", "reg_pred"])

    with torch.no_grad():
        clf_prob = clf_model(X_clf).numpy()
        reg_pred = reg_model(X_reg).numpy()

    return pd.DataFrame({"symbol": syms, "clf_prob": clf_prob, "reg_pred": reg_pred})


# --- Public API ---

def get_predictions_today(symbols):
    """
    Load the winning model and generate predictions for all symbols for today.

    Returns
    -------
    DataFrame with columns: symbol, clf_prob, reg_pred
    """
    meta = load_metadata()
    if meta is None:
        raise RuntimeError(
            "No model metadata found. Run src/aws/retrain.py first."
        )

    log.info(f"Using model: lstm  (trained up to {meta['trained_up_to']})")

    artifacts    = load_artifacts("lstm")
    features_df  = load_latest_features(symbols, window_days=SEQ_LEN + 5)

    preds = _predict_lstm(artifacts, features_df)

    preds = preds.dropna(subset=["clf_prob", "reg_pred"]).reset_index(drop=True)
    log.info(f"Generated predictions for {len(preds)} symbols")
    return preds


def get_today_regime_vix():
    """
    Fetch today's market regime label and VIX level for allocation.

    Returns
    -------
    (regime, vix) — regime is 'bull' | 'sideways' | 'bear', vix is float.
    Always returns a valid regime — falls back to ('sideways', 20.0) on any
    data failure so the allocator always has a sensible input.
    """
    import yfinance as yf

    try:
        today = pd.Timestamp.today().normalize()
        start = (today - pd.Timedelta(days=300)).strftime("%Y-%m-%d")
        end   = today.strftime("%Y-%m-%d")

        spy_regimes = fetch_spy_regimes(start, end)
        regime = spy_regimes.iloc[-1] if not spy_regimes.empty else "sideways"

        try:
            vix_df = yf.download(
                "^VIX",
                start=(today - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                end=end,
                progress=False,
                auto_adjust=True,
            )
            vix = float(vix_df["Close"].iloc[-1]) if not vix_df.empty else 20.0
        except Exception:
            vix = 20.0

    except Exception as e:
        import traceback, boto3, os
        log.warning(f"Regime detection failed ({e}) — defaulting to sideways/VIX 20")
        try:
            boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1")).put_object(
                Bucket=os.getenv("AWS_S3_BUCKET", "webull-trade-ai"),
                Key=f"diagnostics/{pd.Timestamp.today().strftime('%Y-%m-%d')}-regime-error.txt",
                Body=traceback.format_exc(),
                ContentType="text/plain",
            )
        except Exception:
            pass
        regime, vix = "sideways", 20.0

    log.info(f"Today's regime: {regime}  VIX: {vix:.1f}")
    return regime, vix


# Target allocations per regime:
#   bull:     60% venture picks + 25% SPY (filler) + 15% cash
#   sideways: 40% venture picks + 20% SPY (filler) + 40% cash
#   bear:     15% venture picks + 30% SH/SQQQ (hedge) + 55% cash
#
# Goal: track 80-90% of bull index returns via high-conviction picks + SPY filler;
# preserve capital in bear via cash + short ETFs; deploy cash aggressively in recovery.
_REGIME_ALLOCATION = {
    "bull":     {"venture_pct": 0.60, "safety_pct": 0.25, "hedge_pct": 0.00, "cash_pct": 0.15},
    "sideways": {"venture_pct": 0.40, "safety_pct": 0.20, "hedge_pct": 0.00, "cash_pct": 0.40},
    "bear":     {"venture_pct": 0.15, "safety_pct": 0.00, "hedge_pct": 0.30, "cash_pct": 0.55},
}


def get_allocation_today(artifacts, regime, vix):
    """
    Return today's bucket allocation based on regime.

    Parameters
    ----------
    artifacts : dict returned by load_artifacts() — unused, kept for signature compat
    regime    : str — 'bull' | 'sideways' | 'bear'
    vix       : float — current VIX level

    Returns
    -------
    dict: venture_pct, safety_pct, hedge_pct, cash_pct, regime, vix
    """
    alloc = _REGIME_ALLOCATION.get(regime, _REGIME_ALLOCATION["sideways"]).copy()
    alloc["regime"] = regime
    alloc["vix"]    = round(vix, 1)
    return alloc
