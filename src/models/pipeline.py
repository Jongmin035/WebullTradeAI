import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_exploration import load_csv
from indicators import Indicators
from backtest import run_backtest

# --- Configuration ---
SYMBOLS      = ["AAPL", "XOM", "NVDA", "NFLX"]
ALL_YEARS    = [2022, 2023, 2024, 2025]
N_ESTIMATORS = 100
MAX_DEPTH    = 5
RANDOM_STATE = 42
MIN_TRAIN_MONTHS = 6

FEATURES = [
    "close", "volume",
    "sma20", "sma50",
    "rsi",
    "macd", "signal", "histogram",
    "return_lag1", "return_lag2", "return_lag3",
    "volatility10",
    "price_range",
    "close_vs_sma20", "close_vs_sma50",
]


# --- Data Loading ---
def load_all_symbols(years):
    """Load classification and regression targets for all symbols."""
    clf_frames, reg_frames = [], []
    for symbol in SYMBOLS:
        for year in years:
            records = load_csv(symbol, year)

            df_clf = pd.DataFrame(records)
            df_clf = Indicators(df_clf).build(target_type="classification")
            df_clf["symbol"] = symbol
            clf_frames.append(df_clf)

            df_reg = pd.DataFrame(records)
            df_reg = Indicators(df_reg).build(target_type="regression")
            df_reg["symbol"] = symbol
            reg_frames.append(df_reg)

    clf = pd.concat(clf_frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    reg = pd.concat(reg_frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    return clf, reg


# --- Walk-Forward Pipeline ---
def run_pipeline(clf_df, reg_df):
    """
    For each test month (expanding window), train both models and collect predictions.
    Returns a predictions DataFrame ready for backtesting.
    """
    clf_df = clf_df.copy()
    reg_df = reg_df.copy()
    clf_df["year_month"] = clf_df["date"].dt.to_period("M")
    reg_df["year_month"] = reg_df["date"].dt.to_period("M")
    months = sorted(clf_df["year_month"].unique())

    all_preds = []

    for i in range(MIN_TRAIN_MONTHS, len(months) - 1):
        train_months = months[:i + 1]
        test_month   = months[i + 1]

        # --- Classifier ---
        clf_train = clf_df[clf_df["year_month"].isin(train_months)]
        clf_test  = clf_df[clf_df["year_month"] == test_month]

        clf_model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE
        )
        clf_model.fit(clf_train[FEATURES], clf_train["target"])
        clf_prob = clf_model.predict_proba(clf_test[FEATURES])[:, 1]

        # --- Regressor ---
        reg_train = reg_df[reg_df["year_month"].isin(train_months)]
        reg_test  = reg_df[reg_df["year_month"] == test_month]

        reg_model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE
        )
        reg_model.fit(reg_train[FEATURES], reg_train["target"])
        reg_pred = reg_model.predict(reg_test[FEATURES])

        # --- Combine predictions ---
        preds = clf_test[["date", "symbol"]].copy().reset_index(drop=True)
        preds["clf_prob"]      = clf_prob
        preds["reg_pred"]      = reg_pred
        preds["actual_return"] = reg_test["target"].values
        all_preds.append(preds)

        print(f"  {test_month}  train={len(clf_train):>5}  test={len(clf_test):>3}")

    return pd.concat(all_preds, ignore_index=True)


# --- Main ---
if __name__ == "__main__":
    print("Loading data...")
    clf_df, reg_df = load_all_symbols(ALL_YEARS)
    print(f"Classifier samples: {len(clf_df)}, Regressor samples: {len(reg_df)}\n")

    print("=== Running Walk-Forward Pipeline ===")
    predictions = run_pipeline(clf_df, reg_df)
    print(f"\nTotal predictions: {len(predictions)}\n")

    print("=== Backtest Results (top_k=1) ===")
    run_backtest(predictions, top_k=1)

    print("\n=== Backtest Results (top_k=2) ===")
    run_backtest(predictions, top_k=2, plot=False)
