import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_exploration import load_csv
from indicators import Indicators
from metrics import regression_metrics

# --- Configuration ---
SYMBOLS      = ["AAPL", "XOM", "NVDA", "NFLX"]
ALL_YEARS    = [2022, 2023, 2024, 2025]
N_ESTIMATORS = 100
MAX_DEPTH    = 5
RANDOM_STATE = 42

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
    frames = []
    for symbol in SYMBOLS:
        for year in years:
            records = load_csv(symbol, year)
            df = pd.DataFrame(records)
            df = Indicators(df).build(target_type="regression")
            df["symbol"] = symbol
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined


# --- Training ---
def train(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model


# --- Walk-Forward Validation ---
def walk_forward_validate(df, min_train_months=6):
    df = df.copy()
    df["year_month"] = df["date"].dt.to_period("M")
    months = sorted(df["year_month"].unique())

    results = []
    for i in range(min_train_months, len(months) - 1):
        train_months = months[:i + 1]
        test_month   = months[i + 1]

        train_mask = df["year_month"].isin(train_months)
        test_mask  = df["year_month"] == test_month

        X_train = df.loc[train_mask, FEATURES]
        y_train = df.loc[train_mask, "target"]
        X_test  = df.loc[test_mask,  FEATURES]
        y_test  = df.loc[test_mask,  "target"]

        if len(X_test) == 0:
            continue

        model = train(X_train, y_train)
        y_pred = model.predict(X_test)
        dir_acc = accuracy_score(np.sign(y_test) > 0, np.sign(y_pred) > 0)
        results.append({"month": str(test_month), "dir_acc": dir_acc, "n_test": len(X_test)})
        print(f"  {test_month}  train={len(X_train):>5}  test={len(X_test):>3}  dir_acc={dir_acc:.4f}")

    avg = np.mean([r["dir_acc"] for r in results])
    print(f"\nWalk-Forward Average Directional Accuracy: {avg:.4f} over {len(results)} months")
    return results


# --- Main ---
if __name__ == "__main__":
    print("Loading data...")
    df = load_all_symbols(ALL_YEARS)
    print(f"Total samples: {len(df)}\n")

    print("=== Walk-Forward Validation ===")
    walk_forward_validate(df)

    print("\n=== Final Model: Train 2022-2024, Test 2025 ===")
    train_df = df[df["date"].dt.year.isin([2022, 2023, 2024])]
    test_df  = df[df["date"].dt.year == 2025]

    model = train(train_df[FEATURES], train_df["target"])

    print("\n--- Train ---")
    regression_metrics(train_df["target"], model.predict(train_df[FEATURES]))

    print("\n--- Test ---")
    regression_metrics(test_df["target"], model.predict(test_df[FEATURES]))

    importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    print("\nFeature Importances:")
    for name, imp in importances:
        print(f"  {name}: {imp:.4f}")
