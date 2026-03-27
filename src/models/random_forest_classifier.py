import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  # used in walk_forward_validate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_exploration import load_csv
from indicators import Indicators
from metrics import classification_metrics

# --- Configuration ---
SYMBOLS      = ["AAPL", "XOM", "NVDA", "NFLX"]
ALL_YEARS    = [2022, 2023, 2024, 2025]
N_ESTIMATORS = 100
MAX_DEPTH    = 5    # tuned via walk-forward search over depths 5-14
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
    """Load and combine feature data for all symbols across given years."""
    frames = []
    for symbol in SYMBOLS:
        for year in years:
            records = load_csv(symbol, year)
            df = pd.DataFrame(records)
            df = Indicators(df).build()
            df["symbol"] = symbol
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined


# --- Training ---
def train(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model


# --- Evaluation ---
def evaluate(model, X_test, y_test, label="Test"):
    y_pred = model.predict(X_test)
    print(f"\n--- {label} ---")
    return classification_metrics(y_test, y_pred)


# --- Walk-Forward Validation ---
def walk_forward_validate(df, min_train_months=6):
    """
    Expanding window walk-forward validation.
    Train on all data up to month M, test on month M+1.
    Requires at least min_train_months of data before the first test.
    """
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
        acc = accuracy_score(y_test, y_pred)
        results.append({"month": str(test_month), "accuracy": acc, "n_test": len(X_test)})
        print(f"  {test_month}  train={len(X_train):>5}  test={len(X_test):>3}  acc={acc:.4f}")

    avg_acc = np.mean([r["accuracy"] for r in results])
    print(f"\nWalk-Forward Average Accuracy: {avg_acc:.4f} over {len(results)} months")
    return results


# --- Main ---
if __name__ == "__main__":
    print("Loading data...")
    df = load_all_symbols(ALL_YEARS)
    print(f"Total samples: {len(df)}, up days: {df['target'].mean():.2%}\n")

    print("=== Walk-Forward Validation ===")
    walk_forward_validate(df)

    print("\n=== Final Model: Train 2022-2024, Test 2025 ===")
    train_df = df[df["date"].dt.year.isin([2022, 2023, 2024])]
    test_df  = df[df["date"].dt.year == 2025]

    model = train(train_df[FEATURES], train_df["target"])
    evaluate(model, train_df[FEATURES], train_df["target"], label="Train")
    evaluate(model, test_df[FEATURES],  test_df["target"],  label="Test")

    importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    print("\nFeature Importances:")
    for name, imp in importances:
        print(f"  {name}: {imp:.4f}")
