import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)


# --- Classification ---

def classification_metrics(y_true, y_pred, label=""):
    acc = accuracy_score(y_true, y_pred)
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Down", "Up"]))
    return acc


# --- Regression ---

def regression_metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    dir_acc = directional_accuracy(y_true, y_pred)

    prefix = f"[{label}] " if label else ""
    print(f"{prefix}MAE:  {mae:.4f}")
    print(f"{prefix}RMSE: {rmse:.4f}")
    print(f"{prefix}R²:   {r2:.4f}")
    print(f"{prefix}Directional Accuracy: {dir_acc:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2, "dir_acc": dir_acc}


def directional_accuracy(y_true, y_pred):
    """Fraction of predictions where the sign (up/down) is correct."""
    correct = np.sign(y_true) == np.sign(y_pred)
    return np.mean(correct)


# --- Sharpe Ratio ---

def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Annualized Sharpe ratio given a series of periodic returns.
    periods_per_year: 252 for daily, 52 for weekly.
    """
    returns = np.array(returns)
    excess  = returns - risk_free_rate / periods_per_year
    if excess.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()
