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


# --- Kelly Criterion ---

def kelly_fraction(p, b, half_kelly=True):
    """
    Compute the Kelly fraction: optimal fraction of capital to bet.

    Uses the edge-based formula f = 2p - 1, which is the correct Kelly
    formula when win and loss magnitudes are symmetric (e.g. daily stock
    returns). The classical binary formula (p - q/b) gives near-zero
    fractions for stocks because daily returns b ~ 0.5-2% are tiny.

    Parameters
    ----------
    p          : float — probability of winning (clf_prob, 0-1)
    b          : float — expected return if the trade wins (reg_pred).
                         Used as a gate: no position if b <= 0.
    half_kelly : bool  — use half-Kelly (f/2) to reduce risk; recommended
                         because model estimates are never perfect

    Returns
    -------
    float in [0, 1] — fraction of capital to allocate to this trade
    """
    if b <= 0 or p <= 0.5:
        return 0.0
    f = 2.0 * p - 1.0     # edge fraction: how far above 50/50 we are
    f = min(f, 1.0)
    return f * 0.5 if half_kelly else f


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
