"""
Portfolio allocator: regime + VIX → [venture, safety, hedge, cash] percentages.

Architecture:
  13 parameters (pre-softmax weights per regime + VIX slope):
    [0:4]  = bull     raw weights [venture, safety, hedge, cash]
    [4:8]  = sideways raw weights
    [8:12] = bear     raw weights
    [12]   = vix_slope: fraction of venture shifted to hedge per unit of
             normalized VIX above VIX_BASE (vix_factor = clip((vix-20)/(40-20), 0, 1))

Training:
  Walk-forward scipy.optimize (Nelder-Mead) maximizing out-of-sample Sharpe.
  Warm-starts from the previous month's optimal params to reduce iterations.

Bucket ETFs:
  Safety : SPY, XLP, XLU  — equal-weighted, stable/defensive
  Hedge  : GLDM, SH, SQQQ — equal-weighted, crisis-resilient
"""

import numpy as np
import pandas as pd

SAFETY_ETFS = ["SPY", "XLP", "XLU"]
HEDGE_ETFS  = ["GLDM", "SH", "SQQQ"]
REGIMES     = ["bull", "sideways", "bear"]

VIX_BASE      = 20   # below this VIX level: no adjustment applied
VIX_CAP       = 40   # above this VIX level: maximum adjustment applied
VENTURE_FLOOR = 0.30 # minimum venture allocation regardless of regime

MIN_TRAIN_MONTHS = 12   # months of history before allocator starts optimizing

# Initial params: contrarian (Buffett-style) — accumulate cash in bull markets,
# deploy aggressively during bear markets / recoveries.
# Approximate starting allocations (softmax of these weights):
#   bull:     ~14% venture, 22% safety,  3% hedge, 61% cash  — wait for the crash
#   sideways: ~28% venture, 23% safety, 10% hedge, 38% cash  — balanced
#   bear:     ~53% venture,  7% safety, 32% hedge,  7% cash  — buy the dip + hedge
# VIX slope 0.3: at peak VIX (40+), shifts ~16% venture → hedge during the worst
# of the crash, then naturally unwinds as VIX calms and the recovery begins.
_INIT = np.array([
    0.0,  0.5, -1.5,  1.5,   # bull:     cash-heavy, some safety
    0.5,  0.3, -0.5,  0.8,   # sideways: balanced, slight cash lean
    1.5, -0.5,  1.0, -0.5,   # bear:     venture + hedge (buy the dip)
    0.3,                       # vix_slope: moderate hedge shift when VIX spikes
])


# --- Parameter encoding ---

def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def _parse_params(flat_params):
    """Decode flat array into allocation dict with regime arrays + vix_slope."""
    return {
        "bull":      _softmax(flat_params[0:4]),
        "sideways":  _softmax(flat_params[4:8]),
        "bear":      _softmax(flat_params[8:12]),
        "vix_slope": float(np.clip(flat_params[12], 0.0, 1.0)),
    }


# --- Allocation computation (vectorized) ---

def _allocate(flat_params, regimes_arr, vix_arr):
    """
    Compute allocation matrix for an array of (regime, vix) observations.

    Returns ndarray of shape (n, 4): columns = [venture, safety, hedge, cash].
    """
    parsed  = _parse_params(flat_params)
    n       = len(regimes_arr)
    result  = np.zeros((n, 4))
    vix_factor = np.clip((vix_arr - VIX_BASE) / (VIX_CAP - VIX_BASE), 0.0, 1.0)

    for regime in REGIMES:
        mask = regimes_arr == regime
        if not mask.any():
            continue
        base  = parsed[regime].copy()                 # shape (4,)
        shift = parsed["vix_slope"] * vix_factor[mask] * base[0]  # shift venture→hedge
        alloc = np.tile(base, (mask.sum(), 1))         # (m, 4)
        alloc[:, 0] -= shift
        alloc[:, 2] += shift
        alloc = np.clip(alloc, 0.0, 1.0)
        alloc /= alloc.sum(axis=1, keepdims=True)
        # Enforce venture floor — take shortfall proportionally from other buckets
        shortfall = np.maximum(VENTURE_FLOOR - alloc[:, 0], 0.0)
        alloc[:, 0] += shortfall
        other_total = alloc[:, 1:].sum(axis=1, keepdims=True)
        scale = np.where(other_total > 0, (other_total - shortfall[:, np.newaxis]) / other_total, 1.0)
        alloc[:, 1:] *= scale
        result[mask] = alloc

    return result


def predict_allocation(flat_params, regime, vix):
    """
    Single-day allocation from trained params.
    Returns dict: venture_pct, safety_pct, hedge_pct, cash_pct.
    """
    alloc = _allocate(flat_params, np.array([regime]), np.array([float(vix)]))[0]
    return {
        "venture_pct": float(alloc[0]),
        "safety_pct":  float(alloc[1]),
        "hedge_pct":   float(alloc[2]),
        "cash_pct":    float(alloc[3]),
    }


# --- Objective function ---

def _neg_sharpe(flat_params, regimes_arr, vix_arr, returns_matrix):
    """
    returns_matrix: (n, 3) — [venture_return, safety_return, hedge_return].
    Cash earns 0%, so the 4th column is omitted.
    """
    allocs  = _allocate(flat_params, regimes_arr, vix_arr)   # (n, 4)
    r       = (allocs[:, :3] * returns_matrix).sum(axis=1)
    std     = r.std()
    if std < 1e-9:
        return 0.0
    return -(r.mean() / std) * np.sqrt(252)


# --- Training ---

def train_allocator(daily_df, init_params=None):
    """
    Fit allocator params on daily_df.

    daily_df must have: regime, vix, venture_return, safety_return, hedge_return.
    Returns flat_params array.
    """
    from scipy.optimize import minimize

    regimes_arr    = daily_df["regime"].values
    vix_arr        = daily_df["vix"].fillna(VIX_BASE).values.astype(float)
    returns_matrix = daily_df[["venture_return", "safety_return", "hedge_return"]].values

    result = minimize(
        _neg_sharpe,
        init_params if init_params is not None else _INIT.copy(),
        args=(regimes_arr, vix_arr, returns_matrix),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-5},
    )
    return result.x


# --- Walk-forward ---

def walk_forward_allocate(daily_df, min_train_months=MIN_TRAIN_MONTHS):
    """
    Walk-forward allocation: for each month, optimize on all preceding months
    (out-of-sample test on the next month). Warm-starts from previous params.

    daily_df must have: date, regime, vix, venture_return, safety_return, hedge_return.

    Returns DataFrame: date, venture_pct, safety_pct, hedge_pct, cash_pct.
    Also returns the final trained params (for deployment).
    """
    daily_df = daily_df.copy()
    daily_df["year_month"] = pd.to_datetime(daily_df["date"]).dt.to_period("M")
    months = sorted(daily_df["year_month"].unique())

    all_rows   = []
    last_params = None

    for i in range(min_train_months, len(months) - 1):
        train_months = months[: i + 1]
        test_month   = months[i + 1]

        train_df = daily_df[daily_df["year_month"].isin(train_months)]
        test_df  = daily_df[daily_df["year_month"] == test_month].copy()

        last_params = train_allocator(train_df, init_params=last_params)
        parsed      = _parse_params(last_params)

        test_regimes = test_df["regime"].values
        test_vix     = test_df["vix"].fillna(VIX_BASE).values.astype(float)
        allocs       = _allocate(last_params, test_regimes, test_vix)

        for j, (_, row) in enumerate(test_df.iterrows()):
            all_rows.append({
                "date":        row["date"],
                "venture_pct": float(allocs[j, 0]),
                "safety_pct":  float(allocs[j, 1]),
                "hedge_pct":   float(allocs[j, 2]),
                "cash_pct":    float(allocs[j, 3]),
            })

        regime_dist = test_df["regime"].value_counts().to_dict()
        mean_alloc  = allocs.mean(axis=0)
        print(
            f"  {test_month}  regime={regime_dist}  "
            f"venture={mean_alloc[0]:.0%}  safety={mean_alloc[1]:.0%}  "
            f"hedge={mean_alloc[2]:.0%}  cash={mean_alloc[3]:.0%}",
            flush=True,
        )

    # Train final params on all data (for deployment)
    final_params = train_allocator(daily_df, init_params=last_params)

    return pd.DataFrame(all_rows), final_params


# --- Market data ---

def fetch_bucket_returns(start, end):
    """
    Fetch daily returns for safety + hedge ETFs and VIX level.
    Returns DataFrame: date, safety_return, hedge_return, vix.
    Missing ETF history (e.g. GLDM before 2018) is filled with 0.0.
    """
    import yfinance as yf

    tickers  = SAFETY_ETFS + HEDGE_ETFS + ["^VIX"]
    raw      = yf.download(tickers, start=start, end=end,
                           auto_adjust=True, progress=False, group_by="ticker")

    def _returns(symbols):
        cols = []
        for sym in symbols:
            try:
                close = raw[sym]["Close"].squeeze()
                cols.append(close.pct_change())
            except Exception:
                cols.append(pd.Series(dtype=float))
        if not cols:
            return pd.Series(0.0, index=raw.index)
        df = pd.concat(cols, axis=1)
        return df.mean(axis=1)

    safety_ret = _returns(SAFETY_ETFS)
    hedge_ret  = _returns(HEDGE_ETFS)

    try:
        vix = raw["^VIX"]["Close"].squeeze()
    except Exception:
        vix = pd.Series(VIX_BASE, index=raw.index)

    out = pd.DataFrame({
        "date":          pd.to_datetime(raw.index).normalize(),
        "safety_return": safety_ret.values,
        "hedge_return":  hedge_ret.values,
        "vix":           vix.values,
    }).dropna(subset=["date"])
    out["safety_return"] = out["safety_return"].fillna(0.0)
    out["hedge_return"]  = out["hedge_return"].fillna(0.0)
    out["vix"]           = out["vix"].fillna(VIX_BASE)
    return out.reset_index(drop=True)


def build_allocator_df(venture_results_df, bucket_returns_df, spy_regimes):
    """
    Combine venture daily returns, bucket returns, and regimes into one DataFrame
    ready for walk_forward_allocate().

    venture_results_df : output of run_backtest() — has date, return columns.
    bucket_returns_df  : output of fetch_bucket_returns().
    spy_regimes        : Series indexed by date, values 'bull'/'bear'/'sideways'.
    """
    df = venture_results_df[["date", "return"]].rename(columns={"return": "venture_return"})
    df = df.merge(bucket_returns_df, on="date", how="left")
    df["safety_return"] = df["safety_return"].fillna(0.0)
    df["hedge_return"]  = df["hedge_return"].fillna(0.0)
    df["vix"]           = df["vix"].fillna(VIX_BASE)

    regime_series = spy_regimes.reset_index()
    regime_series.columns = ["date", "regime"]
    regime_series["date"] = pd.to_datetime(regime_series["date"]).dt.normalize()
    df = df.merge(regime_series, on="date", how="left")
    df["regime"] = df["regime"].fillna("sideways")

    return df.dropna(subset=["venture_return"]).reset_index(drop=True)