import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _d in (_src, os.path.join(_src, "pipeline")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from metrics import sharpe_ratio, sortino_ratio, kelly_fraction


def run_backtest(predictions_df, half_kelly=True, rebalance_threshold=0.02,
                 transaction_cost=0.0003, plot=True):
    """
    Target-weight rebalancing backtest with Kelly position sizing.

    Each day:
      1. Compute target weights from model predictions:
             score  = clf_prob * reg_pred  (confidence x magnitude)
             kelly  = kelly_fraction(clf_prob, reg_pred)
             target = kelly / max(1, sum(kelly))  -- never exceed 100% invested
      2. Rebalance: only trade stocks where |target - current| > rebalance_threshold
             -- avoids noise-driven churn; positions are held until signal meaningfully changes
      3. Deduct transaction cost on each traded dollar amount
      4. Earn actual_return on post-rebalance holdings
      5. Update weights for price drift (stocks that went up now occupy a larger fraction)

    Parameters
    ----------
    predictions_df      : DataFrame — date, symbol, clf_prob, reg_pred, actual_return
    half_kelly          : bool  — half-Kelly sizing (default True, recommended)
    rebalance_threshold : float — minimum |target - current| weight change to trigger a trade
                          Default 0.02 (2%): calibrated for Webull commission-free large-caps
                          where bid-ask round-trip is ~0.03%, so a 2% trade costs ~1.5% of its value
    transaction_cost    : float — one-way cost per unit traded (default 0.0003 = 0.03%)
                          Covers bid-ask spread on liquid large-cap stocks; Webull has no commission
    plot                : bool  — show equity curve when done

    Returns
    -------
    dict with cumulative_return, sharpe, win_rate, n_days_invested, results_df
    """
    all_dates = sorted(predictions_df["date"].unique())
    holdings  = {}   # {symbol: current portfolio weight (fraction of total value)}
    results   = []

    for date in all_dates:
        day = predictions_df[predictions_df["date"] == date]

        # --- Step 1: Compute target weights ---
        # Score combines direction confidence (clf_prob) and expected magnitude (reg_pred)
        candidates = day[(day["clf_prob"] > 0.5) & (day["reg_pred"] > 0)].copy()
        candidates["kelly"] = candidates.apply(
            lambda r: kelly_fraction(r["clf_prob"], r["reg_pred"], half_kelly=half_kelly),
            axis=1
        )
        candidates = candidates[candidates["kelly"] > 0]

        target = {}
        if not candidates.empty:
            total_kelly = candidates["kelly"].sum()
            for _, row in candidates.iterrows():
                # Normalize if total Kelly > 100% so we never over-invest
                target[row["symbol"]] = row["kelly"] / max(1.0, total_kelly)

        # --- Step 2: Rebalance — only trade if drift exceeds threshold ---
        all_symbols = set(holdings.keys()) | set(target.keys())
        trade_cost  = 0.0
        new_holdings = dict(holdings)

        for s in all_symbols:
            current_w = holdings.get(s, 0.0)
            target_w  = target.get(s, 0.0)
            delta = target_w - current_w

            if abs(delta) > rebalance_threshold:
                new_holdings[s] = target_w
                trade_cost += abs(delta) * transaction_cost

        # Drop positions that became negligibly small
        holdings = {s: w for s, w in new_holdings.items() if w > 1e-4}

        # --- Step 3: Earn actual returns on post-rebalance holdings ---
        actual_returns = day.set_index("symbol")["actual_return"]
        gross_return = sum(
            w * actual_returns.get(s, 0.0) for s, w in holdings.items()
        )
        daily_return = gross_return - trade_cost

        # --- Step 4: Update weights for price drift ---
        # Stocks that went up now represent a larger share of the portfolio
        if holdings:
            portfolio_factor = 1.0 + gross_return
            if portfolio_factor > 0:
                holdings = {
                    s: w * (1.0 + actual_returns.get(s, 0.0)) / portfolio_factor
                    for s, w in holdings.items()
                }
            holdings = {s: w for s, w in holdings.items() if w > 1e-4}

        results.append({"date": date, "return": daily_return, "n_stocks": len(holdings)})

    if not results:
        print("No trades executed.")
        return {}

    results_df = pd.DataFrame(results).sort_values("date").reset_index(drop=True)

    # --- Metrics ---
    returns         = results_df["return"].values
    cum_return      = (1 + pd.Series(returns)).prod() - 1
    sharpe          = sharpe_ratio(returns, periods_per_year=252)
    sortino         = sortino_ratio(returns, periods_per_year=252)
    win_rate        = np.mean(returns > 0)
    n_days_invested = int((results_df["n_stocks"] > 0).sum())

    print(f"Days simulated   : {len(results_df)}")
    print(f"Days invested    : {n_days_invested}")
    print(f"Win rate         : {win_rate:.2%}")
    print(f"Cumulative return: {cum_return:.2%}")
    print(f"Sharpe ratio     : {sharpe:.4f}")
    print(f"Sortino ratio    : {sortino:.4f}")

    if plot:
        equity_curve = (1 + pd.Series(returns)).cumprod()
        plt.figure(figsize=(12, 4))
        plt.plot(results_df["date"], equity_curve, label="Strategy")
        plt.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
        plt.title("Equity Curve")
        plt.ylabel("Portfolio Value (starting at 1.0)")
        plt.xlabel("Date")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    return {
        "cumulative_return": cum_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "win_rate": win_rate,
        "n_days_invested": n_days_invested,
        "results_df": results_df,
    }


def run_bucket_backtest(venture_results_df, allocation_df, bucket_returns_df, plot=True):
    """
    Evaluate the full portfolio (venture + safety + hedge + cash) using
    pre-computed venture daily returns and walk-forward allocations.

    Parameters
    ----------
    venture_results_df : DataFrame — date, return  (output of run_backtest)
    allocation_df      : DataFrame — date, venture_pct, safety_pct, hedge_pct, cash_pct
    bucket_returns_df  : DataFrame — date, safety_return, hedge_return
    """
    df = venture_results_df[["date", "return"]].rename(columns={"return": "venture_return"})
    df = df.merge(allocation_df, on="date", how="inner")
    df = df.merge(bucket_returns_df[["date", "safety_return", "hedge_return"]], on="date", how="left")
    df["safety_return"] = df["safety_return"].fillna(0.0)
    df["hedge_return"]  = df["hedge_return"].fillna(0.0)

    returns = (
        df["venture_pct"] * df["venture_return"] +
        df["safety_pct"]  * df["safety_return"]  +
        df["hedge_pct"]   * df["hedge_return"]
        # cash_pct earns 0%
    ).values

    cum_return = (1 + pd.Series(returns)).prod() - 1
    sharpe     = sharpe_ratio(returns, periods_per_year=252)
    win_rate   = np.mean(returns > 0)

    print(f"Days simulated   : {len(df)}")
    print(f"Win rate         : {win_rate:.2%}")
    print(f"Cumulative return: {cum_return:.2%}")
    print(f"Sharpe ratio     : {sharpe:.4f}")

    if plot:
        equity_curve = (1 + pd.Series(returns)).cumprod()
        plt.figure(figsize=(12, 4))
        plt.plot(df["date"].values, equity_curve, label="Bucket Strategy")
        plt.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
        plt.title("Equity Curve (Bucket Portfolio)")
        plt.ylabel("Portfolio Value (starting at 1.0)")
        plt.xlabel("Date")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    return {
        "cumulative_return": cum_return,
        "sharpe":   sharpe,
        "win_rate": win_rate,
        "results_df": df.assign(Return=returns),
    }
