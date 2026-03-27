import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import sharpe_ratio


def run_backtest(predictions_df, top_k=1, horizon=5, plot=True):
    """
    Simulate a weekly trading strategy using classifier + regressor predictions.

    Parameters
    ----------
    predictions_df : DataFrame with columns:
        - date         : the prediction date (start of holding period)
        - symbol       : stock symbol
        - clf_prob     : classifier probability of going up (0-1)
        - reg_pred     : regressor predicted 5-day return
        - actual_return: actual 5-day return that occurred
    top_k   : number of stocks to buy each period (highest reg_pred among clf-filtered)
    horizon : holding period in trading days — trades every N days to avoid overlap
    plot    : whether to plot the equity curve

    Returns
    -------
    dict with cumulative_return, sharpe, win_rate, n_trades
    """
    # Sample trade entry dates every `horizon` days to avoid overlapping positions
    all_dates = sorted(predictions_df["date"].unique())
    entry_dates = all_dates[::horizon]
    predictions_df = predictions_df[predictions_df["date"].isin(entry_dates)]

    results = []

    for date, group in predictions_df.groupby("date"):
        # Step 1: filter to stocks the classifier thinks will go up
        candidates = group[group["clf_prob"] >= 0.5].copy()

        if candidates.empty:
            continue

        # Step 2: rank by regressor predicted return, pick top K
        candidates = candidates.sort_values("reg_pred", ascending=False)
        selected = candidates.head(top_k)

        # Step 3: record the actual return (equal weight across selected stocks)
        period_return = selected["actual_return"].mean()
        results.append({"date": date, "return": period_return, "n_stocks": len(selected)})

    if not results:
        print("No trades executed.")
        return {}

    results_df = pd.DataFrame(results).sort_values("date").reset_index(drop=True)

    # --- Metrics ---
    returns = results_df["return"].values
    cum_return   = (1 + pd.Series(returns)).prod() - 1
    sharpe       = sharpe_ratio(returns, periods_per_year=52)  # weekly periods
    win_rate     = np.mean(returns > 0)
    n_trades     = len(results_df)

    print(f"Trades executed : {n_trades}")
    print(f"Win rate        : {win_rate:.2%}")
    print(f"Cumulative return: {cum_return:.2%}")
    print(f"Sharpe ratio    : {sharpe:.4f}")

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
        "win_rate": win_rate,
        "n_trades": n_trades,
        "results_df": results_df,
    }
