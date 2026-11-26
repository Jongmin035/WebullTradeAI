"""
Offline experiment: does a longer prediction horizon help, when paired with
an exit rule that's actually consistent with it?

Motivation: the production model trains clf_df/reg_df with horizon=5 (predict
5 trading days ahead), but EVERY exit path in the live bot (target_w hitting
0, the conviction-flip trigger) re-evaluates that same horizon fresh every
single day. A stock bought on a 5-day thesis can be sold the very next day if
today's freshly recomputed 5-day forecast flips -- the thesis never gets time
to play out. Lengthening the horizon to reduce training noise while leaving
that same daily-recompute-driven exit logic in place would make the mismatch
*worse*, not better: longer bets, same noise-sensitive trigger pulling out of
them early.

This script tests the paired change: a longer horizon (default 10 trading
days) AND a backtest exit rule that gives positions a minimum holding period
before the routine "model doesn't want this anymore" sell applies -- while
keeping a fast, price-based stop-loss (not forecast-based) as the override
that can still exit early regardless of the holding period, mirroring the
ATR trailing stop that already governs live exits independent of the model's
forecast horizon.

actual_df (horizon=1, the realistic daily P&L series) is unchanged -- it
reflects what the bot actually captures day to day, not what it's trained to
predict, so it has to stay fixed for a fair comparison against the horizon=5
baseline.

Usage:
    python src/aws/experiment_horizon.py --horizon 10 --min-hold-days 10 --stop-loss-pct 0.08
    python src/aws/experiment_horizon.py --horizon 10 --min-hold-days 0   # isolate horizon alone,
                                                                           # old exit rule, for comparison

Does NOT touch production retrain.py, trader.py, or any live model/config.
Results upload to s3://{bucket}/retrain/eval/experiment-h{horizon}-mh{min_hold_days}-{date}-*,
a separate prefix from the weekly production eval files.
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

HERE     = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.dirname(HERE)
for d in (SRC_DIR, os.path.join(SRC_DIR, "core"), os.path.join(SRC_DIR, "pipeline"), os.path.join(SRC_DIR, "models")):
    if d not in sys.path:
        sys.path.insert(0, d)

from data_pipeline import load_all_symbols, fetch_sp500_symbols, ETF_SYMBOLS
from metrics import kelly_fraction, sharpe_ratio, sortino_ratio
from retrain import (
    download_data_from_s3, eval_lstm, ALL_YEARS,
)

BUCKET = os.getenv("AWS_S3_BUCKET", "webull-trade-ai")


def run_backtest_min_hold(predictions_df, min_hold_days, stop_loss_pct,
                           half_kelly=True, rebalance_threshold=0.02, transaction_cost=0.0003):
    """
    Same mechanics as backtest.run_backtest, plus:
      - min_hold_days: a position the model no longer wants isn't sold via the
        routine selection path until this many trading days have elapsed since
        entry. 0 reproduces the original (no minimum hold) behavior exactly.
      - stop_loss_pct: a price-based override -- if cumulative return since
        entry drops below -stop_loss_pct, exit immediately regardless of the
        minimum hold. Proxies the live ATR trailing stop using actual_return,
        since this offline eval has no raw OHLC series to compute ATR from.
    """
    all_dates = sorted(predictions_df["date"].unique())
    holdings, entry_day, cum_since_entry = {}, {}, {}
    results = []

    for day_idx, date in enumerate(all_dates):
        day = predictions_df[predictions_df["date"] == date]
        candidates = day[(day["clf_prob"] > 0.5) & (day["reg_pred"] > 0)].copy()
        candidates["kelly"] = candidates.apply(
            lambda r: kelly_fraction(r["clf_prob"], r["reg_pred"], half_kelly=half_kelly), axis=1
        )
        candidates = candidates[candidates["kelly"] > 0]

        target = {}
        if not candidates.empty:
            total_kelly = candidates["kelly"].sum()
            for _, row in candidates.iterrows():
                target[row["symbol"]] = row["kelly"] / max(1.0, total_kelly)

        actual_returns = day.set_index("symbol")["actual_return"]
        forced_exits = {s for s in holdings if cum_since_entry.get(s, 0.0) <= -stop_loss_pct}

        all_symbols  = set(holdings.keys()) | set(target.keys())
        trade_cost   = 0.0
        new_holdings = dict(holdings)

        for s in all_symbols:
            current_w = holdings.get(s, 0.0)
            target_w  = target.get(s, 0.0)

            if s in forced_exits:
                if current_w > 0:
                    trade_cost += current_w * transaction_cost
                new_holdings[s] = 0.0
                continue

            if current_w > 0 and target_w == 0.0:
                held_days = day_idx - entry_day.get(s, day_idx)
                if held_days < min_hold_days:
                    new_holdings[s] = current_w   # hold -- model's routine drop-out is ignored early
                    continue

            delta = target_w - current_w
            if abs(delta) > rebalance_threshold:
                new_holdings[s] = target_w
                trade_cost += abs(delta) * transaction_cost
                if target_w > 0 and current_w == 0:
                    entry_day[s]        = day_idx
                    cum_since_entry[s]  = 0.0

        holdings = {s: w for s, w in new_holdings.items() if w > 1e-4}
        for s in list(cum_since_entry):
            if s not in holdings:
                del cum_since_entry[s]
                entry_day.pop(s, None)

        gross_return = sum(w * actual_returns.get(s, 0.0) for s, w in holdings.items())
        daily_return = gross_return - trade_cost

        if holdings:
            portfolio_factor = 1.0 + gross_return
            new_h = {}
            for s, w in holdings.items():
                r = actual_returns.get(s, 0.0)
                cum_since_entry[s] = (1 + cum_since_entry.get(s, 0.0)) * (1 + r) - 1
                new_h[s] = w * (1.0 + r) / portfolio_factor if portfolio_factor > 0 else w
            holdings = {s: w for s, w in new_h.items() if w > 1e-4}

        results.append({"date": date, "return": daily_return, "n_stocks": len(holdings)})

    results_df = pd.DataFrame(results).sort_values("date").reset_index(drop=True)
    returns    = results_df["return"].values
    return {
        "cumulative_return": float((1 + pd.Series(returns)).prod() - 1),
        "sharpe":            float(sharpe_ratio(returns, periods_per_year=252)),
        "sortino":           float(sortino_ratio(returns, periods_per_year=252)),
        "win_rate":          float((returns > 0).mean()),
        "results_df":        results_df,
    }


def compute_turnover(predictions_df, top_n=30):
    """
    Day-over-day overlap of the top-N picks by clf_prob, same metric used in
    the September investigation. Low overlap = high noise-driven turnover in
    what the model is picking, independent of how the backtest actually trades it.
    """
    by_date = {}
    for date, group in predictions_df.groupby("date"):
        by_date[date] = set(group.nlargest(top_n, "clf_prob")["symbol"])

    dates = sorted(by_date.keys())
    overlaps = []
    for i in range(1, len(dates)):
        a, b = by_date[dates[i - 1]], by_date[dates[i]]
        overlaps.append(len(a & b) / top_n)
    return {
        "mean_overlap_fraction": float(pd.Series(overlaps).mean()) if overlaps else None,
        "n_day_pairs": len(overlaps),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=10, help="Forward-looking label horizon in trading days (default 10)")
    parser.add_argument("--min-hold-days", type=int, default=10, help="Minimum days before a routine model-driven sell applies (0 = original behavior)")
    parser.add_argument("--stop-loss-pct", type=float, default=0.08, help="Price-based stop-loss override, as a fraction (default 0.08 = 8%%, matching live ATR_FALLBACK)")
    parser.add_argument("--tag", default="", help="Extra suffix for the S3 result tag, so a repeat run doesn't overwrite a prior one on the same day")
    args = parser.parse_args()

    print(f"=== Horizon experiment: horizon={args.horizon}  min_hold_days={args.min_hold_days}  stop_loss_pct={args.stop_loss_pct} ===")

    download_data_from_s3()
    symbols = list(set(fetch_sp500_symbols()) | set(ETF_SYMBOLS))

    clf_df    = load_all_symbols(symbols, target_type="classification", horizon=args.horizon)
    reg_df    = load_all_symbols(symbols, target_type="regression",     horizon=args.horizon)
    actual_df = load_all_symbols(symbols, target_type="regression",     horizon=1)   # unchanged -- real daily P&L
    clf_df    = clf_df[clf_df["date"].dt.year.isin(ALL_YEARS)].reset_index(drop=True)
    reg_df    = reg_df[reg_df["date"].dt.year.isin(ALL_YEARS)].reset_index(drop=True)
    actual_df = actual_df[actual_df["date"].dt.year.isin(ALL_YEARS)].reset_index(drop=True)
    for df in (clf_df, reg_df, actual_df):
        df["year_month"] = df["date"].dt.to_period("M")

    predictions, _, _ = eval_lstm(clf_df, reg_df, actual_df)

    turnover = compute_turnover(predictions)
    print(f"Turnover (top-30 day-over-day overlap): {turnover['mean_overlap_fraction']:.3f}  "
          f"over {turnover['n_day_pairs']} day-pairs")

    metrics = run_backtest_min_hold(
        predictions, min_hold_days=args.min_hold_days, stop_loss_pct=args.stop_loss_pct
    )
    print(f"Sharpe={metrics['sharpe']:.4f}  Sortino={metrics['sortino']:.4f}  "
          f"CumReturn={metrics['cumulative_return']:.2%}  WinRate={metrics['win_rate']:.2%}")

    today_str = datetime.now().strftime("%Y-%m-%d")
    tag_suffix = f"-{args.tag}" if args.tag else ""
    tag = f"experiment-h{args.horizon}-mh{args.min_hold_days}-{today_str}{tag_suffix}"
    try:
        import boto3
        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
        metrics["results_df"].to_csv(f"/tmp/{tag}-results.csv", index=False)
        predictions.to_parquet(f"/tmp/{tag}-predictions.parquet", index=False)
        s3.upload_file(f"/tmp/{tag}-results.csv", BUCKET, f"retrain/eval/{tag}-results.csv")
        s3.upload_file(f"/tmp/{tag}-predictions.parquet", BUCKET, f"retrain/eval/{tag}-predictions.parquet")
        import json
        summary = {
            "horizon": args.horizon, "min_hold_days": args.min_hold_days, "stop_loss_pct": args.stop_loss_pct,
            "sharpe": metrics["sharpe"], "sortino": metrics["sortino"],
            "cumulative_return": metrics["cumulative_return"], "win_rate": metrics["win_rate"],
            "turnover_mean_overlap_fraction": turnover["mean_overlap_fraction"],
        }
        s3.put_object(Bucket=BUCKET, Key=f"retrain/eval/{tag}-summary.json", Body=json.dumps(summary, indent=2))
        print(f"Uploaded to s3://{BUCKET}/retrain/eval/{tag}-*")
    except Exception as e:
        print(f"Upload failed (results still printed above): {e}")
