"""
Offline experiment: does seed-averaged ensembling tame the training-
randomness variance we measured directly?

A horizon=5 repeat run -- same config, same data, nothing changed but random
initialization -- swung Sharpe 0.43 -> 1.70 and Sortino 0.76 -> 3.13. That's
larger than the entire difference previously observed between horizon=5 and
horizon=10, meaning no single eval_lstm run can be trusted in isolation for
any comparison.

This trains N models per walk-forward month (different seeds, same
architecture/data/horizon) and averages their predictions before backtesting
-- the standard fix for exactly this kind of variance. Same walk-forward
structure as retrain.eval_lstm, same horizon=5 (matching production) and the
standard exit rule, so this isolates "does ensembling help" from the other
open questions (horizon length, model class).

Cost: roughly N times eval_lstm's own cost (~45 min x N on GPU).

Usage:
    python src/aws/experiment_ensemble.py --n-seeds 5
    python src/aws/experiment_ensemble.py --n-seeds 3   # cheaper, still meaningfully reduces variance

Does NOT touch production retrain.py/trader.py or any live model/config
(it only reuses _train_lstm/_predict_lstm as library functions). Results
upload to s3://{bucket}/retrain/eval/experiment-ensemble{n}-{date}-*.
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE     = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.dirname(HERE)
for d in (SRC_DIR, os.path.join(SRC_DIR, "core"), os.path.join(SRC_DIR, "pipeline"), os.path.join(SRC_DIR, "models")):
    if d not in sys.path:
        sys.path.insert(0, d)

from data_pipeline import load_all_symbols, fetch_sp500_symbols, ETF_SYMBOLS
from backtest import run_backtest
from retrain import (
    download_data_from_s3, _make_sequences, _train_lstm, _predict_lstm,
    ALL_YEARS, EVAL_MONTHS, MIN_TRAIN_MONTHS, LSTM_TRAIN_WINDOW, TRAIN_SEED,
)

BUCKET = os.getenv("AWS_S3_BUCKET", "webull-trade-ai")


def eval_lstm_ensemble(clf_df, reg_df, actual_df, n_seeds, base_seed=TRAIN_SEED):
    clf_df["year_month"]    = clf_df["date"].dt.to_period("M")
    reg_df["year_month"]    = reg_df["date"].dt.to_period("M")
    actual_df["year_month"] = actual_df["date"].dt.to_period("M")

    all_months = sorted(clf_df["year_month"].unique())
    eval_set   = set(all_months[-EVAL_MONTHS:])
    months     = all_months
    all_preds  = []

    for i in range(MIN_TRAIN_MONTHS, len(months) - 1):
        test_month = months[i + 1]
        if test_month not in eval_set:
            continue

        train_months  = months[:i + 1]
        window_months = train_months[-LSTM_TRAIN_WINDOW:]
        clf_tr_flat  = clf_df[clf_df["year_month"].isin(window_months)]
        clf_te_flat  = clf_df[clf_df["year_month"] == test_month]
        reg_tr_flat  = reg_df[reg_df["year_month"].isin(window_months)]
        reg_te_flat  = reg_df[reg_df["year_month"] == test_month]
        if len(clf_te_flat) == 0:
            continue

        clf_comb = pd.concat([clf_tr_flat, clf_te_flat]).sort_values(["symbol", "date"]).reset_index(drop=True)
        clf_comb["year_month"] = clf_comb["date"].dt.to_period("M")
        X_clf, y_clf, meta_clf = _make_sequences(clf_comb)
        tr_m = meta_clf["year_month"].isin(train_months).values
        te_m = (meta_clf["year_month"] == test_month).values
        if tr_m.sum() == 0 or te_m.sum() == 0:
            continue

        reg_comb = pd.concat([reg_tr_flat, reg_te_flat]).sort_values(["symbol", "date"]).reset_index(drop=True)
        reg_comb["year_month"] = reg_comb["date"].dt.to_period("M")
        X_reg, y_reg, meta_reg = _make_sequences(reg_comb)
        tr_m_r = meta_reg["year_month"].isin(train_months).values
        te_m_r = (meta_reg["year_month"] == test_month).values

        # Train n_seeds independent models per task, average their predictions --
        # this is the actual ensembling step.
        clf_prob_runs, reg_pred_runs = [], []
        for seed_offset in range(n_seeds):
            seed = base_seed + seed_offset
            clf_model, clf_scaler = _train_lstm(X_clf[tr_m], y_clf[tr_m], clf_tr_flat, "clf", seed=seed)
            clf_prob_runs.append(_predict_lstm(clf_model, clf_scaler, X_clf[te_m], "clf"))
            reg_model, reg_scaler = _train_lstm(X_reg[tr_m_r], y_reg[tr_m_r], reg_tr_flat, "reg", seed=seed)
            reg_pred_runs.append(_predict_lstm(reg_model, reg_scaler, X_reg[te_m_r], "reg"))

        clf_prob = np.mean(clf_prob_runs, axis=0)
        reg_pred = np.mean(reg_pred_runs, axis=0)

        preds_clf = meta_clf[te_m][["date", "symbol"]].copy().reset_index(drop=True)
        preds_clf["clf_prob"] = clf_prob
        preds_reg = meta_reg[te_m_r][["date", "symbol"]].copy().reset_index(drop=True)
        preds_reg["reg_pred"] = reg_pred
        preds = preds_clf.merge(preds_reg, on=["date", "symbol"], how="inner")

        actual = (
            actual_df[actual_df["year_month"] == test_month][["date", "symbol", "target"]]
            .rename(columns={"target": "actual_return"})
        )
        preds = preds.merge(actual, on=["date", "symbol"], how="inner")
        if len(preds) == 0:
            continue

        all_preds.append(preds)
        print(f"  ensemble({n_seeds}) {test_month}  train_seq={tr_m.sum():,}  test_seq={te_m.sum():,}")

    predictions = pd.concat(all_preds, ignore_index=True)
    metrics     = run_backtest(predictions, plot=False)
    return predictions, metrics


def compute_turnover(predictions_df, top_n=30):
    by_date = {}
    for date, group in predictions_df.groupby("date"):
        by_date[date] = set(group.nlargest(top_n, "clf_prob")["symbol"])
    dates = sorted(by_date.keys())
    overlaps = [len(by_date[dates[i - 1]] & by_date[dates[i]]) / top_n for i in range(1, len(dates))]
    return {
        "mean_overlap_fraction": float(pd.Series(overlaps).mean()) if overlaps else None,
        "n_day_pairs": len(overlaps),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of independently-seeded models to average per walk-forward month")
    parser.add_argument("--horizon", type=int, default=5, help="Label horizon, default 5 to match production")
    parser.add_argument("--tag", default="", help="Extra suffix for the S3 result tag")
    args = parser.parse_args()

    print(f"=== Ensemble experiment: n_seeds={args.n_seeds}  horizon={args.horizon} ===")

    download_data_from_s3()
    symbols = list(set(fetch_sp500_symbols()) | set(ETF_SYMBOLS))

    clf_df    = load_all_symbols(symbols, target_type="classification", horizon=args.horizon)
    reg_df    = load_all_symbols(symbols, target_type="regression",     horizon=args.horizon)
    actual_df = load_all_symbols(symbols, target_type="regression",     horizon=1)
    clf_df    = clf_df[clf_df["date"].dt.year.isin(ALL_YEARS)].reset_index(drop=True)
    reg_df    = reg_df[reg_df["date"].dt.year.isin(ALL_YEARS)].reset_index(drop=True)
    actual_df = actual_df[actual_df["date"].dt.year.isin(ALL_YEARS)].reset_index(drop=True)

    predictions, metrics = eval_lstm_ensemble(clf_df, reg_df, actual_df, n_seeds=args.n_seeds)

    turnover = compute_turnover(predictions)
    print(f"Turnover (top-30 day-over-day overlap): {turnover['mean_overlap_fraction']:.3f}  "
          f"over {turnover['n_day_pairs']} day-pairs")
    print(f"Sharpe={metrics['sharpe']:.4f}  Sortino={metrics['sortino']:.4f}  "
          f"CumReturn={metrics['cumulative_return']:.2%}  WinRate={metrics['win_rate']:.2%}")

    today_str = datetime.now().strftime("%Y-%m-%d")
    tag_suffix = f"-{args.tag}" if args.tag else ""
    tag = f"experiment-ensemble{args.n_seeds}-h{args.horizon}-{today_str}{tag_suffix}"
    try:
        import boto3, json
        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
        metrics["results_df"].to_csv(f"/tmp/{tag}-results.csv", index=False)
        predictions.to_parquet(f"/tmp/{tag}-predictions.parquet", index=False)
        s3.upload_file(f"/tmp/{tag}-results.csv", BUCKET, f"retrain/eval/{tag}-results.csv")
        s3.upload_file(f"/tmp/{tag}-predictions.parquet", BUCKET, f"retrain/eval/{tag}-predictions.parquet")
        summary = {
            "n_seeds": args.n_seeds, "horizon": args.horizon,
            "sharpe": metrics["sharpe"], "sortino": metrics["sortino"],
            "cumulative_return": metrics["cumulative_return"], "win_rate": metrics["win_rate"],
            "turnover_mean_overlap_fraction": turnover["mean_overlap_fraction"],
        }
        s3.put_object(Bucket=BUCKET, Key=f"retrain/eval/{tag}-summary.json", Body=json.dumps(summary, indent=2))
        print(f"Uploaded to s3://{BUCKET}/retrain/eval/{tag}-*")
    except Exception as e:
        print(f"Upload failed (results still printed above): {e}")
