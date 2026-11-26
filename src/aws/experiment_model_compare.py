"""
Offline experiment: LightGBM / ElasticNet vs. the production LSTM.

Same walk-forward structure as retrain.eval_lstm -- train on a rolling window
of prior months (same window length as production, LSTM_TRAIN_WINDOW), test
on the next month, repeat across the last EVAL_MONTHS. Unlike the horizon
experiment, this one changes ONLY the model type: horizon=5 (matching
production) and the standard (unmodified) run_backtest exit rule, so the
comparison isolates "does model choice matter" from the other open questions.

Why these two:
  - lightgbm: the real "is a tree ensemble better suited to this data"
    test -- same family as XGBoost, generally faster/slightly stronger in
    current practice on large tabular datasets like this one. Tree splits
    are discrete thresholds, not continuous nonlinear transforms stacked
    across time steps, so they're structurally less prone to the kind of
    day-to-day prediction flips the LSTM showed in the September
    investigation (17-30 of 30 top picks changing with no retrain in
    between).
  - elasticnet: a near-free complexity sanity check. If a plain regularized
    linear model performs close to the LSTM, that's a real signal this
    problem's signal-to-noise ratio doesn't reward model complexity at all.

Neither needs FEATURES built into sequences (_make_sequences/SEQ_LEN) --
both train directly on the flat per (date, symbol) feature rows, which also
means they don't lose each symbol's first SEQ_LEN rows per window the way
the LSTM does.

Usage:
    python src/aws/experiment_model_compare.py --model lightgbm
    python src/aws/experiment_model_compare.py --model elasticnet

Does NOT touch production retrain.py/trader.py or any live model/config.
Results upload to s3://{bucket}/retrain/eval/experiment-{model}-{date}-*.
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
from regime_pipeline import FEATURES
from backtest import run_backtest
from retrain import (
    download_data_from_s3, ALL_YEARS, EVAL_MONTHS, MIN_TRAIN_MONTHS, LSTM_TRAIN_WINDOW,
)

BUCKET = os.getenv("AWS_S3_BUCKET", "webull-trade-ai")


RANDOM_STATE = 42  # same reasoning as retrain.TRAIN_SEED -- makes repeat/comparison runs
                    # reproducible instead of confounding real effects with run-to-run noise


def _fit_predict_lightgbm(X_tr, y_tr, X_te, task):
    import lightgbm as lgb
    # First attempt (num_leaves=31, default min_child_samples=20) produced
    # clf_prob clustered tightly around the base rate (mean 0.57, std 0.068,
    # 82% of rows passing both candidate filters) -- too conservative/
    # averaging to discriminate between stocks, not an honest reflection of
    # weak-but-real signal. More leaves + a lower min_child_samples lets the
    # model carve out sharper, more specific splits instead of regressing
    # everything toward the mean.
    common = dict(n_estimators=300, learning_rate=0.05, num_leaves=63,
                   min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                   verbosity=-1, random_state=RANDOM_STATE)
    if task == "clf":
        model = lgb.LGBMClassifier(**common)
        model.fit(X_tr, y_tr)
        return model.predict_proba(X_te)[:, 1]
    else:
        model = lgb.LGBMRegressor(**common)
        model.fit(X_tr, y_tr)
        return model.predict(X_te)


def _fit_predict_elasticnet(X_tr, y_tr, X_te, task):
    from sklearn.linear_model import LogisticRegression, ElasticNet
    from sklearn.preprocessing import StandardScaler
    # Linear models need scaled features; trees don't, which is why this
    # helper (unlike the lightgbm one) scales internally.
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    if task == "clf":
        model = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                    C=1.0, max_iter=2000, random_state=RANDOM_STATE)
        model.fit(X_tr_s, y_tr)
        return model.predict_proba(X_te_s)[:, 1]
    else:
        model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=RANDOM_STATE)
        model.fit(X_tr_s, y_tr)
        return model.predict(X_te_s)


FIT_PREDICT = {"lightgbm": _fit_predict_lightgbm, "elasticnet": _fit_predict_elasticnet}


def eval_tabular_model(clf_df, reg_df, actual_df, model_type):
    fit_predict = FIT_PREDICT[model_type]

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
        window_months = train_months[-LSTM_TRAIN_WINDOW:]   # same cap as production, for a fair comparison

        clf_tr = clf_df[clf_df["year_month"].isin(window_months)]
        clf_te = clf_df[clf_df["year_month"] == test_month]
        reg_tr = reg_df[reg_df["year_month"].isin(window_months)]
        reg_te = reg_df[reg_df["year_month"] == test_month]
        if len(clf_te) == 0 or len(reg_te) == 0:
            continue

        clf_prob = fit_predict(clf_tr[FEATURES].values, clf_tr["target"].values, clf_te[FEATURES].values, "clf")
        reg_pred = fit_predict(reg_tr[FEATURES].values, reg_tr["target"].values, reg_te[FEATURES].values, "reg")

        preds_clf = clf_te[["date", "symbol"]].copy().reset_index(drop=True)
        preds_clf["clf_prob"] = clf_prob
        preds_reg = reg_te[["date", "symbol"]].copy().reset_index(drop=True)
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
        print(f"  {model_type} {test_month}  train_rows={len(clf_tr):,}  test_rows={len(clf_te):,}")

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
    parser.add_argument("--model", choices=["lightgbm", "elasticnet"], required=True)
    args = parser.parse_args()

    print(f"=== Model comparison experiment: {args.model} (horizon=5, standard exit rule, matching production) ===")

    download_data_from_s3()
    symbols = list(set(fetch_sp500_symbols()) | set(ETF_SYMBOLS))

    clf_df    = load_all_symbols(symbols, target_type="classification", horizon=5)
    reg_df    = load_all_symbols(symbols, target_type="regression",     horizon=5)
    actual_df = load_all_symbols(symbols, target_type="regression",     horizon=1)
    clf_df    = clf_df[clf_df["date"].dt.year.isin(ALL_YEARS)].reset_index(drop=True)
    reg_df    = reg_df[reg_df["date"].dt.year.isin(ALL_YEARS)].reset_index(drop=True)
    actual_df = actual_df[actual_df["date"].dt.year.isin(ALL_YEARS)].reset_index(drop=True)

    predictions, metrics = eval_tabular_model(clf_df, reg_df, actual_df, args.model)

    turnover = compute_turnover(predictions)
    print(f"Turnover (top-30 day-over-day overlap): {turnover['mean_overlap_fraction']:.3f}  "
          f"over {turnover['n_day_pairs']} day-pairs")
    print(f"Sharpe={metrics['sharpe']:.4f}  Sortino={metrics['sortino']:.4f}  "
          f"CumReturn={metrics['cumulative_return']:.2%}  WinRate={metrics['win_rate']:.2%}")

    today_str = datetime.now().strftime("%Y-%m-%d")
    tag = f"experiment-{args.model}-{today_str}"
    try:
        import boto3, json
        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
        metrics["results_df"].to_csv(f"/tmp/{tag}-results.csv", index=False)
        predictions.to_parquet(f"/tmp/{tag}-predictions.parquet", index=False)
        s3.upload_file(f"/tmp/{tag}-results.csv", BUCKET, f"retrain/eval/{tag}-results.csv")
        s3.upload_file(f"/tmp/{tag}-predictions.parquet", BUCKET, f"retrain/eval/{tag}-predictions.parquet")
        summary = {
            "model": args.model,
            "sharpe": metrics["sharpe"], "sortino": metrics["sortino"],
            "cumulative_return": metrics["cumulative_return"], "win_rate": metrics["win_rate"],
            "turnover_mean_overlap_fraction": turnover["mean_overlap_fraction"],
        }
        s3.put_object(Bucket=BUCKET, Key=f"retrain/eval/{tag}-summary.json", Body=json.dumps(summary, indent=2))
        print(f"Uploaded to s3://{BUCKET}/retrain/eval/{tag}-*")
    except Exception as e:
        print(f"Upload failed (results still printed above): {e}")
