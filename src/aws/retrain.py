"""
Weekly retraining pipeline.

Runs on EC2 every Sunday night. Downloads the latest data from S3, evaluates
the LSTM on the last EVAL_MONTHS of walk-forward predictions, trains a final
model on all available data, saves artifacts locally, and uploads everything to S3.

Usage:
    python src/aws/retrain.py              # full run  (500 symbols, 2006-present)
    python src/aws/retrain.py --test       # smoke test (4 symbols, 2022-2025, ~15-30 min on GPU)

Schedule: see src/aws/retrain.timer (Saturday 12:00 UTC)
"""

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

HERE     = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.dirname(HERE)
ROOT_DIR = os.path.dirname(SRC_DIR)
for d in (SRC_DIR, os.path.join(SRC_DIR, "core"), os.path.join(SRC_DIR, "pipeline"), os.path.join(SRC_DIR, "models")):
    if d not in sys.path:
        sys.path.insert(0, d)

load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

from data_pipeline import load_all_symbols, fetch_sp500_symbols, ETF_SYMBOLS, INDICATORS_DIR
from sentiment import download_from_s3 as download_sentiment, update_all as update_sentiment
from backtest import run_backtest
from regime_pipeline import fetch_spy_regimes, FEATURES
from model_store import (
    save_artifacts, save_metadata,
    upload_artifacts_to_s3, upload_metadata_to_s3,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ============================================================
# Retrain run log  (saved to S3 after every run)
# ============================================================

class RetrainLog:
    """Structured log written to s3://<bucket>/retrain/last_run.json after every run."""

    def __init__(self, mode="full"):
        self._t       = None
        self._current = None
        self.data = {
            "mode":        mode,
            "started_at":  datetime.utcnow().isoformat() + "Z",
            "finished_at": None,
            "status":      "running",
            "steps":       [],
            "error":       None,
        }

    def begin_step(self, name):
        self._t       = time.time()
        self._current = name

    def end_step(self, **metrics):
        self.data["steps"].append({
            "name":       self._current,
            "status":     "ok",
            "duration_s": round(time.time() - self._t, 1),
            **{k: v for k, v in metrics.items() if v is not None},
        })
        self._current = None

    def skip_step(self, reason=""):
        self.data["steps"].append({
            "name":   self._current,
            "status": "skipped",
            "reason": reason,
        })
        self._current = None

    def fail(self, exc):
        self.data["steps"].append({
            "name":       self._current or "unknown",
            "status":     "failed",
            "duration_s": round(time.time() - self._t, 1) if self._t else None,
            "error":      str(exc),
        })
        self.data["status"]      = "failed"
        self.data["finished_at"] = datetime.utcnow().isoformat() + "Z"
        self.data["error"] = {
            "step":      self._current or "unknown",
            "type":      type(exc).__name__,
            "message":   str(exc),
            "traceback": traceback.format_exc(),
        }

    def finish(self):
        self.data["status"]      = "success"
        self.data["finished_at"] = datetime.utcnow().isoformat() + "Z"

    def save(self, bucket):
        if self.data["finished_at"] is None:
            self.data["finished_at"] = datetime.utcnow().isoformat() + "Z"
        try:
            import boto3
            boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1")).put_object(
                Bucket=bucket,
                Key="retrain/last_run.json",
                Body=json.dumps(self.data, indent=2, default=str).encode(),
                ContentType="application/json",
            )
            log.info(f"Retrain log saved → s3://{bucket}/retrain/last_run.json")
        except Exception as e:
            log.warning(f"Could not save retrain log to S3: {e}")


# --- Config ---
ALL_YEARS        = list(range(2006, pd.Timestamp.today().year + 1))
EVAL_MONTHS      = 12      # months of walk-forward used for model comparison
MIN_TRAIN_MONTHS = 6
SEQ_LEN               = 20
HIDDEN_SIZE           = 32
NUM_LAYERS            = 2
DROPOUT               = 0.0
LR                    = 1e-3
BATCH_SIZE            = 512
NUM_WORKERS           = 4
PATIENCE              = 10
MAX_EPOCHS            = 200
LSTM_TRAIN_WINDOW     = 36   # months; caps LSTM training data to avoid OOM on 500 symbols
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# S3 data download
# ============================================================

def download_data_from_s3():
    import boto3
    bucket = os.getenv("AWS_S3_BUCKET", "webull-trade-ai")
    s3     = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
    os.makedirs(INDICATORS_DIR, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    keys = [
        obj["Key"]
        for page in paginator.paginate(Bucket=bucket, Prefix="data/indicators/")
        for obj in page.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    ]
    log.info(f"Downloading {len(keys)} parquet files...")
    for i, key in enumerate(keys, 1):
        local = os.path.join(INDICATORS_DIR, os.path.basename(key))
        if not os.path.exists(local):
            s3.download_file(bucket, key, local)
        if i % 100 == 0 or i == len(keys):
            log.info(f"  {i}/{len(keys)} ready")


# ============================================================
# Live performance stats
# ============================================================

def compute_live_stats():
    """
    Compute live trading performance metrics from balance_history.csv.
    Returns a stats dict, or None if there is insufficient history (< 2 days).
    """
    from dashboard_logger import BALANCE_HISTORY_FILE

    if not os.path.exists(BALANCE_HISTORY_FILE):
        log.warning("balance_history.csv not found — skipping live stats.")
        return None

    bal = pd.read_csv(BALANCE_HISTORY_FILE, parse_dates=["date"])
    if len(bal) < 2:
        log.info("Not enough history for live stats (need at least 2 days).")
        return None

    bal = bal.sort_values("date").reset_index(drop=True)
    bal["daily_return"] = bal["total_balance"].pct_change()
    daily_returns = bal["daily_return"].dropna()

    initial = float(bal["total_balance"].iloc[0])
    current = float(bal["total_balance"].iloc[-1])
    cumulative_return = (current - initial) / initial

    daily_ret = float(daily_returns.iloc[-1])

    # Monthly and annual return (or annualized from inception)
    cutoff_monthly = bal["date"].iloc[-1] - pd.DateOffset(months=1)
    bal_mo = bal[bal["date"] >= cutoff_monthly]
    monthly_return = (
        (float(bal_mo["total_balance"].iloc[-1]) - float(bal_mo["total_balance"].iloc[0]))
        / float(bal_mo["total_balance"].iloc[0])
        if len(bal_mo) >= 2 else cumulative_return
    )

    cutoff_annual = bal["date"].iloc[-1] - pd.DateOffset(years=1)
    bal_yr = bal[bal["date"] >= cutoff_annual]
    if len(bal_yr) >= 2:
        annual_return = (
            (float(bal_yr["total_balance"].iloc[-1]) - float(bal_yr["total_balance"].iloc[0]))
            / float(bal_yr["total_balance"].iloc[0])
        )
    else:
        n_days = max((bal["date"].iloc[-1] - bal["date"].iloc[0]).days, 1)
        annual_return = (1.0 + cumulative_return) ** (365.0 / n_days) - 1.0

    # Sharpe (annualized from daily returns)
    sharpe = (
        float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
        if len(daily_returns) >= 2 and daily_returns.std() > 1e-9 else None
    )

    # Win rate: % of trading days with a positive return
    win_rate = float((daily_returns > 0).mean()) if len(daily_returns) > 0 else None

    # Max drawdown
    rolling_max  = bal["total_balance"].cummax()
    max_drawdown = float(((rolling_max - bal["total_balance"]) / rolling_max).max())

    # SPY comparison over the same period
    spy_return = None
    try:
        import yfinance as yf
        spy_start = bal["date"].iloc[0].strftime("%Y-%m-%d")
        spy_end   = (bal["date"].iloc[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        spy_hist  = yf.Ticker("SPY").history(start=spy_start, end=spy_end)
        if not spy_hist.empty:
            spy_return = float(
                (spy_hist["Close"].iloc[-1] - spy_hist["Close"].iloc[0])
                / spy_hist["Close"].iloc[0]
            )
    except Exception:
        pass

    stats = {
        "as_of":                   bal["date"].iloc[-1].strftime("%Y-%m-%d"),
        "days_live":               len(bal),
        "initial_capital":         round(initial, 2),
        "current_value":           round(current, 2),
        "daily_return":            round(daily_ret, 6),
        "monthly_return":          round(monthly_return, 6),
        "annual_return":           round(annual_return, 6),
        "cumulative_return":       round(cumulative_return, 6),
        "sharpe_ratio":            round(sharpe, 4) if sharpe is not None else None,
        "win_rate":                round(win_rate, 4) if win_rate is not None else None,
        "max_drawdown":            round(max_drawdown, 6),
        "spy_return_same_period":  round(spy_return, 6) if spy_return is not None else None,
        "excess_return":           round(cumulative_return - spy_return, 6) if spy_return is not None else None,
    }

    log.info(
        f"Live stats ({stats['days_live']} days): "
        f"cumulative={cumulative_return:+.2%}  "
        f"Sharpe={stats['sharpe_ratio']}  "
        f"vs SPY={f'{spy_return:+.2%}' if spy_return is not None else 'N/A'}"
    )
    return stats


# ============================================================
# LSTM helpers  (GPU-enabled)
# ============================================================

class LSTMModel(nn.Module):
    def __init__(self, input_dim, task="clf"):
        super().__init__()
        self.task = task
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.drop = nn.Dropout(DROPOUT)
        self.fc   = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        logit  = self.fc(self.drop(out[:, -1, :])).squeeze(1)
        return torch.sigmoid(logit) if self.task == "clf" else logit


def _make_sequences(df):
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    X_list, y_list, meta = [], [], []
    for sym, grp in df.groupby("symbol", sort=False):
        grp = grp.reset_index(drop=True)
        vals    = grp[FEATURES].values.astype(np.float32)
        targets = grp["target"].values.astype(np.float32)
        for i in range(SEQ_LEN, len(grp)):
            X_list.append(vals[i - SEQ_LEN: i])
            y_list.append(targets[i])
            meta.append({"date": grp["date"].iloc[i], "symbol": sym,
                         "year_month": grp["year_month"].iloc[i]})
    return (
        np.stack(X_list).astype(np.float32),
        np.array(y_list, dtype=np.float32),
        pd.DataFrame(meta),
    )


def _train_lstm(X_train, y_train, df_train_flat, task="clf"):
    scaler = StandardScaler()
    scaler.fit(df_train_flat[FEATURES].values)

    def scale(X):
        n, s, f = X.shape
        return scaler.transform(X.reshape(-1, f)).reshape(n, s, f).astype(np.float32)

    split = int(len(X_train) * 0.8)
    X_tr_s, X_val_s = scale(X_train[:split]), scale(X_train[split:])
    y_tr,   y_val   = y_train[:split], y_train[split:]

    def loader(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.float32))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))

    model     = LSTMModel(len(FEATURES), task=task).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.BCELoss() if task == "clf" else nn.MSELoss()
    best_loss, best_w, patience_left = float("inf"), None, PATIENCE

    for _ in range(MAX_EPOCHS):
        model.train()
        for X_b, y_b in loader(X_tr_s, y_tr, shuffle=True):
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss_fn(model(X_b), y_b).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in loader(X_val_s, y_val, shuffle=False):
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                val_loss += loss_fn(model(X_b), y_b).item() * len(X_b)
        val_loss /= len(y_val)
        if val_loss < best_loss:
            best_loss = val_loss
            best_w    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(best_w)
    return model, scaler


def _predict_lstm(model, scaler, X_seq, task="clf"):
    model.eval()
    n, s, f  = X_seq.shape
    X_scaled = scaler.transform(X_seq.reshape(-1, f)).reshape(n, s, f).astype(np.float32)
    with torch.no_grad():
        return model(torch.tensor(X_scaled).to(DEVICE)).cpu().numpy()


# ============================================================
# Walk-forward evaluation  (last EVAL_MONTHS months only)
# ============================================================

def _eval_months(clf_df):
    """Return the last EVAL_MONTHS months as the evaluation set."""
    all_months = sorted(clf_df["year_month"].unique())
    return set(all_months[-EVAL_MONTHS:])


def eval_lstm(clf_df, reg_df):
    """Walk-forward LSTM eval on last EVAL_MONTHS. Returns (predictions_df, last_models_dict)."""
    clf_df["year_month"] = clf_df["date"].dt.to_period("M")
    reg_df["year_month"] = reg_df["date"].dt.to_period("M")
    eval_set = _eval_months(clf_df)
    months   = sorted(clf_df["year_month"].unique())
    all_preds, last_models = [], {}

    for i in range(MIN_TRAIN_MONTHS, len(months) - 1):
        test_month = months[i + 1]
        if test_month not in eval_set:
            continue

        train_months  = months[:i + 1]
        lstm_months   = train_months[-LSTM_TRAIN_WINDOW:]   # cap to avoid OOM
        clf_tr_flat  = clf_df[clf_df["year_month"].isin(lstm_months)]
        clf_te_flat  = clf_df[clf_df["year_month"] == test_month]
        reg_tr_flat  = reg_df[reg_df["year_month"].isin(lstm_months)]
        reg_te_flat  = reg_df[reg_df["year_month"] == test_month]

        if len(clf_te_flat) == 0:
            continue

        # Classifier sequences
        clf_comb = pd.concat([clf_tr_flat, clf_te_flat]).sort_values(["symbol", "date"]).reset_index(drop=True)
        clf_comb["year_month"] = clf_comb["date"].dt.to_period("M")
        X_clf, y_clf, meta_clf = _make_sequences(clf_comb)
        tr_m = meta_clf["year_month"].isin(train_months).values
        te_m = (meta_clf["year_month"] == test_month).values
        if tr_m.sum() == 0 or te_m.sum() == 0:
            continue

        clf_model, clf_scaler = _train_lstm(X_clf[tr_m], y_clf[tr_m], clf_tr_flat, "clf")
        clf_prob = _predict_lstm(clf_model, clf_scaler, X_clf[te_m], "clf")

        # Regressor sequences
        reg_comb = pd.concat([reg_tr_flat, reg_te_flat]).sort_values(["symbol", "date"]).reset_index(drop=True)
        reg_comb["year_month"] = reg_comb["date"].dt.to_period("M")
        X_reg, y_reg, meta_reg = _make_sequences(reg_comb)
        tr_m_r = meta_reg["year_month"].isin(train_months).values
        te_m_r = (meta_reg["year_month"] == test_month).values

        reg_model, reg_scaler = _train_lstm(X_reg[tr_m_r], y_reg[tr_m_r], reg_tr_flat, "reg")
        reg_pred = _predict_lstm(reg_model, reg_scaler, X_reg[te_m_r], "reg")

        last_models = {"clf_model": clf_model, "clf_scaler": clf_scaler,
                       "reg_model": reg_model, "reg_scaler": reg_scaler}

        preds_clf = meta_clf[te_m][["date", "symbol"]].copy().reset_index(drop=True)
        preds_clf["clf_prob"] = clf_prob
        preds_reg = meta_reg[te_m_r][["date", "symbol"]].copy().reset_index(drop=True)
        preds_reg["reg_pred"] = reg_pred
        preds = preds_clf.merge(preds_reg, on=["date", "symbol"], how="inner")
        actual = (
            reg_df[reg_df["year_month"] == test_month][["date", "symbol", "target"]]
            .rename(columns={"target": "actual_return"})
        )
        preds = preds.merge(actual, on=["date", "symbol"], how="inner")
        if len(preds) == 0:
            continue

        all_preds.append(preds)
        log.info(f"  LSTM {test_month}  train_seq={tr_m.sum():,}  test_seq={te_m.sum():,}")

    predictions = pd.concat(all_preds, ignore_index=True)
    metrics     = run_backtest(predictions, plot=False)
    return predictions, metrics, last_models


# ============================================================
# Final model training  (all data, for deployment)
# ============================================================

def train_final_lstm(clf_df, reg_df):
    """Train LSTM clf + reg on last LSTM_TRAIN_WINDOW months. Returns artifacts dict."""
    clf_df["year_month"] = clf_df["date"].dt.to_period("M")
    reg_df["year_month"] = reg_df["date"].dt.to_period("M")
    window = set(sorted(clf_df["year_month"].unique())[-LSTM_TRAIN_WINDOW:])
    clf_df = clf_df[clf_df["year_month"].isin(window)]
    reg_df = reg_df[reg_df["year_month"].isin(window)]

    X_clf, y_clf, _ = _make_sequences(clf_df)
    X_reg, y_reg, _ = _make_sequences(reg_df)

    clf_model, clf_scaler = _train_lstm(X_clf, y_clf, clf_df, "clf")
    reg_model, reg_scaler = _train_lstm(X_reg, y_reg, reg_df, "reg")

    cfg = {"input_dim": len(FEATURES), "hidden_size": HIDDEN_SIZE,
           "num_layers": NUM_LAYERS, "dropout": DROPOUT}

    return {
        "clf_state_dict": clf_model.state_dict(),
        "clf_scaler":     clf_scaler,
        "reg_state_dict": reg_model.state_dict(),
        "reg_scaler":     reg_scaler,
        "model_config":   cfg,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test", action="store_true",
        help="Smoke-test mode: 4 symbols, 2022-2025. Verifies the full pipeline "
             "(train → save → upload) without running the full 500-symbol job."
    )
    args = parser.parse_args()

    TEST_SYMBOLS = ["AAPL", "XOM", "NVDA", "NFLX"]
    TEST_YEARS   = [2022, 2023, 2024, 2025]

    mode   = "test" if args.test else "full"
    bucket = os.getenv("AWS_S3_BUCKET", "webull-trade-ai")
    rlog   = RetrainLog(mode=mode)

    if args.test:
        log.info(f"=== Retrain SMOKE TEST  {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
        log.info(f"Symbols: {TEST_SYMBOLS}  |  Years: {TEST_YEARS}")
    else:
        log.info(f"=== Weekly Retrain  {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
    log.info(f"Device: {DEVICE}")

    try:
        # 1. Download latest data + build symbol list
        rlog.begin_step("download_data")
        log.info("\n--- Downloading data from S3 ---")
        download_data_from_s3()
        base_symbols = TEST_SYMBOLS if args.test else fetch_sp500_symbols()
        symbols      = list(set(base_symbols) | set(ETF_SYMBOLS))
        ALL_YEARS    = TEST_YEARS if args.test else ALL_YEARS
        log.info(f"Symbols: {len(symbols)} ({len(ETF_SYMBOLS)} ETFs included)  |  Years: {ALL_YEARS[0]}–{ALL_YEARS[-1]}")
        rlog.end_step(symbols=len(symbols))

        # 2. Refresh sentiment
        rlog.begin_step("sentiment_refresh")
        log.info("\n--- Refreshing sentiment data ---")
        download_sentiment()
        update_sentiment(symbols)
        rlog.end_step()

        # 3. Load datasets
        rlog.begin_step("load_data")
        clf_df = load_all_symbols(symbols, target_type="classification", horizon=1)
        reg_df = load_all_symbols(symbols, target_type="regression",     horizon=1)
        clf_df = clf_df[clf_df["date"].dt.year.isin(ALL_YEARS)].reset_index(drop=True)
        reg_df = reg_df[reg_df["date"].dt.year.isin(ALL_YEARS)].reset_index(drop=True)
        clf_df["year_month"] = clf_df["date"].dt.to_period("M")
        reg_df["year_month"] = reg_df["date"].dt.to_period("M")
        log.info(f"Rows — clf: {len(clf_df):,}  reg: {len(reg_df):,}")
        start = clf_df["date"].min().strftime("%Y-%m-%d")
        end   = clf_df["date"].max().strftime("%Y-%m-%d")
        rlog.end_step(clf_rows=len(clf_df), reg_rows=len(reg_df),
                      date_range=f"{start} → {end}")

        # 4. SPY regimes
        rlog.begin_step("spy_regimes")
        log.info(f"Fetching SPY regimes ({start} → {end})...")
        spy_regimes = fetch_spy_regimes(start, end)
        rlog.end_step()

        # 5. Evaluate LSTM
        rlog.begin_step("evaluate_lstm")
        log.info(f"\n--- Evaluating LSTM on last {EVAL_MONTHS} months ---")
        lstm_preds, lstm_metrics, _ = eval_lstm(clf_df.copy(), reg_df.copy())
        sharpes = {"lstm": round(lstm_metrics.get("sharpe", 0.0), 4)}
        winner  = "lstm"
        log.info(f"LSTM Sharpe: {sharpes['lstm']:.4f}")
        rlog.end_step(winner=winner, sharpe=sharpes["lstm"])

        # 6. Train final LSTM
        rlog.begin_step("train_final_lstm")
        log.info("\n--- Training final LSTM model on all data ---")
        artifacts = train_final_lstm(clf_df.copy(), reg_df.copy())
        rlog.end_step()

        # 7. Train portfolio allocator
        rlog.begin_step("train_allocator")
        log.info("\n--- Training portfolio allocator ---")
        try:
            from allocator import fetch_bucket_returns, build_allocator_df, walk_forward_allocate
            venture_results = run_backtest(lstm_preds, plot=False)
            if venture_results:
                bucket_returns = fetch_bucket_returns(start, end)
                alloc_df = build_allocator_df(
                    venture_results["results_df"], bucket_returns, spy_regimes
                )
                _, final_params = walk_forward_allocate(alloc_df, min_train_months=6)
                artifacts["allocator_params"] = final_params
                log.info("Allocator trained — params added to artifacts.")
                rlog.end_step(status_note="ok")
            else:
                log.warning("No venture returns — skipping allocator training.")
                rlog.skip_step("no venture returns from backtest")
        except Exception as e:
            log.warning(f"Allocator training failed: {e} — proceeding without allocator params.")
            rlog.skip_step(f"error: {e}")

        # 8. Save + upload artifacts
        rlog.begin_step("upload_artifacts")
        log.info("\n--- Saving artifacts ---")
        save_artifacts(winner, artifacts)
        save_metadata(
            winner            = winner,
            sharpe_scores     = sharpes,
            trained_up_to     = clf_df["date"].max().date(),
            evaluation_months = EVAL_MONTHS,
        )
        log.info("\n--- Uploading to S3 ---")
        upload_artifacts_to_s3(winner)
        upload_metadata_to_s3()
        rlog.end_step(winner=winner)

        # 9. Upload indicator parquets
        rlog.begin_step("upload_parquets")
        try:
            import boto3
            _s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
            parquet_files = [f for f in os.listdir(INDICATORS_DIR) if f.endswith(".parquet")]
            log.info(f"Uploading {len(parquet_files)} indicator parquets to S3...")
            for i, fname in enumerate(parquet_files, 1):
                _s3.upload_file(
                    os.path.join(INDICATORS_DIR, fname),
                    bucket,
                    f"data/indicators/{fname}",
                )
                if i % 100 == 0 or i == len(parquet_files):
                    log.info(f"  {i}/{len(parquet_files)} uploaded")
            rlog.end_step(parquets_uploaded=len(parquet_files))
        except Exception as e:
            log.warning(f"Indicator parquet upload failed: {e}")
            rlog.skip_step(f"error: {e}")

        # 10. Live performance stats
        rlog.begin_step("live_stats")
        log.info("\n--- Computing live performance stats ---")
        try:
            live_stats = compute_live_stats()
            if live_stats:
                import boto3
                s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
                s3_client.put_object(
                    Bucket=bucket,
                    Key="state/performance_stats.json",
                    Body=json.dumps(live_stats, indent=2).encode(),
                    ContentType="application/json",
                    CacheControl="no-cache",
                )
                log.info(f"Performance stats uploaded to s3://{bucket}/state/performance_stats.json")
                rlog.end_step(days_live=live_stats.get("days_live"),
                              cumulative_return=live_stats.get("cumulative_return"))
            else:
                rlog.skip_step("insufficient balance history")
        except Exception as e:
            log.warning(f"Live stats upload failed: {e}")
            rlog.skip_step(f"error: {e}")

        rlog.finish()
        log.info("\n=== Retrain complete ===")

    except Exception as e:
        rlog.fail(e)
        log.error(f"Retrain failed at step '{rlog.data['error']['step']}': {e}")
        raise

    finally:
        rlog.save(bucket)
