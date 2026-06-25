"""
Daily trading entry point.

Runs on EC2 every weekday at 9:35 AM ET (after market open).
Restores state from S3 if needed, generates predictions using the
current winning model, and executes rebalancing trades.

Usage:
    python src/aws/main.py            # live trading
    python src/aws/main.py --dry-run  # simulate without placing orders

Schedule: see src/aws/bot.timer
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

HERE     = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.dirname(HERE)
ROOT_DIR = os.path.dirname(SRC_DIR)
for d in (SRC_DIR, os.path.join(SRC_DIR, "core"), os.path.join(SRC_DIR, "pipeline"),
          os.path.join(SRC_DIR, "models")):
    if d not in sys.path:
        sys.path.insert(0, d)

load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

_ET = ZoneInfo("America/New_York")


def _write_diag(label):
    """Write a one-line marker to S3 diagnostics/ so we can pinpoint where crashes happen."""
    try:
        import boto3
        boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1")).put_object(
            Bucket=os.getenv("AWS_S3_BUCKET", "webull-trade-ai"),
            Key=f"diagnostics/{datetime.now(_ET).strftime('%Y-%m-%d')}-{label}.txt",
            Body=f"{label} at {datetime.now(_ET).isoformat()}",
            ContentType="text/plain",
        )
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

VERSION = os.environ.get("GIT_SHA", "dev")

MAX_RETRIES      = 2
RETRY_DELAY      = 5 * 60                    # seconds between retry attempts (5 minutes)
TRADING_DEADLINE = dtime(15, 30)             # exit if past 3:30 PM ET before rebalancing


def _is_market_open():
    """Return True if today is a US trading day per Webull's trade calendar."""
    from trader import Trader
    today = datetime.now(_ET).strftime("%Y-%m-%d")
    try:
        res = Trader(dry_run=True).trade.trade_calendar.get_trade_calendar("US", today, today)
        return len(res.json()) > 0
    except Exception as e:
        log.warning(f"Could not check trade calendar ({e}) — assuming market is open.")
        return True  # fail open: better to attempt than miss a trading day


def _save_predictions_to_s3(predictions):
    """Upload today's top predictions to S3 for debugging visibility."""
    try:
        import json
        import boto3
        bucket = os.getenv("AWS_S3_BUCKET")
        if not bucket:
            return
        top = (
            predictions
            .nlargest(30, "clf_prob")[["symbol", "clf_prob", "reg_pred"]]
            .round(4)
            .to_dict("records")
        )
        key = f"predictions/{datetime.now(_ET).strftime('%Y-%m-%d')}.json"
        boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1")).put_object(
            Bucket=bucket, Key=key,
            Body=json.dumps(top, indent=2),
            ContentType="application/json",
        )
        log.info(f"Saved top {len(top)} predictions to s3://{bucket}/{key}")
    except Exception as e:
        log.warning(f"Could not save predictions to S3: {e}")


def main(dry_run=False):
    _write_diag("main-called")

    # Restore state FIRST (so log_error can write stats.json on any later failure)
    log_error = None
    log_warning = None
    try:
        from dashboard_logger import restore_state_from_s3, log_warning, log_error
        log.info("Restoring state from S3 (if needed)...")
        restore_state_from_s3()
        _write_diag("restore-ok")
    except Exception as e:
        _write_diag(f"restore-failed-{type(e).__name__}")
        log.error(f"Failed to restore state: {e}", exc_info=True)
        return

    try:
        from data_pipeline import fetch_sp500_symbols, ETF_SYMBOLS
        from model_store import load_artifacts, load_metadata
        from predict import get_predictions_today, get_today_regime_vix, get_allocation_today, VERSION as PREDICT_VERSION
        from trader import Trader, VERSION as TRADER_VERSION

        log.info(f"bot v{VERSION}  trader v{TRADER_VERSION}  predict v{PREDICT_VERSION}")

        # 0. Exit immediately if today is a holiday or weekend
        if not _is_market_open():
            log.info("Market closed today — nothing to do.")
            return

        # 1. Load symbol list
        symbols = list(set(fetch_sp500_symbols()) | set(ETF_SYMBOLS))
        log.info(f"Universe: {len(symbols)} symbols ({len(ETF_SYMBOLS)} ETFs included)")

        # 2. Generate today's predictions using the current winning model
        log.info("Generating predictions...")
        _write_diag("predictions-start")
        predictions = get_predictions_today(symbols)
        log.info(f"Predictions ready: {len(predictions)} candidates")

        if predictions.empty:
            log.warning("No predictions generated — skipping rebalance.")
            return

        _save_predictions_to_s3(predictions)

        # 2.5. Compute today's bucket allocation
        allocation = None
        try:
            meta = load_metadata()
            if meta:
                artifacts = load_artifacts(meta["winner"])
                regime, vix = get_today_regime_vix()
                allocation  = get_allocation_today(artifacts, regime, vix)
                log.info(
                    f"Allocation — venture: {allocation['venture_pct']:.0%}  "
                    f"safety: {allocation['safety_pct']:.0%}  "
                    f"hedge: {allocation['hedge_pct']:.0%}  "
                    f"cash: {allocation['cash_pct']:.0%}"
                )
        except Exception as e:
            log.warning(f"Could not compute allocation ({e}) — defaulting to 100% venture.")

        # 3. Rebalance with retry (handles transient Webull API failures)
        now_et = datetime.now(_ET).time()
        if now_et > TRADING_DEADLINE:
            msg = f"Past trading deadline ({now_et.strftime('%H:%M')} ET > 15:30 ET) — skipping rebalance."
            log.warning(msg)
            log_warning(msg)
            return

        trader = Trader(dry_run=dry_run)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                trader.rebalance(predictions, allocation=allocation)
                return   # success
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                if attempt < MAX_RETRIES:
                    warning = (
                        f"Rebalance attempt {attempt}/{MAX_RETRIES} failed: {msg}. "
                        f"Retrying in 5 min..."
                    )
                    log.error(warning)
                    log_warning(warning)
                    time.sleep(RETRY_DELAY)
                else:
                    final = (
                        f"All {MAX_RETRIES} rebalance attempts failed — skipping today. "
                        f"Last error: {msg}"
                    )
                    log.error(final)
                    log_warning(final)

    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        log.error(f"Fatal error during trading run: {msg}", exc_info=True)
        _write_diag(f"fatal-{type(e).__name__}")
        try:
            log_error(msg)
        except Exception as e2:
            _write_diag(f"log-error-failed-{type(e2).__name__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute trades but do not submit orders")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
