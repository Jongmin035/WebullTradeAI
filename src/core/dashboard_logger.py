"""
Dashboard logger — records daily trading stats and uploads to S3.

Writes/updates src/dashboard/stats.json after every rebalance.
If AWS_S3_BUCKET is set in .env, also uploads to S3 for the live dashboard.

Usage:
    from dashboard_logger import log_rebalance, log_error, clear_error, load_stats

Setup (one-time):
    Add to .env:
        AWS_S3_BUCKET=your-bucket-name
        AWS_REGION=us-east-1        (optional, defaults to us-east-1)
    Install: pip install boto3

S3 bucket setup:
    1. Create an S3 bucket
    2. Enable "Static website hosting" (index document: index.html)
    3. Set bucket policy to allow public read on stats.json and index.html
    4. Give your EC2 instance an IAM role with s3:PutObject permission on the bucket
"""

import csv
import json
import os
import sys
import logging
from datetime import date, datetime

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _d in (_src, os.path.join(_src, "core"), os.path.join(_src, "pipeline")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

log = logging.getLogger(__name__)

_STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "state")

STATS_FILE           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dashboard", "stats.json")
BALANCE_HISTORY_FILE = os.path.join(_STATE_DIR, "balance_history.csv")
TRADE_LOG_FILE       = os.path.join(_STATE_DIR, "trade_log.csv")

_BALANCE_HEADERS = ["date", "cash_balance", "market_value", "total_balance"]
_TRADE_HEADERS   = ["timestamp", "date", "symbol", "side", "quantity", "price", "value"]


# --- Load / save ---

def load_stats():
    """Load the current stats.json. Returns a fresh structure if not found."""
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE) as f:
            return json.load(f)
    return {
        "status": "ok",
        "error_message": None,
        "initial_capital": None,
        "history": [],
    }


def _load_commands():
    """Read current commands.json so the dashboard can display control state."""
    commands_file = os.path.join(os.path.dirname(__file__), "..", "state", "commands.json")
    if os.path.exists(commands_file):
        with open(commands_file) as f:
            return json.load(f)
    return {"max_capital": None, "manual_symbols": [], "force_sell": []}


def _generate_history_html():
    """Regenerate history.html from the latest CSVs. Silently skips if history.py is unavailable."""
    try:
        import sys as _sys
        _dashboard_dir = os.path.dirname(STATS_FILE)
        if _dashboard_dir not in _sys.path:
            _sys.path.insert(0, _dashboard_dir)
        from history import generate_html
        generate_html()
    except Exception as e:
        log.debug(f"history.html generation skipped: {e}")


def _save_stats(stats):
    os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
    stats["controls"] = _load_commands()   # embed current control state for dashboard
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    _generate_history_html()
    _upload_to_s3()


def _upload_to_s3():
    """Upload dashboard files and all state files to S3 if AWS_S3_BUCKET is configured."""
    bucket = os.getenv("AWS_S3_BUCKET")
    if not bucket:
        return
    try:
        import boto3
        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))

        # Dashboard files (public-facing)
        dashboard_dir = os.path.dirname(STATS_FILE)
        for filename in ["stats.json", "index.html", "history.html"]:
            local_path = os.path.join(dashboard_dir, filename)
            if not os.path.exists(local_path):
                continue
            content_type = "application/json" if filename.endswith(".json") else "text/html"
            s3.upload_file(
                local_path, bucket, filename,
                ExtraArgs={"ContentType": content_type, "CacheControl": "no-cache"},
            )

        # State files (crash recovery backup)
        state_files = [
            (BALANCE_HISTORY_FILE, "state/balance_history.csv",    "text/csv"),
            (TRADE_LOG_FILE,       "state/trade_log.csv",           "text/csv"),
            (os.path.join(_STATE_DIR, "commands.json"),             "state/commands.json",             "application/json"),
            (os.path.join(_STATE_DIR, "peak_portfolio_value.json"), "state/peak_portfolio_value.json", "application/json"),
            (os.path.join(_STATE_DIR, "position_highs.json"),       "state/position_highs.json",       "application/json"),
        ]
        for local_path, s3_key, content_type in state_files:
            if not os.path.exists(local_path):
                continue
            s3.upload_file(
                local_path, bucket, s3_key,
                ExtraArgs={"ContentType": content_type, "CacheControl": "no-cache"},
            )

        log.info(f"Dashboard and state backed up to s3://{bucket}/")
    except Exception as e:
        log.warning(f"S3 upload failed: {e}")


def restore_state_from_s3():
    """
    Download missing state files from S3 on startup.

    Only restores files that do not already exist locally — never overwrites
    existing local state, which may be more recent than the last S3 backup.

    Call this once at bot startup before doing anything else.
    """
    bucket = os.getenv("AWS_S3_BUCKET")
    if not bucket:
        return
    try:
        import boto3
        from botocore.exceptions import ClientError
        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))

        restore_map = [
            ("state/balance_history.csv",    BALANCE_HISTORY_FILE),
            ("state/trade_log.csv",          TRADE_LOG_FILE),
            ("state/commands.json",          os.path.join(_STATE_DIR, "commands.json")),
            ("state/peak_portfolio_value.json", os.path.join(_STATE_DIR, "peak_portfolio_value.json")),
            ("state/position_highs.json",    os.path.join(_STATE_DIR, "position_highs.json")),
            ("stats.json",                   STATS_FILE),
        ]

        restored = []
        for s3_key, local_path in restore_map:
            if os.path.exists(local_path):
                continue  # local file present — do not overwrite
            try:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                s3.download_file(bucket, s3_key, local_path)
                restored.append(s3_key)
            except ClientError as e:
                if e.response["Error"]["Code"] != "404":
                    log.warning(f"Could not restore {s3_key} from S3: {e}")

        if restored:
            log.info(f"Restored from S3: {restored}")
        else:
            log.info("State restore: all local files present, nothing to restore.")
    except Exception as e:
        log.warning(f"State restore from S3 failed: {e}")


# --- Index comparison ---

def _fetch_index_history(start_date_str):
    """
    Fetch cumulative % return for SPY, DIA, QQQ from start_date_str to today.
    Returns dict {ticker: [{date, pct}, ...]} or {} on failure.
    Called once per daily rebalance — result stored in stats.json.
    """
    try:
        import pandas as pd
        import yfinance as yf
        result = {}
        for ticker in ["SPY", "DIA", "QQQ"]:
            hist = yf.Ticker(ticker).history(start=start_date_str, auto_adjust=True)
            if hist.empty or len(hist) < 2:
                continue
            closes = hist["Close"].copy()
            closes.index = pd.to_datetime(closes.index).tz_localize(None)
            base = float(closes.iloc[0])
            if base == 0:
                continue
            result[ticker] = [
                {"date": d.strftime("%Y-%m-%d"), "pct": round((float(c) - base) / base * 100, 4)}
                for d, c in closes.items()
            ]
        return result
    except Exception as e:
        log.warning(f"Could not fetch index history: {e}")
        return {}


# --- CSV helpers ---

def _save_balance_history(today_str, cash_balance, market_value):
    """
    Upsert one row per day into balance_history.csv.
    If today already has an entry (e.g. from log_snapshot earlier in the day),
    it is replaced so the file always holds the most recent values for each date.
    """
    os.makedirs(_STATE_DIR, exist_ok=True)
    total = round(cash_balance + market_value, 2)
    new_row = {
        "date":          today_str,
        "cash_balance":  round(cash_balance, 2),
        "market_value":  round(market_value, 2),
        "total_balance": total,
    }

    rows = []
    if os.path.exists(BALANCE_HISTORY_FILE):
        with open(BALANCE_HISTORY_FILE, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["date"] != today_str]

    rows.append(new_row)
    rows.sort(key=lambda r: r["date"])

    with open(BALANCE_HISTORY_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_BALANCE_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def _append_trades(trades, today_str, timestamp):
    """
    Append executed trades to trade_log.csv. Never overwrites existing rows.
    Each trade gets the rebalance timestamp so you can group by session.
    """
    if not trades:
        return
    os.makedirs(_STATE_DIR, exist_ok=True)
    write_header = not os.path.exists(TRADE_LOG_FILE)
    with open(TRADE_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TRADE_HEADERS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for t in trades:
            writer.writerow({
                "timestamp": timestamp,
                "date":      today_str,
                "symbol":    t.get("symbol", ""),
                "side":      t.get("side", ""),
                "quantity":  t.get("quantity", ""),
                "price":     t.get("price", ""),
                "value":     t.get("value", ""),
            })


# --- Public API ---

def log_rebalance(account, effective_pv, bot_positions, manual_positions, trades):
    """
    Record the results of a completed rebalance.

    Parameters
    ----------
    account : dict
        Full Webull account snapshot: net_account_value, cash_balance, market_value,
        day_pnl, open_pnl  (covers the entire account, not just the bot's portion).
    effective_pv : float
        Bot's effective portfolio value = cap minus manual positions' market value.
        This is what the bot actually controls and what performance is measured against.
    bot_positions : dict {symbol: {weight, market_value, price, open_pnl, day_pnl}}
        Positions actively managed by the bot.
    manual_positions : dict {symbol: {weight, market_value, price, open_pnl, day_pnl}}
        Positions the user has taken control of (excluded from bot's book).
    trades : list of dicts with keys: symbol, side, quantity, price, value
    """
    stats = load_stats()

    # Set initial capital on first run (based on effective bot capital)
    if stats["initial_capital"] is None:
        stats["initial_capital"] = effective_pv

    initial = stats["initial_capital"]
    cumulative_return_pct = round((effective_pv - initial) / initial * 100, 4) if initial else 0

    # Today's change vs previous session
    today_change_pct = None
    prev = next((h for h in reversed(stats["history"]) if not h.get("error")), None)
    if prev:
        prev_value = prev["effective_portfolio_value"]
        if prev_value:
            today_change_pct = round((effective_pv - prev_value) / prev_value * 100, 4)

    # Bot-specific computed stats
    bot_market_value = round(sum(d["market_value"] for d in bot_positions.values()), 2)
    bot_cash         = round(max(effective_pv - bot_market_value, 0), 2)

    # Derive P&L from portfolio value history — reliable regardless of API field names.
    # Open P&L = total gain/loss since bot started.
    # Day P&L  = change vs previous session.
    bot_open_pnl = round(effective_pv - initial, 2) if initial else 0.0
    bot_day_pnl  = round(effective_pv - prev["effective_portfolio_value"], 2) \
                   if prev and prev.get("effective_portfolio_value") else 0.0

    entry = {
        "date":      date.today().isoformat(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),

        # Bot's own book (what performance is tracked against)
        "effective_portfolio_value": round(effective_pv, 2),
        "bot_market_value":          bot_market_value,
        "bot_cash":                  bot_cash,
        "bot_open_pnl":              bot_open_pnl,
        "bot_day_pnl":               bot_day_pnl,
        "cumulative_return_pct":     cumulative_return_pct,
        "today_change_pct":          today_change_pct,

        # Full account from Webull (for reference)
        "net_account_value": round(account.get("net_account_value", effective_pv), 2),
        "cash_balance":      round(account.get("cash_balance", 0), 2),

        # Positions
        "positions":         bot_positions,
        "manual_positions":  manual_positions,
        "trades":            trades,
    }

    # Replace today's entry if already exists (idempotent re-runs)
    today = date.today().isoformat()
    stats["history"] = [h for h in stats["history"] if h["date"] != today]
    stats["history"].append(entry)
    stats["history"].sort(key=lambda h: h["date"])

    # Clear any prior error/warning on a successful run
    stats["status"] = "ok"
    stats["error_message"] = None
    stats["warning_message"] = None

    # Write CSVs first so history.html (generated inside _save_stats) has today's data.
    _save_balance_history(today, bot_cash, bot_market_value)
    _append_trades(trades, today, entry["timestamp"])

    # Fetch benchmark index returns from the bot's start date (stored for chart overlay).
    first_date = next((h["date"] for h in stats["history"] if not h.get("error")), today)
    stats["index_history"] = _fetch_index_history(first_date)

    _save_stats(stats)

    log.info(f"Dashboard updated — cumulative return: {cumulative_return_pct:+.2f}%")


def refresh_controls():
    """Re-save stats.json with the current commands.json content embedded.
    Call this after any control action (add-funds, take-control, etc.) so the
    dashboard immediately reflects the updated settings."""
    _save_stats(load_stats())


def log_warning(message):
    """
    Record a transient warning (e.g. API retry in progress).
    Does NOT block trading on subsequent runs — clears automatically on the
    next successful rebalance. Use log_error() only for permanent issues that
    require manual intervention.
    """
    stats = load_stats()
    stats["status"] = "warning"
    stats["warning_message"] = f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {message}"
    _save_stats(stats)
    log.warning(f"Warning logged to dashboard: {message}")


def log_error(message):
    """
    Record an error that halted trading. Dashboard will turn red.
    Trading will be blocked on subsequent runs until clear_error() is called.
    """
    stats = load_stats()
    stats["status"] = "error"
    stats["error_message"] = f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {message}"

    # Append an error marker to history so the chart shows a gap
    today = date.today().isoformat()
    stats["history"] = [h for h in stats["history"] if h["date"] != today]
    stats["history"].append({
        "date": today,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "error": True,
        "error_message": message,
    })

    _save_stats(stats)
    log.error(f"Error logged to dashboard: {message}")


def clear_error():
    """
    Manually clear an error state to re-enable trading.
    Run this after investigating and fixing the underlying issue:
        python -c "from dashboard_logger import clear_error; clear_error()"
    """
    stats = load_stats()
    stats["status"] = "ok"
    stats["error_message"] = None
    _save_stats(stats)
    log.info("Error cleared — trading re-enabled.")
