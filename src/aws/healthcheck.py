"""
Bot health check.

Reads stats.json and exits with:
  0 — healthy (last run successful and recent)
  1 — unhealthy (error state, or bot hasn't run in too long)

Usage:
    python healthcheck.py              # checks against default MAX_STALE_DAYS=2
    python healthcheck.py --days 3     # custom staleness threshold

On EC2, add to crontab to get notified if the bot goes silent:
    0 20 * * 1-5 python /home/ec2-user/WebullTradeAI/src/core/healthcheck.py >> /var/log/bot_health.log 2>&1
"""

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _src not in sys.path:
    sys.path.insert(0, _src)

STATS_FILE    = os.path.join(_src, "dashboard", "stats.json")
MAX_STALE_DAYS = 2   # alert if no successful run within this many calendar days


def _load_stats():
    if not os.path.exists(STATS_FILE):
        return None
    with open(STATS_FILE) as f:
        return json.load(f)


def check(max_stale_days=MAX_STALE_DAYS):
    """
    Returns (healthy: bool, message: str).
    """
    stats = _load_stats()

    if stats is None:
        return False, "stats.json not found — bot has never run or state is missing"

    if stats.get("status") == "error":
        msg = stats.get("error_message", "unknown error")
        return False, f"Bot is in ERROR state: {msg}"

    history = stats.get("history", [])
    successful = [h for h in history if not h.get("error")]

    if not successful:
        return False, "No successful runs recorded in stats.json"

    last = successful[-1]
    last_date = datetime.strptime(last["date"], "%Y-%m-%d").date()
    days_ago   = (date.today() - last_date).days

    if days_ago > max_stale_days:
        return False, (
            f"Last successful run was {days_ago} day(s) ago ({last_date}) — "
            f"expected within {max_stale_days} day(s)"
        )

    cum_return = last.get("cumulative_return_pct", "n/a")
    return True, (
        f"OK — last run: {last_date} ({days_ago}d ago)  "
        f"cumulative return: {cum_return:+.2f}%"
        if isinstance(cum_return, float) else
        f"OK — last run: {last_date} ({days_ago}d ago)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=MAX_STALE_DAYS,
                        help="Max calendar days since last successful run before alerting")
    args = parser.parse_args()

    healthy, message = check(max_stale_days=args.days)
    status = "HEALTHY" if healthy else "UNHEALTHY"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {status}: {message}")
    sys.exit(0 if healthy else 1)
