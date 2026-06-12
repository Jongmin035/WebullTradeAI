"""
Live trading executor.

Takes today's model predictions, computes Kelly target weights, fetches
current Webull positions, and executes rebalancing orders — the live
equivalent of run_backtest().

Usage:
    from trader import Trader
    trader = Trader()
    trader.rebalance(predictions_today)   # predictions_today: DataFrame with
                                          # columns: symbol, clf_prob, reg_pred

Environment variables (set in .env):
    WEBULL_APP_KEY
    WEBULL_APP_SECRET
    WEBULL_REGION        (default: us)
    WEBULL_ACCOUNT_ID
    WEBULL_ENV           (uat | prd, default: uat)
"""

import os
import sys
import uuid
import logging
import pandas as pd
from dotenv import load_dotenv

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _d in (_src, os.path.join(_src, "core"), os.path.join(_src, "pipeline")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import webull
import webull.core
webull.__version__ = webull.core.__version__

from webull.core.client import ApiClient
from webull.trade.trade_client import TradeClient

from metrics import kelly_fraction
from safeguards import run_checks, check_stop_losses, update_position_highs
from dashboard_logger import load_stats, log_rebalance
from controls import load_commands, save_commands

load_dotenv()

# --- Configuration ---
REBALANCE_THRESHOLD    = 0.02  # minimum weight delta to trigger a trade
HALF_KELLY             = True
MIN_TRADE_VALUE        = 1.0   # ignore positions worth less than $1 (rounding)
MAX_CAPITAL            = None  # None = full account control
CLF_PROB_THRESHOLD     = 0.60  # minimum model confidence to include a position
MIN_POSITION_WEIGHT    = 0.03  # drop positions that would be < 3% of portfolio
MAX_VENTURE_POSITIONS  = 10    # keep only top-N venture stocks by Kelly score

SAFETY_ETFS = ["SPY", "XLP", "XLU"]
HEDGE_ETFS  = ["GLDM", "SH", "SQQQ"]

UAT_ENDPOINT = "us-openapi-alb.uat.webullbroker.com"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

VERSION = "__VERSION__"


def _etf_bucket_weights(etf_list, predictions_today, bucket_pct):
    """
    Distribute bucket_pct among ETFs that pass the model gate.

    Only ETFs with clf_prob >= CLF_PROB_THRESHOLD and reg_pred > 0 are
    included. Weights within the bucket are proportional to Kelly fraction.
    Falls back to equal-weight among all listed ETFs if none pass the gate,
    so the bucket allocation is never silently dropped to cash.
    """
    pool = predictions_today[predictions_today["symbol"].isin(etf_list)].copy()
    qualified = pool[
        (pool["clf_prob"] >= CLF_PROB_THRESHOLD) & (pool["reg_pred"] > 0)
    ].copy()

    if qualified.empty:
        # Fallback: equal-weight all ETFs in the list (preserves old behaviour)
        per_etf = bucket_pct / len(etf_list)
        return {sym: per_etf for sym in etf_list}

    qualified["kelly"] = qualified.apply(
        lambda r: kelly_fraction(r["clf_prob"], r["reg_pred"], half_kelly=HALF_KELLY),
        axis=1,
    )
    qualified = qualified[qualified["kelly"] > 0]

    if qualified.empty:
        per_etf = bucket_pct / len(etf_list)
        return {sym: per_etf for sym in etf_list}

    total_kelly = qualified["kelly"].sum()
    return {
        row["symbol"]: (row["kelly"] / total_kelly) * bucket_pct
        for _, row in qualified.iterrows()
    }


class Trader:
    """
    Connects to Webull and executes model-driven rebalancing trades.

    Parameters
    ----------
    dry_run : bool
        If True, compute and log all intended trades but do NOT submit orders.
        Always use dry_run=True until you are ready to trade real money.
    """

    def __init__(self, dry_run=True):
        self.dry_run    = dry_run
        self.account_id = os.getenv("WEBULL_ACCOUNT_ID")
        env             = os.getenv("WEBULL_ENV", "uat").lower()

        app_key    = os.getenv("WEBULL_APP_KEY")
        app_secret = os.getenv("WEBULL_APP_SECRET")
        region     = os.getenv("WEBULL_REGION", "us")

        if not all([app_key, app_secret, self.account_id]):
            raise EnvironmentError(
                "Missing Webull credentials. Set WEBULL_APP_KEY, WEBULL_APP_SECRET, "
                "and WEBULL_ACCOUNT_ID in your .env file."
            )

        self.client = ApiClient(app_key, app_secret, region)
        if env == "uat":
            self.client.add_endpoint(region, UAT_ENDPOINT)
            log.info("Connected to Webull UAT (test environment)")
        else:
            log.info("Connected to Webull PRODUCTION")

        self.trade = TradeClient(self.client)

        if dry_run:
            log.info("DRY RUN mode — no orders will be submitted")

    # ------------------------------------------------------------------
    # Account helpers
    # ------------------------------------------------------------------

    def get_account_snapshot(self):
        """
        Return a full account-level snapshot from Webull.

        Returns dict with keys:
            net_account_value, cash_balance, market_value, day_pnl, open_pnl
        """
        res = self.trade.account_v2.get_account_balance(self.account_id)
        if res.status_code != 200:
            raise RuntimeError(f"Failed to fetch account balance: {res.text}")
        data = res.json()
        return {
            "net_account_value": float(data.get("total_net_liquidation_value", 0)),
            "cash_balance":      float(data.get("cash_balance", data.get("cash", 0))),
            "market_value":      float(data.get("stock_value",  data.get("market_value", 0))),
            "day_pnl":           float(data.get("day_pnl",      data.get("unrealized_pnl_today", 0))),
            "open_pnl":          float(data.get("open_pnl",     data.get("unrealized_pnl", 0))),
        }

    def get_position_details(self, portfolio_value):
        """
        Return current positions with full per-position details.

        Returns dict {symbol: {weight, market_value, price, open_pnl, day_pnl}}
        Also returns plain weights dict for backward-compatible use.
        """
        res = self.trade.account_v2.get_account_position(self.account_id)
        if res.status_code != 200:
            raise RuntimeError(f"Failed to fetch positions: {res.text}")

        raw = res.json()
        positions_list = raw if isinstance(raw, list) else raw.get("positions", [])

        details = {}
        for pos in positions_list:
            symbol       = pos.get("symbol")
            market_value = float(pos.get("market_value", 0))
            if not symbol or market_value <= MIN_TRADE_VALUE:
                continue
            details[symbol] = {
                "weight":       round(market_value / portfolio_value, 4),
                "market_value": round(market_value, 2),
                "price":        float(pos.get("last_price", pos.get("price", 0))),
                "open_pnl":     round(float(pos.get("unrealized_profit_loss", 0)), 2),
                "day_pnl":      round(float(pos.get("day_profit_loss", 0)), 2),
            }
        return details

    def get_current_price(self, symbol):
        """
        Fetch the latest price for a symbol via yfinance.
        Used to convert a dollar trade amount into a share quantity.
        """
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        price  = ticker.fast_info.get("lastPrice") or ticker.fast_info.get("previousClose")
        if not price:
            raise RuntimeError(f"Could not fetch price for {symbol} via yfinance")
        return float(price)

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def _place_order(self, symbol, side, dollar_amount, current_price):
        """
        Place a market order for a given dollar amount.
        Converts dollar amount to share quantity (whole shares only).
        """
        quantity = int(dollar_amount / current_price)
        if quantity < 1:
            log.info(f"  SKIP {symbol}: quantity rounds to 0 (${dollar_amount:.2f} / ${current_price:.2f})")
            return False

        order = [{
            "client_order_id":         uuid.uuid4().hex,
            "combo_type":              "NORMAL",
            "symbol":                  symbol,
            "instrument_type":         "EQUITY",
            "market":                  "US",
            "order_type":              "MARKET",
            "quantity":                str(quantity),
            "support_trading_session": "N",
            "side":                    side,
            "time_in_force":           "DAY",
            "entrust_type":            "QTY",
        }]

        trade_info = {"symbol": symbol, "side": side, "quantity": quantity,
                      "price": round(current_price, 4), "value": round(quantity * current_price, 2)}

        if self.dry_run:
            log.info(f"  DRY RUN  {side:4s} {quantity:6d} shares of {symbol} (~${dollar_amount:.2f})")
            return trade_info

        res = self.trade.order_v2.place_order(account_id=self.account_id, new_orders=order)
        if res.status_code == 200:
            log.info(f"  ORDER    {side:4s} {quantity:6d} shares of {symbol} (~${dollar_amount:.2f})")
            return trade_info
        else:
            log.error(f"  FAILED   {side:4s} {symbol}: {res.text}")
            return None

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def _load_book_symbols(self):
        """
        Return the set of symbols the bot had in its book at the last
        successful rebalance (from stats.json).  Empty set on first run.
        """
        stats = load_stats()
        last = next(
            (h for h in reversed(stats.get("history", [])) if not h.get("error")),
            None,
        )
        return set(last["positions"].keys()) if last else set()

    def _reconcile_pre_trade(self, actual_bot_symbols, target_weights):
        """
        Compare last-logged bot positions (book) with what's actually in the
        account right now.  Any symbol that was in the book but has since
        disappeared was manually removed between rebalances.

        Conservative action: drop it from target_weights so the bot does NOT
        re-buy it this rebalance.  No permanent change to manual_symbols —
        if the model still wants it next rebalance the user can decide via
        the take-control mechanism.

        Returns the (possibly modified) target_weights dict.
        """
        book_symbols = self._load_book_symbols()
        if not book_symbols:
            log.info("Pre-trade reconciliation: no prior book — skipping (first run).")
            return target_weights

        manually_removed = book_symbols - actual_bot_symbols
        if not manually_removed:
            log.info("Pre-trade reconciliation: book matches account. ✓")
            return target_weights

        for sym in sorted(manually_removed):
            log.warning(
                f"RECONCILE PRE-TRADE: {sym} was in bot book but is not in account "
                f"— assumed manually removed. Dropping from target this rebalance."
            )
            target_weights.pop(sym, None)

        return target_weights

    def _reconcile_post_trade(self, expected_symbols, effective_pv):
        """
        After trades execute, wait briefly for orders to settle then re-read
        actual positions.  Log any symbols we expected to hold that are
        missing (order failure, partial fill, etc.).

        Conservative action: log only — the next rebalance auto-corrects by
        reading Webull's actual state.  We never place extra corrective orders.
        """
        import time
        log.info("Post-trade reconciliation: waiting 10 s for orders to settle…")
        time.sleep(10)

        try:
            actual = self.get_position_details(effective_pv)
        except Exception as e:
            log.warning(f"Post-trade reconciliation: could not fetch positions ({e}). Skipping.")
            return

        actual_symbols = set(actual.keys())
        missing = expected_symbols - actual_symbols

        if missing:
            log.warning(
                f"RECONCILE POST-TRADE: expected to hold {sorted(missing)} after trades "
                f"but not found in account. Possible order failures or partial fills. "
                f"Will auto-correct next rebalance."
            )
        else:
            log.info("Post-trade reconciliation: all expected positions confirmed in account. ✓")

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def compute_target_weights(self, predictions_today, allocation=None):
        """
        Compute target weights from today's model predictions and bucket allocation.

        Parameters
        ----------
        predictions_today : DataFrame with columns: symbol, clf_prob, reg_pred
        allocation : dict with venture_pct, safety_pct, hedge_pct, cash_pct, or None.
                     When None, behaves as pure venture (original behaviour).

        Returns
        -------
        dict {symbol: target_weight}  — weights are fractions of effective_pv
        """
        venture_pct = allocation["venture_pct"] if allocation else 1.0
        safety_pct  = allocation["safety_pct"]  if allocation else 0.0
        hedge_pct   = allocation["hedge_pct"]   if allocation else 0.0

        candidates = predictions_today[
            (predictions_today["clf_prob"] >= CLF_PROB_THRESHOLD) &
            (predictions_today["reg_pred"] > 0)
        ].copy()

        candidates["kelly"] = candidates.apply(
            lambda r: kelly_fraction(r["clf_prob"], r["reg_pred"], half_kelly=HALF_KELLY),
            axis=1,
        )
        candidates = candidates[candidates["kelly"] > 0]

        target = {}

        if not candidates.empty:
            candidates = candidates.sort_values("kelly", ascending=False).head(MAX_VENTURE_POSITIONS)
            total_kelly = candidates["kelly"].sum()
            for _, row in candidates.iterrows():
                # Proportional Kelly: each position gets its natural share of venture_pct.
                # Positions below the minimum floor are excluded; remaining capital stays cash.
                weight = (row["kelly"] / total_kelly) * venture_pct
                if weight < MIN_POSITION_WEIGHT:
                    continue
                target[row["symbol"]] = weight

        if safety_pct > 0:
            target.update(_etf_bucket_weights(SAFETY_ETFS, predictions_today, safety_pct))

        if hedge_pct > 0:
            target.update(_etf_bucket_weights(HEDGE_ETFS, predictions_today, hedge_pct))

        return target

    def rebalance(self, predictions_today, allocation=None):
        """
        Main entry point. Computes target weights from predictions and
        executes the minimum set of trades to reach them.

        Parameters
        ----------
        predictions_today : DataFrame with columns: symbol, clf_prob, reg_pred
        allocation : dict with venture_pct, safety_pct, hedge_pct, cash_pct, or None.
                     When None, treats all capital as venture (original behaviour).
        """
        log.info("=" * 55)
        log.info("Starting rebalance")

        # Emergency stop — checked first, before anything else
        cmds_check = load_commands()
        if cmds_check.get("emergency_stop"):
            log.error("=" * 55)
            log.error("EMERGENCY STOP IS ACTIVE — all trading halted.")
            log.error("To re-enable: python controls.py clear-emergency")
            log.error("=" * 55)
            return

        # Halt if a previous run logged an error — requires manual clear_error() call
        stats = load_stats()
        if stats.get("status") == "error":
            log.error("Previous run had an error — trading halted.")
            log.error(f"Reason: {stats.get('error_message')}")
            log.error("Fix the issue then run: python -c \"from dashboard_logger import clear_error; clear_error()\"")
            return

        try:
            self._rebalance(predictions_today, allocation)
        except Exception as e:
            log.error(f"Rebalance failed — {type(e).__name__}: {e}")
            raise

    def _rebalance(self, predictions_today, allocation=None):
        """Internal rebalance logic (called by rebalance after error-state check)."""
        # Load live commands (capital cap, manual symbols, force-sells)
        cmds = load_commands()
        cap  = cmds.get("max_capital", MAX_CAPITAL)

        account         = self.get_account_snapshot()
        portfolio_value = account["net_account_value"] if cap is None else min(account["net_account_value"], cap)
        cap_str         = "no limit" if cap is None else f"${cap:,.2f}"
        log.info(f"Capital cap: {cap_str}  |  Using: ${portfolio_value:,.2f}")
        if allocation:
            log.info(
                f"Allocation — venture: {allocation['venture_pct']:.0%}  "
                f"safety: {allocation['safety_pct']:.0%}  "
                f"hedge: {allocation['hedge_pct']:.0%}  "
                f"cash: {allocation['cash_pct']:.0%}"
            )

        position_details = self.get_position_details(portfolio_value)
        manual           = set(cmds.get("manual_symbols", []))

        # Split positions into bot-managed and manual
        manual_details = {sym: d for sym, d in position_details.items() if sym in manual}
        bot_details    = {sym: d for sym, d in position_details.items() if sym not in manual}

        # Effective portfolio value = cap minus capital tied up in manual positions.
        # Manual positions occupied part of the bot's original allocation, so they
        # must be subtracted before the bot sizes any new trades.
        manual_market_value  = sum(d["market_value"] for d in manual_details.values())
        effective_pv         = max(portfolio_value - manual_market_value, 0)
        log.info(f"Effective bot capital: ${effective_pv:,.2f}  "
                 f"(cap ${portfolio_value:,.2f} − manual ${manual_market_value:,.2f})")

        # Recompute bot positions' weights against effective portfolio value
        if effective_pv > 0:
            for d in bot_details.values():
                d["weight"] = round(d["market_value"] / effective_pv, 4)

        current_weights = {sym: d["weight"] for sym, d in bot_details.items()}
        target_weights  = self.compute_target_weights(predictions_today, allocation)

        # Remove manual symbols from target (bot will not buy/sell them)
        if manual:
            log.info(f"Manual control active for: {sorted(manual)}")
            for sym in manual:
                target_weights.pop(sym, None)

        # Apply force-sells: override target to 0 (full exit) then clear the list
        force_sell = set(cmds.get("force_sell", []))
        if force_sell:
            log.info(f"Force-selling: {sorted(force_sell)}")
            for sym in force_sell:
                target_weights[sym] = 0.0
            cmds["force_sell"] = []
            save_commands(cmds)

        # --- Fetch current prices for held positions (needed for stop-loss check) ---
        current_prices = {}
        for sym in current_weights:
            try:
                current_prices[sym] = self.get_current_price(sym)
            except Exception as e:
                log.warning(f"Could not fetch price for {sym}: {e} — skipping stop-loss check for this symbol")

        # --- Trailing stop-loss: force full exit on any position down >= threshold ---
        stopped = check_stop_losses(current_prices)
        for sym in stopped:
            target_weights[sym] = 0.0   # override model — full exit

        # --- Pre-trade reconciliation: detect manually removed bot positions ---
        # Any symbol in the last book (stats.json) that is no longer in the account
        # was manually removed between rebalances.  Drop it from target so the bot
        # doesn't immediately re-buy it.
        target_weights = self._reconcile_pre_trade(set(bot_details.keys()), target_weights)

        # --- Safety checks (VIX circuit breaker, drawdown halt, sanity checks) ---
        action, reason, scale = run_checks(portfolio_value, predictions_today, target_weights)
        if action == "HALT":
            log.warning(f"SAFEGUARD HALT: {reason}")
            log.warning("Selling all positions and skipping new buys.")
            target_weights = {}   # empty target → all current holdings become sells
        elif action == "REDUCE":
            log.warning(f"SAFEGUARD REDUCE: {reason}")
            target_weights = {sym: w * scale for sym, w in target_weights.items()}
        else:
            log.info(f"Safeguards OK: {reason}")

        log.info(f"Bot holdings (current): {list(current_weights.keys())}")
        log.info(f"Bot holdings (target) : {list(target_weights.keys())}")

        all_symbols = set(current_weights) | set(target_weights)
        sells, buys = [], []

        for symbol in all_symbols:
            current_w = current_weights.get(symbol, 0.0)
            target_w  = target_weights.get(symbol, 0.0)
            delta     = target_w - current_w

            if abs(delta) <= REBALANCE_THRESHOLD:
                continue  # within threshold, no trade needed

            dollar_delta = abs(delta) * effective_pv   # size trades against effective capital only
            if delta < 0:
                sells.append((symbol, dollar_delta))
            else:
                buys.append((symbol, dollar_delta))

        executed_trades = []

        if not sells and not buys:
            log.info("No trades needed — all positions within rebalance threshold")
        else:
            # Execute sells first to free up cash before buys
            log.info(f"Sells: {len(sells)}  Buys: {len(buys)}")
            for symbol, dollar_amount in sells:
                price = self.get_current_price(symbol)
                trade = self._place_order(symbol, "SELL", dollar_amount, price)
                if trade:
                    executed_trades.append(trade)

            # Re-fetch cash after sells — in a cash account, same-day sell proceeds
            # are not settled (T+1), so only pre-existing cash is available for buys.
            # Execute buys in descending order of size (highest Kelly first) and stop
            # when settled cash is exhausted.
            if buys:
                settled_cash = self.get_account_snapshot()["cash_balance"]
                log.info(f"Settled cash for buys: ${settled_cash:,.2f}")
                remaining = settled_cash
                for symbol, dollar_amount in sorted(buys, key=lambda x: x[1], reverse=True):
                    required = dollar_amount * 1.02  # Webull requires 2% buffer
                    if remaining < required:
                        log.warning(
                            f"Skipping BUY {symbol} (~${dollar_amount:.0f}) — "
                            f"need ${required:.0f}, only ${remaining:.0f} settled cash left"
                        )
                        continue
                    price = self.get_current_price(symbol)
                    trade = self._place_order(symbol, "BUY", dollar_amount, price)
                    if trade:
                        executed_trades.append(trade)
                        remaining -= dollar_amount

        # --- Post-trade reconciliation: confirm expected positions exist ---
        # Skip in dry-run (no real orders placed) and when there were no trades.
        if not self.dry_run and executed_trades:
            expected_after = {sym for sym, w in target_weights.items() if w > 0}
            self._reconcile_post_trade(expected_after, effective_pv)

        # Update high-water marks using actual post-trade positions, not target weights.
        # target_weights excludes positions kept below the rebalance threshold, so using
        # it would strip stop-loss protection from small but genuinely held positions.
        actual_after = set(self.get_position_details(effective_pv).keys()) - set(cmds.get("manual_symbols", []))
        update_position_highs(current_prices, actual_after)

        # Log to dashboard — pass bot positions and manual positions separately
        log_rebalance(account, effective_pv, bot_details, manual_details, executed_trades)

        log.info("Rebalance complete")
        log.info("=" * 55)


# --- Quick connectivity test ---
if __name__ == "__main__":
    trader = Trader(dry_run=True)

    # Verify we can reach the API and fetch account info
    log.info("Testing API connectivity...")
    value = trader.get_portfolio_value()
    log.info(f"Portfolio value: ${value:,.2f}")

    weights = trader.get_current_weights(value)
    log.info(f"Current positions: {weights}")

    # Simulate predictions from a model
    import pandas as pd
    mock_predictions = pd.DataFrame([
        {"symbol": "AAPL", "clf_prob": 0.72, "reg_pred": 0.012},
        {"symbol": "NVDA", "clf_prob": 0.65, "reg_pred": 0.008},
        {"symbol": "MSFT", "clf_prob": 0.55, "reg_pred": 0.003},
    ])

    log.info("\nMock predictions:")
    log.info(mock_predictions.to_string(index=False))

    target = trader.compute_target_weights(mock_predictions)
    log.info(f"\nTarget weights: { {k: f'{v:.1%}' for k, v in target.items()} }")

    log.info("\nSimulating rebalance (dry run)...")
    trader.rebalance(mock_predictions)
