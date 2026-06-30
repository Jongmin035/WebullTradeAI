"""
Pre-trade safety checks.

Checks performed before every rebalance:
  1. VIX circuit breaker  — VIX >= 40: sell all; VIX >= 25: halve weights
  2. Drawdown halt         — portfolio down >= 15% from peak: sell all
  3. Prediction sanity     — clf_prob in [0, 1]
  4. Weight sanity         — total allocation <= 100%, no single position > 20%

Usage:
    from safeguards import run_checks
    action, reason, scale = run_checks(portfolio_value, predictions, target_weights)
    # action : "HALT" | "REDUCE" | "OK"
    # reason : human-readable explanation logged and emailed on HALT
    # scale  : multiply all target_weights by this before trading
    #          1.0 = normal, 0.5 = reduced (high VIX), 0.0 = halt (sell all)
"""

import json
import os
import sys
import logging
import numpy as np
import yfinance as yf

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _d in (_src, os.path.join(_src, "core"), os.path.join(_src, "pipeline")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

log = logging.getLogger(__name__)

# --- Config ---
VIX_HALT_THRESHOLD   = 40    # VIX at or above → sell all positions, no new trades
VIX_REDUCE_THRESHOLD = 25    # VIX at or above → halve Kelly weights
MAX_DRAWDOWN         = 0.15  # 15% drop from peak → sell all, halt trading
MAX_SINGLE_WEIGHT    = 0.20  # 20% max per position
ATR_MULTIPLIER       = 2.0   # stop triggers when price falls 2×ATR14 below its high-water mark
ATR_FALLBACK         = 0.08  # fallback trailing stop % if ATR fetch fails
ATR_PERIOD           = 14
PEAK_FILE            = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "state", "peak_portfolio_value.json")
POSITION_HIGHS_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "state", "position_highs.json")


# --- VIX ---

def get_vix():
    """Fetch the latest VIX close from yfinance."""
    hist = yf.Ticker("^VIX").history(period="2d")
    if hist.empty:
        raise RuntimeError("Could not fetch VIX from yfinance")
    return float(hist["Close"].iloc[-1])


# --- Peak portfolio value (for drawdown tracking) ---

def load_peak(current_value):
    """Load the saved all-time-high portfolio value. Returns current_value if no history."""
    if os.path.exists(PEAK_FILE):
        with open(PEAK_FILE) as f:
            peak = json.load(f)["peak"]
        if peak is not None:
            return float(peak)
    return current_value


def save_peak(peak):
    with open(PEAK_FILE, "w") as f:
        json.dump({"peak": peak}, f)


def reset_peak(new_value=None):
    """
    Manually reset the peak after recovering from a drawdown halt.
    Call this before re-enabling trading: reset_peak(current_portfolio_value).
    """
    if new_value is not None:
        save_peak(new_value)
        log.info(f"Peak reset to ${new_value:,.2f}")
    elif os.path.exists(PEAK_FILE):
        os.remove(PEAK_FILE)
        log.info("Peak file removed — will reset on next rebalance.")


# --- Trailing stop-loss ---

def load_position_highs():
    """Load the saved high-water mark price for each held position."""
    if os.path.exists(POSITION_HIGHS_FILE):
        with open(POSITION_HIGHS_FILE) as f:
            return json.load(f)
    return {}


def save_position_highs(highs):
    with open(POSITION_HIGHS_FILE, "w") as f:
        json.dump(highs, f)


def _compute_atr(symbol):
    """Fetch ATR14 for symbol via yfinance. Returns None on failure."""
    try:
        hist = yf.Ticker(symbol).history(period=f"{ATR_PERIOD + 6}d")
        if len(hist) < ATR_PERIOD + 1:
            return None
        high  = hist["High"].values
        low   = hist["Low"].values
        close = hist["Close"].values
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )
        return float(tr[-ATR_PERIOD:].mean())
    except Exception:
        return None


def check_stop_losses(current_prices):
    """
    Compare each held position's current price against its ATR-based stop.
    Stop price = high_water_mark − ATR_MULTIPLIER × ATR14.
    Falls back to ATR_FALLBACK percentage if ATR fetch fails.

    Returns a dict {symbol: drawdown_pct} for positions that breached their stop.
    """
    highs = load_position_highs()
    triggered = {}
    new_positions = []

    for sym, price in current_prices.items():
        if sym not in highs:
            # First time seeing this position — start tracking from today's price.
            # Don't trigger a stop yet: we have no historical reference.
            highs[sym] = price
            new_positions.append(sym)
            continue
        high = highs[sym]
        atr  = _compute_atr(sym)
        if atr is not None:
            stop_price = high - ATR_MULTIPLIER * atr
        else:
            stop_price = high * (1.0 - ATR_FALLBACK)
        if price <= stop_price:
            drawdown = (high - price) / high
            triggered[sym] = drawdown
            log.warning(
                f"STOP-LOSS {sym}: price ${price:.2f} ≤ stop ${stop_price:.2f} "
                f"(high ${high:.2f}, drawdown {drawdown:.1%}) — forcing full exit"
            )

    if new_positions:
        log.info(f"Stop-loss: started tracking new positions at today's price: {new_positions}")
        save_position_highs(highs)

    return triggered


def update_position_highs(current_prices, held_symbols):
    """
    Update the high-water mark for each still-held symbol and remove exited ones.
    Call this AFTER executing trades so exited positions are correctly dropped.

    Parameters
    ----------
    current_prices : dict {symbol: current_price}
    held_symbols   : set/list of symbols still held after today's rebalance.
    """
    highs = load_position_highs()

    # Remove symbols that were exited
    for sym in list(highs.keys()):
        if sym not in held_symbols:
            del highs[sym]

    # Update (or initialise) high-water mark for held positions
    for sym in held_symbols:
        price = current_prices.get(sym)
        if price is None:
            continue
        highs[sym] = max(highs.get(sym, price), price)

    save_position_highs(highs)


# --- Individual checks ---

def _check_vix(vix):
    if vix >= VIX_HALT_THRESHOLD:
        return "HALT", f"VIX={vix:.1f} >= {VIX_HALT_THRESHOLD} — crash mode, selling all positions"
    if vix >= VIX_REDUCE_THRESHOLD:
        return "REDUCE", f"VIX={vix:.1f} >= {VIX_REDUCE_THRESHOLD} — elevated fear, halving position sizes"
    return "OK", f"VIX={vix:.1f} (normal)"


def _check_drawdown(current_value, peak):
    drawdown = (peak - current_value) / peak
    if drawdown >= MAX_DRAWDOWN:
        return "HALT", (
            f"Portfolio down {drawdown:.1%} from peak ${peak:,.2f} — "
            f"halting trading. Call safeguards.reset_peak() to re-enable."
        )
    return "OK", f"Drawdown={drawdown:.1%} from peak ${peak:,.2f}"


def _check_predictions(predictions):
    bad = ~((predictions["clf_prob"] >= 0) & (predictions["clf_prob"] <= 1))
    if bad.any():
        bad_symbols = predictions.loc[bad, "symbol"].tolist()
        return False, f"clf_prob out of [0,1] for {bad_symbols}"
    return True, "OK"


_BUCKET_ETFS = {"SPY", "XLP", "XLU", "GLDM", "SH", "SQQQ"}  # exempt from single-position cap

def _check_weights(target_weights):
    total = sum(target_weights.values())
    if total > 1.0 + 1e-6:
        return False, f"Total allocation {total:.1%} exceeds 100%"
    for sym, w in target_weights.items():
        if sym in _BUCKET_ETFS:
            continue
        if w > MAX_SINGLE_WEIGHT:
            return False, f"{sym} weight {w:.1%} exceeds {MAX_SINGLE_WEIGHT:.0%} single-position max"
    return True, "OK"


# --- Main entry point ---

def run_checks(portfolio_value, predictions=None, target_weights=None):
    """
    Run all pre-trade safety checks.

    Parameters
    ----------
    portfolio_value : float   Current total portfolio value in USD.
    predictions     : DataFrame with columns [symbol, clf_prob, reg_pred], or None.
    target_weights  : dict {symbol: weight} from compute_target_weights(), or None.

    Returns
    -------
    action : str    "HALT" | "REDUCE" | "OK"
    reason : str    Human-readable explanation (log on HALT).
    scale  : float  Multiply all target_weights by this before executing trades.
                    1.0 = normal, 0.5 = halved (elevated VIX), 0.0 = halt (sell all).
    """
    target_weights = target_weights or {}
    scale = 1.0

    # 1. VIX circuit breaker
    try:
        vix = get_vix()
        action, reason = _check_vix(vix)
        log.info(f"Safeguard VIX check: {reason}")
        if action == "HALT":
            return "HALT", reason, 0.0
        if action == "REDUCE":
            scale = 0.5
    except Exception as e:
        log.warning(f"Safeguard VIX check skipped — could not fetch VIX: {e}")

    # 2. Drawdown halt
    peak = max(load_peak(portfolio_value), portfolio_value)
    save_peak(peak)
    action, reason = _check_drawdown(portfolio_value, peak)
    log.info(f"Safeguard drawdown check: {reason}")
    if action == "HALT":
        return "HALT", reason, 0.0

    # 3. Prediction sanity
    if predictions is not None and not predictions.empty:
        ok, reason = _check_predictions(predictions)
        if not ok:
            return "HALT", f"Bad model output — {reason}", 0.0

    # 4. Weight sanity
    if target_weights:
        ok, reason = _check_weights(target_weights)
        if not ok:
            return "HALT", f"Weight violation — {reason}", 0.0

    if scale < 1.0:
        return "REDUCE", f"VIX elevated — position sizes halved", scale
    return "OK", "All checks passed", 1.0
