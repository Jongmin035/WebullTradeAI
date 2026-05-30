"""
Bot control tool — manage capital allocation and position ownership.

Usage:
    python controls.py status
    python controls.py add-funds 500
    python controls.py remove-funds 300
    python controls.py take-control AAPL           # stop bot managing AAPL, keep position
    python controls.py take-control AAPL --sell    # stop bot managing AAPL, sell position too
    python controls.py release-control AAPL        # hand AAPL back to the bot
    python controls.py emergency-stop              # halt all trading, hand over all positions
    python controls.py clear-emergency             # re-enable trading after emergency stop
"""

import json
import os
import sys

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _d in (_src, os.path.join(_src, "core"), os.path.join(_src, "pipeline")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

COMMANDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "state", "commands.json")
STATS_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dashboard", "stats.json")

DEFAULT_COMMANDS = {
    "max_capital":      2000.0,
    "manual_symbols":   [],     # bot will not touch these symbols
    "force_sell":       [],     # sold at next rebalance, then cleared
    "emergency_stop":   False,  # if True, bot refuses to run at all
}


# --- Load / save ---

def load_commands():
    if os.path.exists(COMMANDS_FILE):
        with open(COMMANDS_FILE) as f:
            return json.load(f)
    return DEFAULT_COMMANDS.copy()


def save_commands(cmds):
    with open(COMMANDS_FILE, "w") as f:
        json.dump(cmds, f, indent=2)


def _latest_stats():
    """Return the most recent non-error history entry from stats.json, or None."""
    if not os.path.exists(STATS_FILE):
        return None, None
    with open(STATS_FILE) as f:
        stats = json.load(f)
    history = [h for h in stats.get("history", []) if not h.get("error")]
    latest  = history[-1] if history else None
    return stats, latest


class ControlError(Exception):
    pass


def _err(msg):
    raise ControlError(msg)


# --- Operations ---

def status():
    cmds  = load_commands()
    _, latest = _latest_stats()

    print("=" * 45)
    print("Bot Control Status")
    print("=" * 45)
    print(f"  Max capital    : ${cmds['max_capital']:,.2f}")
    if latest:
        print(f"  Net acct value : ${latest.get('net_account_value', 0):,.2f}")
        print(f"  Market value   : ${latest.get('market_value', 0):,.2f}")
        print(f"  Cash balance   : ${latest.get('cash_balance', 0):,.2f}")
        print(f"  Open P&L       : ${latest.get('open_pnl', 0):+,.2f}")
        print(f"  Day's P&L      : ${latest.get('day_pnl', 0):+,.2f}")
    print(f"  Manual symbols : {cmds['manual_symbols'] or 'none'}")
    print(f"  Pending sells  : {cmds['force_sell'] or 'none'}")
    print("=" * 45)


def add_funds(amount):
    if amount <= 0:
        _err("Amount must be positive.")

    cmds = load_commands()
    _, latest = _latest_stats()

    old_cap  = cmds["max_capital"]
    new_cap  = old_cap + amount

    if latest:
        net_value = latest.get("net_account_value", 0)
        mkt_value = latest.get("market_value", 0)
        # Available cash not already committed to the bot
        available = net_value - mkt_value - old_cap
        if amount > available + 1e-2:
            _err(
                f"Adding ${amount:,.2f} would exceed available funds.\n"
                f"  Net account value : ${net_value:,.2f}\n"
                f"  Already in bot    : ${old_cap:,.2f} (cap) / ${mkt_value:,.2f} (invested)\n"
                f"  Available to add  : ${max(available, 0):,.2f}"
            )

    cmds["max_capital"] = round(new_cap, 2)
    save_commands(cmds)
    print(f"OK  Max capital updated: ${old_cap:,.2f} → ${new_cap:,.2f}")
    print(f"    Takes effect at next rebalance.")


def remove_funds(amount):
    if amount <= 0:
        _err("Amount must be positive.")

    cmds = load_commands()
    _, latest = _latest_stats()

    old_cap = cmds["max_capital"]
    new_cap = old_cap - amount

    if new_cap < 0:
        _err(f"Cannot reduce below $0 (current cap: ${old_cap:,.2f}).")

    if latest:
        mkt_value = latest.get("market_value", 0)
        if new_cap < mkt_value - 1e-2:
            _err(
                f"New cap ${new_cap:,.2f} is less than current invested value ${mkt_value:,.2f}.\n"
                f"The bot cannot free that cash without selling positions.\n"
                f"Options:\n"
                f"  1. Remove less (max safe removal: ${max(old_cap - mkt_value, 0):,.2f})\n"
                f"  2. Use take-control --sell on positions first, then remove funds."
            )

    cmds["max_capital"] = round(new_cap, 2)
    save_commands(cmds)
    print(f"OK  Max capital updated: ${old_cap:,.2f} → ${new_cap:,.2f}")
    print(f"    Takes effect at next rebalance.")


def take_control(symbol, sell=False):
    symbol = symbol.upper()
    cmds   = load_commands()
    _, latest = _latest_stats()

    # Warn if symbol is not currently held
    if latest:
        positions = latest.get("positions", {})
        if symbol not in positions:
            print(f"Warning: {symbol} is not currently held by the bot.")

    if symbol not in cmds["manual_symbols"]:
        cmds["manual_symbols"].append(symbol)

    if sell and symbol not in cmds["force_sell"]:
        cmds["force_sell"].append(symbol)

    save_commands(cmds)
    if sell:
        print(f"OK  {symbol} will be sold at next rebalance and then managed manually.")
    else:
        print(f"OK  {symbol} handed to manual control — bot will no longer trade it.")
        print(f"    Existing position is kept as-is.")


def emergency_stop():
    cmds = load_commands()
    _, latest = _latest_stats()

    # Collect all positions currently on the bot's book
    bot_positions    = list((latest.get("positions", {}) if latest else {}).keys())
    manual_positions = list((latest.get("manual_positions", {}) if latest else {}).keys())
    all_held         = bot_positions + manual_positions

    # Hand everything over to manual control
    for sym in all_held:
        if sym not in cmds["manual_symbols"]:
            cmds["manual_symbols"].append(sym)

    cmds["emergency_stop"] = True
    save_commands(cmds)

    print("=" * 45)
    print("!!! EMERGENCY STOP ACTIVATED !!!")
    print("=" * 45)
    print("  Trading is halted immediately.")
    if all_held:
        print(f"  Positions handed to manual control: {all_held}")
    else:
        print("  No open positions found in last dashboard snapshot.")
        print("  If the bot holds positions not yet logged, run again")
        print("  after the next rebalance cycle completes.")
    print()
    print("  To re-enable trading:")
    print("    python controls.py clear-emergency")
    print("=" * 45)


def clear_emergency():
    cmds = load_commands()
    if not cmds.get("emergency_stop"):
        print("No emergency stop is active.")
        return
    cmds["emergency_stop"] = False
    save_commands(cmds)
    print("Emergency stop cleared — trading will resume at next rebalance.")
    print("Note: positions are still under manual control.")
    print("Use 'release-control SYMBOL' to hand them back to the bot.")


def release_control(symbol):
    symbol = symbol.upper()
    cmds   = load_commands()

    if symbol not in cmds["manual_symbols"]:
        print(f"Warning: {symbol} was not under manual control.")
    else:
        cmds["manual_symbols"].remove(symbol)

    # Also clear any pending force-sell in case user changes mind
    if symbol in cmds["force_sell"]:
        cmds["force_sell"].remove(symbol)

    save_commands(cmds)
    print(f"OK  {symbol} released back to bot. Takes effect at next rebalance.")


# --- CLI ---

def main():
    args = sys.argv[1:]

    try:
        if not args or args[0] == "status":
            status()

        elif args[0] == "add-funds":
            if len(args) < 2:
                _err("Usage: python controls.py add-funds <amount>")
            add_funds(float(args[1]))

        elif args[0] == "remove-funds":
            if len(args) < 2:
                _err("Usage: python controls.py remove-funds <amount>")
            remove_funds(float(args[1]))

        elif args[0] == "take-control":
            if len(args) < 2:
                _err("Usage: python controls.py take-control <SYMBOL> [--sell]")
            sell = "--sell" in args
            take_control(args[1], sell=sell)

        elif args[0] == "release-control":
            if len(args) < 2:
                _err("Usage: python controls.py release-control <SYMBOL>")
            release_control(args[1])

        elif args[0] == "emergency-stop":
            emergency_stop()

        elif args[0] == "clear-emergency":
            clear_emergency()

        else:
            print(__doc__)
            sys.exit(1)

    except ControlError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
