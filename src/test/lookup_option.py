"""
Lookup valid AAPL option expiration dates and near-the-money strikes
from Webull's production quote API, so we can use real values in
the UAT option order test.
"""

from webull import webull

SYMBOL = "AAPL"

def fetch_expiration_dates(wb, symbol):
    dates = wb.get_options_expiration_dates(stock=symbol)
    return [d["date"] for d in dates]

def fetch_nearest_atm_option(wb, symbol, expire_date):
    """Return the nearest ATM call option for a given expiration date."""
    chain = wb.get_options(stock=symbol, expireDate=expire_date, direction="call")
    if not chain:
        return None

    # Get current price to find ATM strike
    quote = wb.get_quote(stock=symbol)
    current_price = float(quote.get("close") or quote.get("pPrice") or 0)

    # Find strike closest to current price
    best = min(
        chain,
        key=lambda x: abs(float(x["strikePrice"]) - current_price)
    )
    return {
        "expire_date": expire_date,
        "strike_price": best["strikePrice"],
        "current_price": current_price,
        "call": best.get("call", {}),
    }


if __name__ == "__main__":
    wb = webull()

    print(f"=== Available expiration dates for {SYMBOL} ===")
    dates = fetch_expiration_dates(wb, SYMBOL)
    for d in dates[:10]:  # show first 10
        print(f"  {d}")

    if dates:
        nearest = dates[0]
        print(f"\n=== Nearest ATM CALL for {SYMBOL} expiring {nearest} ===")
        result = fetch_nearest_atm_option(wb, SYMBOL, nearest)
        if result:
            print(f"  Current price : {result['current_price']}")
            print(f"  Strike price  : {result['strike_price']}")
            print(f"  Expire date   : {result['expire_date']}")
            call = result["call"]
            print(f"  Ask price     : {call.get('askList', [{}])[0].get('price', 'N/A')}")
            print(f"  Bid price     : {call.get('bidList', [{}])[0].get('price', 'N/A')}")
            print(f"\n--- Use these values in example.py option order ---")
            print(f'  "option_expire_date": "{result["expire_date"]}"')
            print(f'  "strike_price": "{result["strike_price"]}"')
