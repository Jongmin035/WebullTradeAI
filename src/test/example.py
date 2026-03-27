import json
import uuid
from time import sleep

import webull
import webull.core
webull.__version__ = webull.core.__version__  # fix SDK bug: __import__('webull.core') returns top-level webull

from webull.core.client import ApiClient
from webull.trade.trade_client import TradeClient

# PRD env host: api.webull.com;
# Test env host: us-openapi-alb.uat.webullbroker.com
optional_api_endpoint = "us-openapi-alb.uat.webullbroker.com"
your_app_key = "eecbf4489f460ad2f7aecef37b267618"
your_app_secret = "8abf920a9cc3cb7af3ea5e9e03850692"
region_id = "us"
account_id = "4BJITU00JUIVEDO5V3PRA5C5G8"
api_client = ApiClient(your_app_key, your_app_secret, region_id)
api_client.add_endpoint(region_id, optional_api_endpoint)


def call(label, fn, *args, **kwargs):
    """Call an API function, print the result, and swallow errors so the script continues."""
    try:
        res = fn(*args, **kwargs)
        if res.status_code == 200:
            print(f"{label}=" + json.dumps(res.json(), indent=4))
        else:
            print(f"{label} failed: HTTP {res.status_code} — {res.text}")
    except Exception as e:
        print(f"{label} error: {e}")


if __name__ == '__main__':
    trade_client = TradeClient(api_client)

    call("account_list", trade_client.account_v2.get_account_list)
    call("account_balance", trade_client.account_v2.get_account_balance, account_id)
    call("account_position", trade_client.account_v2.get_account_position, account_id)

    # --- Equity order ---
    preview_orders = [
        {
            "client_order_id": uuid.uuid4().hex,
            "combo_type": "NORMAL",
            "symbol": "AAPL",
            "instrument_type": "EQUITY",
            "market": "US",
            "order_type": "MARKET",
            "quantity": "1",
            "support_trading_session": "N",
            "side": "BUY",
            "time_in_force": "DAY",
            "entrust_type": "QTY"
        }
    ]
    call("preview_res", trade_client.order_v2.preview_order,
         account_id=account_id, preview_orders=preview_orders)

    client_order_id = uuid.uuid4().hex
    new_orders = [
        {
            "client_order_id": client_order_id,
            "combo_type": "NORMAL",
            "symbol": "AAPL",
            "instrument_type": "EQUITY",
            "market": "US",
            "order_type": "LIMIT",
            "limit_price": "188",
            "quantity": "1",
            "support_trading_session": "N",
            "side": "BUY",
            "time_in_force": "DAY",
            "entrust_type": "QTY",
        }
    ]
    call("place_order_res", trade_client.order_v2.place_order,
         account_id=account_id, new_orders=new_orders)
    sleep(5)

    modify_orders = [
        {
            "client_order_id": client_order_id,
            "quantity": "100",
            "limit_price": "200"
        }
    ]
    call("replace_order_res", trade_client.order_v2.replace_order,
         account_id=account_id, modify_orders=modify_orders)
    sleep(5)

    call("cancel_order_res", trade_client.order_v2.cancel_order_v2,
         account_id=account_id, client_order_id=client_order_id)

    call("order_history_res", trade_client.order_v2.get_order_history_request,
         account_id=account_id)

    call("order_detail", trade_client.order_v2.get_order_detail,
         account_id=account_id, client_order_id=client_order_id)

    # --- Option order ---
    # For option order inquiries, use the V2 query interface: order_v2.get_order_detail
    option_client_order_id = uuid.uuid4().hex
    option_new_orders = [
        {
            "client_order_id": option_client_order_id,
            "combo_type": "NORMAL",
            "order_type": "LIMIT",
            "limit_price": "6.00",
            "quantity": "1",
            "option_strategy": "SINGLE",
            "side": "BUY",
            "time_in_force": "DAY",
            "entrust_type": "QTY",
            "instrument_type": "OPTION",
            "market": "US",
            "symbol": "AAPL",
            "legs": [
                {
                    "side": "BUY",
                    "quantity": "1",
                    "symbol": "AAPL",
                    "strike_price": "255",
                    "option_expire_date": "2026-04-01",
                    "instrument_type": "OPTION",
                    "option_type": "CALL",
                    "market": "US"
                }
            ]
        }
    ]

    call("preview_option", trade_client.order_v2.preview_option,
         account_id, option_new_orders)
    sleep(5)

    call("place_option", trade_client.order_v2.place_option,
         account_id, option_new_orders)
    sleep(5)

    # NOTE: replace_option consistently returns "invalid order_id" in UAT regardless of payload.
    # Likely a UAT bug — reported to Webull. Skipping for now.

    sleep(5)

    call("cancel_option", trade_client.order_v2.cancel_option,
         account_id, option_client_order_id)
