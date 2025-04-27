import alpaca_trade_api as tradeapi
from alpaca.trading.client import TradingClient
from collections import defaultdict
from datetime import datetime
import json
from logger_config import logger

# Load Secrets from secrets.json
with open("secrets.json", "r") as secrets_file:
    secrets = json.load(secrets_file)
API_KEY = secrets["alpaca"]["api_key"]
SECRET_KEY = secrets["alpaca"]["secret_key"]
BASE_URL = secrets["alpaca"]["base_url"]

# Load configuration from config.json
with open("config.json", "r") as config_file:
    config = json.load(config_file)
limit_order_price_buffer = float(config.get("limit_order_price_buffer", 1.007))  # Default to 0.7% if not specified
buy_price_rounding = int(config.get("buy_price_rounding", 2))  # Default to 2 decimal places if not specified

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
# Instead of a market order:


# Function to execute paper trades
def execute_trade(symbol, decision, quantity):
    try:
        if decision == "BUY":
            # Instead of a market order:
            # Use a limit order around the current ask:
            quote = api.get_latest_quote(symbol)
            buy_price = quote.ask_price  * limit_order_price_buffer  # Add a small buffer to the ask price
            # Clean the decimals to avoid issues with Alpaca
            buy_price = round(buy_price, buy_price_rounding)
            api.submit_order(
                symbol=symbol,
                qty=quantity,
                side="buy",
                type="limit",
                limit_price=buy_price,
                time_in_force="gtc"
            )
            logger.info(f"Submitted BUY order for {quantity} shares of {symbol}.")
        elif decision == "SELL":
            api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            logger.info(f"Submitted SELL order for {quantity} shares of {symbol}.")
        else:
            logger.info(f"No action taken for {symbol}. Decision: {decision}")
    except Exception as e:
        logger.error(f"Error executing trade for {symbol}: {e}")

if __name__ == "__main__":
    
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

    position = trading_client.get_all_positions()
    print("Current positions:")
    for pos in position:
        print(f"{pos.symbol}: {pos.qty} shares at {pos.current_price}")
    from datetime import datetime
    import pytz

    filled_orders = []

    for order in api.list_orders(status="closed", limit=500):
        if order.filled_at and order.filled_avg_price:
            # Safely parse date
            if isinstance(order.filled_at, str):
                filled_date = datetime.fromisoformat(order.filled_at).date()
            else:
                filled_date = order.filled_at.date()
            
            filled_orders.append({
                "symbol": order.symbol,
                "price": float(order.filled_avg_price),
                "qty": float(order.filled_qty),
                "side": order.side,
                "date": filled_date
            })
    from collections import defaultdict

    positions_by_entry = defaultdict(lambda: {"qty": 0, "total": 0})

    for o in filled_orders:
        key = (o["symbol"], o["price"], o["date"])
        positions_by_entry[key]["qty"] += o["qty"] if o["side"] == "buy" else -o["qty"]
        positions_by_entry[key]["total"] += o["qty"] * o["price"] if o["side"] == "buy" else -o["qty"] * o["price"]

    # Print the positions by entry
    print("Positions by entry:")
    for (symbol, price, date), data in positions_by_entry.items():
        print(f"{symbol} at {price} on {date}: {data['qty']} shares, total value: {data['total']}")
    

    print("Cash balance:", trading_client.get_account().cash)