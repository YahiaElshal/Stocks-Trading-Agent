import json
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest

from data_pipeline.fetch_data import fetch_yahoo_data
from data_pipeline.preprocess import preprocess_data
from data_pipeline.train import build_and_train_lstm_model, save_model
from analysis.utils import init_subreddits_for_ticker

from logger_config import logger

# Load Secrets from secrets.json
with open("secrets.json", "r") as secrets_file:
    secrets = json.load(secrets_file)
ALPACA_API_KEY = secrets["alpaca"]["api_key"]
ALPACA_SECRET_KEY = secrets["alpaca"]["secret_key"]

# Create the historical data client
client = StockHistoricalDataClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY)

def get_last_trade_price(symbol) -> float:
    try:
        request_params = StockLatestTradeRequest(symbol_or_symbols=symbol)
        latest_trade = client.get_stock_latest_trade(request_params)
        last_price = latest_trade[symbol].price  # This is the last traded price
        return last_price
    except Exception as e:
        logger.error(f"Error fetching last trade price for {symbol}: {e}")
        return None

def init_new_ticker(ticker: str, company: str, start_date: str = "2020-01-01", end_date: str = "2025-4-20"):
    """
    Initialize a new ticker symbol for trading.
    This function can be expanded to include more complex initialization logic.
    """

    # Fetch historical data for the new ticker
    data = fetch_yahoo_data(ticker, start_date, end_date)
    if data is None:
        logger.error(f"Error fetching last trade price for {ticker}")
        return
    
    # Preprocess the data
    processed_data = preprocess_data(data, ticker=ticker)
    if processed_data is None:
        logger.error(f"Failed to preprocess data for {ticker}.")
        return
    
    # Build and train the LSTM model
    model = build_and_train_lstm_model(
        x_train=processed_data['x_train'],
        y_train=processed_data['y_train'],
    )
    if model is None:
        logger.error(f"Failed to build and train model for {ticker}.")
        return
    
    # Save the model
    save_model(model, f"models/{ticker}.keras")

    logger.info(f"Model for {ticker} saved successfully.")
    # Initialize subreddits for the ticker
    subs = init_subreddits_for_ticker(ticker, company)
    if subs is None:
        logger.error(f"Failed to initialize subreddits for {ticker}.")
        return
    logger.info(f"Subreddits for {ticker} initialized successfully.")
    logger.debug(f"Subreddits for {ticker}: {subs}")

    logger.info(f"Initialized new ticker: {ticker}")

# Example usage
if __name__ == "__main__":
    price = get_last_trade_price("AAPL")
    if price is not None:
        print(f"Last trade price for AAPL: ${price:.2f}")
    else:
        print("Failed to fetch last trade price.")

    init_new_ticker("AAPL")