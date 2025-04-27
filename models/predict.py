import os
import numpy as np
import datetime
import tensorflow as tf
import joblib

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import random
from datetime import timedelta, datetime
import pandas as pd
from data_pipeline.fetch_data import fetch_yahoo_data
import json

# Load Secrets from secrets.json
with open("secrets.json", "r") as secrets_file:
    secrets = json.load(secrets_file)
ALPACA_API_KEY = secrets["alpaca"]["api_key"]
ALPACA_SECRET_KEY = secrets["alpaca"]["secret_key"]

def predict_price(ticker, date=None, api_key=ALPACA_API_KEY, api_secret=ALPACA_SECRET_KEY, sequence_length=100, model_path=None, mode="live", include_today=True):
    """
    Predicts the stock price using a trained model and real-time or historical data.

    Parameters:
    - ticker: Stock ticker symbol (e.g., 'AAPL')
    - date: The date for which to make the prediction (datetime.date object, required for backtesting)
    - api_key: Alpaca API key
    - api_secret: Alpaca API secret
    - sequence_length: Length of sequence input for LSTM (default: 100)
    - model_path: Path to the saved model file (default: None, will use f"{ticker}.model")
    - mode: "live" for live testing, "backtest" for backtesting (default: "live")
    - include_today: Whether to include today's incomplete data if market is open (default: True, only applies in live mode)

    Returns:
    - Dictionary with prediction details
    """
    try:
        # Initialize Alpaca clients
        stock_client = StockHistoricalDataClient(api_key, api_secret)

        # Determine model and scaler paths
        if model_path is None and mode == "live":
            model_path = f"models/{ticker}.keras"
            scaler_path = f"models/{ticker}_scaler.pkl"
        elif model_path is None and mode == "backtest":
            model_path = f"models/{ticker}_backtest.keras"
            scaler_path = f"models/{ticker}_backtest_scaler.pkl"
        # Check if model and scaler exist
        if not os.path.exists(model_path):
            return {"error": f"Model not found at {model_path}"}
        if not os.path.exists(scaler_path):
            return {"error": f"Scaler not found at {scaler_path}"}

        # Load model and scaler
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)

        # # Handle live mode
        if mode == "live":
            now = datetime.now()
            market_open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
            is_weekday = now.weekday() < 5  # Monday to Friday are weekdays
            market_open = is_weekday and market_open_time <= now <= market_close_time

            # Get historical data (more than needed to ensure we have enough after filtering)
            start_date = (now - timedelta(days=sequence_length * 2)).strftime('%Y-%m-%d')
            end_date = now.strftime('%Y-%m-%d')

            bars_request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute if market_open else TimeFrame.Day,
                start=start_date,
                end=end_date
            )

            bars_data = stock_client.get_stock_bars(bars_request).df

            if len(bars_data) == 0:
                return {"error": f"No data available for {ticker}"}

            # Extract the data for the specific ticker
            if bars_data.index.nlevels > 1:
                bars_data = bars_data.loc[ticker].copy()

            # Format to match training data format
            data = bars_data.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # If market is open and we don't want to include today's data, filter it out
            if market_open and not include_today:
                today_str = now.strftime('%Y-%m-%d')
                data = data[data.index.date.astype(str) != today_str]

        # Handle backtest mode
        elif mode == "backtest":
            if date is None:
                return {"error": "Date is required for backtesting mode."}

            # Ensure the date is a datetime.date object
            if isinstance(date, str):
                date = datetime.strptime(date, '%Y-%m-%d').date()

            # Get historical data up to the given date
            end_date = date
            start_date = date - timedelta(days=sequence_length * 2)
            df = fetch_yahoo_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

            # Ensure the index is a DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                else:
                    raise ValueError("DataFrame must have a 'Date' column to set as index.")

            # Format to match training data format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            data = df

        else:
            return {"error": f"Invalid mode: {mode}. Use 'live' or 'backtest'."}
        
        # Extract features in the correct order
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data[features]

        # Check if we have enough data
        if len(data) < sequence_length:
            return {"error": f"Not enough data for sequence length {sequence_length}. Have {len(data)} days."}

        # Get last price for reference
        last_date = data.index[-1]
        last_price = data['Close'].iloc[-1]

        # Remove feature names before scaling
        data_values = data.values  # Convert to numpy array without feature names

        # Scale the data
        scaled_data = scaler.transform(data_values)

        # Create input sequence (last sequence_length days)
        input_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, len(features))

        # Generate prediction
        prediction_scaled = model.predict(input_sequence, verbose=0)

        # Prepare for inverse scaling
        dummy_array = np.zeros((1, len(features)))
        dummy_array[0, 3] = prediction_scaled[0, 0]  # 3 is the index of 'Close'

        # Inverse transform to get the actual price
        predicted_price = scaler.inverse_transform(dummy_array)[0, 3]

        # Calculate change
        price_change = predicted_price - last_price
        price_change_percent = (price_change / last_price) * 100

        # Result dictionary
        prediction_result = {
            "ticker": ticker,
            "last_price": last_price,
            "last_date": last_date.strftime('%Y-%m-%d'),
            "predicted_price": predicted_price,
            "price_change": price_change,
            "price_change_percent": price_change_percent,
            "timestamp": date 
        }

        return prediction_result

    except Exception as e:
        return {"error": str(e), "ticker": ticker}

if __name__ == "__main__":

    # Example usage

    ticker = 'TCOM'
    # date = "2024-11-07"  # String date
    # date = datetime.strptime(date, '%Y-%m-%d').date()  # Convert to datetime.date object

    # prediction = predict_price(ticker, date=date, mode="backtest")
    # print(f"Prediction for {ticker} on {date}:")
    # print(prediction)
    # print(predict_price(ticker)['predicted_price'])