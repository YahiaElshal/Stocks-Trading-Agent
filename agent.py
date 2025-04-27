import time
import threading
from datetime import datetime, timedelta
from data_pipeline.fetch_data import fetch_yahoo_data
from multi_factor_strategy import make_trade_decision
from analysis.reddit_sentiment import get_reddit_sentiment_score
from analysis.news_sentiment import get_news_sentiment_score
from analysis.technical_indicators import indicators_score, stock_with_indicators
from models.predict import predict_price
from alpaca_trading import execute_trade
from utils import get_last_trade_price
import praw
import backtrader as bt
import pandas as pd
import json
from alpaca.trading.client import TradingClient
import plotly.graph_objects as go
from logger_config import logger 

# Load Secrets from secrets.json
with open("secrets.json", "r") as secrets_file:
    secrets = json.load(secrets_file)
API_KEY = secrets["alpaca"]["api_key"]
SECRET_KEY = secrets["alpaca"]["secret_key"]
BASE_URL = secrets["alpaca"]["base_url"]
REDDIT_CLIENT_ID = secrets["reddit"]["client_id"]
REDDIT_CLIENT_SECRET = secrets["reddit"]["client_secret"]
REDDIT_USER_AGENT = secrets["reddit"]["user_agent"]

# Load configuration from config.json
with open("config.json", "r") as config_file:
    config = json.load(config_file)
sell_off_peak_drop_percent = float(config.get("sell_off_peak_drop_percent", 2))  # Default to 2% if not specified
hours_bet_decisions = float(config.get("hours_bet_decisions", 12))  # Default to 12 hour if not specified


class TradingAgent:
    def __init__(self, tickers, mode, risk=0.3, duration=90, cash_balance=100000):
        """
        Initialize the TradingAgent with the given settings.
        """
        self.tickers = tickers  # List of tickers
        self.mode = mode
        self.risk = risk
        self.duration = duration
        self.cash_balance = {ticker: cash_balance for ticker in tickers}  # Cash balance per ticker
        self.current_positions = {ticker: [] for ticker in tickers}  # Track positions as a list of dicts
        self.local_peaks = {ticker: None for ticker in tickers}  # Initialize local peaks for each ticker
        self.reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent= REDDIT_USER_AGENT
        )
        self.trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

        self.threads = []  # To keep track of ticker threads
        self.running = False  # To control the execution of threads

    def fetch_live_data(self, ticker):
        """
        Fetch live data for the given ticker.
        """
        try:
            # Fetch last trade price using utility function
            current_price = get_last_trade_price(ticker)

            # Fetch historical data from Yahoo Finance for indicators
            end_date = datetime.now()
            start_date = end_date - timedelta(days=201)  # Fetch 201 days earlier
            # Fetch data from Yahoo Finance
            df = fetch_yahoo_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            technical_indicators = indicators_score(df)  # Get technical indicators score

            # Fetch sentiment scores
            news_sentiment = get_news_sentiment_score(ticker, ticker, end_date)
            reddit_sentiment = get_reddit_sentiment_score(ticker, end_date, self.reddit)

            # Return all data
            return {
                "datetime": end_date.strftime('%Y-%m-%d %H:%M:%S'), # Format datetime for consistency
                "current_price": current_price,
                "technical_indicators": technical_indicators,  # Now a float
                "news_sentiment": news_sentiment,
                "reddit_sentiment": reddit_sentiment,
            }
        except Exception as e:
            logger.error(f"Error fetching live data for {ticker}: {e}")
            return None

    def make_paper_decision(self, ticker):
        """
        Make a trading decision for a given ticker in live paper trading.
        """
        try:
            # Fetch live data
            current_data = self.fetch_live_data(ticker)
            if not current_data:
                return

            # Fetch current position size using Alpaca API
            try:
                position = self.trading_client.get_all_positions(ticker)
                logger.info(f"Current position for {ticker}: {current_position_size} shares")
                current_position_size = int(position.qty)  # Convert to integer
                has_open_position = current_position_size > 0
            except Exception:
                # No position exists for this ticker
                current_position_size = 0
                has_open_position = False

            # Initialize or update the local peak close price
            if has_open_position:
                if ticker not in self.local_peaks or current_data["current_price"] > self.local_peaks[ticker]:
                    self.local_peaks[ticker] = current_data["current_price"]

                # Check if today's price is below the sell-off threshold
                if current_data["current_price"] < (1 - (sell_off_peak_drop_percent/100)) * self.local_peaks[ticker]:
                    logger.info(f"Exiting position for {ticker}: Current price ({current_data['current_price']}) is "
                          f"{sell_off_peak_drop_percent}% below the local peak ({self.local_peaks[ticker]}).")
                    decision = {"action": "SELL", "shares": current_position_size}
                else:
                    # Call the strategy to make a decision
                    predictions = predict_price(ticker)
                    predicted_price = predictions.get('predicted_price')
                    if predicted_price is None:
                        logger.error(f"Prediction failed for {ticker}.")
                        return

                    # Predict yesterday's price
                    yesterday_date = datetime.now() - timedelta(days=1)
                    yesterday_predictions = predict_price(ticker, yesterday_date, mode="backtest")
                    yesterday_predicted_price = yesterday_predictions.get('predicted_price')

                    decision = make_trade_decision(
                        ticker=ticker,
                        price_now=current_data["current_price"],
                        predicted_price=predicted_price,
                        yesterday_predicted_price=yesterday_predicted_price,
                        news_sent=current_data["news_sentiment"],
                        reddit_sent=current_data["reddit_sentiment"],
                        indicators_score=current_data["technical_indicators"],
                        risk=self.risk,
                        duration=self.duration,
                        cash_balance=self.cash_balance[ticker],
                        current_position=current_position_size,
                    )
            else:
                # Reset the local peak if no position is open
                self.local_peaks[ticker] = None

                # Call the strategy to make a decision
                predictions = predict_price(ticker)
                predicted_price = predictions.get('predicted_price')
                if predicted_price is None:
                    logger.error(f"Prediction failed for {ticker}.")
                    return

                # Predict yesterday's price
                yesterday_date = datetime.now() - timedelta(days=1)
                yesterday_predictions = predict_price(ticker, yesterday_date, mode="backtest")
                yesterday_predicted_price = yesterday_predictions.get('predicted_price')

                decision = make_trade_decision(
                    ticker=ticker,
                    price_now=current_data["current_price"],
                    predicted_price=predicted_price,
                    yesterday_predicted_price=yesterday_predicted_price,
                    news_sent=current_data["news_sentiment"],
                    reddit_sent=current_data["reddit_sentiment"],
                    indicators_score=current_data["technical_indicators"],
                    risk=self.risk,
                    duration=self.duration,
                    cash_balance=self.cash_balance[ticker],
                    current_position=0,
                )

            # Extract action and shares from the decision
            action = decision["action"]
            shares = decision["shares"]

            # Execute the trade
            if action in ["BUY", "SELL"]:
                execute_trade(symbol=ticker, decision=action, quantity=shares)
                time.sleep(30)  # Wait for the order to be processed
                # Update cash balance and positions
                self.cash_balance[ticker] = self.trading_client.get_account().cash

            # Log the decision
            orders = self.trading_client.get_orders(status='all', symbols=[ticker])
            if orders:
                last_order = orders[0]  # Assuming the most recent order is the first in the list
                order_filled = last_order.status == 'filled'
            else:
                order_filled = False
            if not order_filled:
                logger.info(f"[{current_data['datetime']}] Ticker: {ticker}, Action: {action}, Shares: {shares}, "
                      f"Cash Balance (before order filled): {self.cash_balance[ticker]}")
            else:
                logger.info(f"[{current_data['datetime']}] Ticker: {ticker}, Action: {action}, Shares: {shares}, "
                      f"Cash Balance (after order filled): {self.cash_balance[ticker]}")
        except Exception as e:
            logger.error(f"Error making decision for {ticker}: {e}")

    def make_backtest_decision(self, ticker, date):
        """
        Make a trading decision for a given ticker during backtesting.
        """

        # Adjust the start and end dates
        adjusted_start_date = date - timedelta(days=201)
        adjusted_end_date = date

        # Fetch historical data for the adjusted date range
        df = fetch_yahoo_data(ticker, adjusted_start_date, adjusted_end_date)
        if df is None or df.empty:
            logger.error(f"No data available for ticker {ticker} on {date}. Skipping...")
            return

        # Preprocess the data and calculate indicators
        technical_indicators = indicators_score(df)

        # Extract relevant data
        close = df['Close'].iloc[-1]

        # Fetch sentiment scores
        reddit_sentiment = get_reddit_sentiment_score(ticker, date, self.reddit)
        news_sentiment = get_news_sentiment_score(ticker, ticker, date)

        # Predict today's price
        predictions = predict_price(ticker, date, mode="backtest")
        predicted_price = predictions.get('predicted_price')

        # Predict yesterday's price
        yesterday_date = date - timedelta(days=1)
        yesterday_predictions = predict_price(ticker, yesterday_date, mode="backtest")
        yesterday_predicted_price = yesterday_predictions.get('predicted_price')

        if predicted_price is None:
            logger.error(f"Prediction failed for {ticker}.")
            return

        # Call the strategy to make a decision
        decision = make_trade_decision(
            ticker=ticker,
            price_now=close,
            predicted_price=predicted_price,
            news_sent=news_sentiment,
            reddit_sent=reddit_sentiment,
            indicators_score=technical_indicators,
            risk=self.risk,
            duration=self.duration,
            cash_balance=self.cash_balance[ticker],
            current_position=0,  # No current position in backtesting
            yesterday_predicted_price=yesterday_predicted_price,  # Pass yesterday's prediction
        )

        # Log the decision
        logger.info(f"[{date}] Ticker: {ticker}, Action: {decision['action']}, Shares: {decision['shares']}")
        return decision

    def run_paper_trading(self):
        """
        Run the paper trading mode for all selected tickers.
        Each ticker runs in its own thread and makes decisions every hour.
        """
        self.running = True

        def trade_ticker(ticker):
            while self.running:
                self.make_paper_decision(ticker)
                time.sleep(hours_bet_decisions*3600)  # Wait for x hours before the next decision

        # Start a thread for each ticker
        for ticker in self.tickers:
            thread = threading.Thread(target=trade_ticker, args=(ticker,))
            thread.start()
            self.threads.append(thread)

    def run_backtesting(self, start_date, end_date, cash_balance=100000):
        """
        Run the backtesting mode for all selected tickers and generate an interactive Plotly chart.
        """
        # Adjust the start date to fetch extra data for indicators
        adjusted_start = (start_date - timedelta(days=201)).strftime('%Y-%m-%d')

        # Set up Backtrader
        cerebro = bt.Cerebro()

        # Add data feeds for all tickers
        for ticker in self.tickers:
            logger.info(f"Fetching data for {ticker} from {adjusted_start} to {end_date}")
            df = fetch_yahoo_data(ticker, adjusted_start, end_date)
            if df is None or df.empty:
                logger.error(f"No data available for ticker {ticker}. Skipping...")
                continue

            # Add technical indicators
            df = stock_with_indicators(df)

            # Reset the index to make the datetime column accessible
            df.reset_index(inplace=True)

            # Ensure proper column names for Backtrader
            df.rename(columns={
                'Date': 'datetime',
                'Close': 'close',
                'High': 'high',
                'Low': 'low',
                'Open': 'open',
                'Volume': 'volume'
            }, inplace=True)

            # Convert 'datetime' column to datetime64[ns] type
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)

            # Drop rows with missing indicator values
            df.dropna(subset=['macd_hist'], inplace=True)

            # Filter the DataFrame to remove rows before the start_date
            df = df[df.index >= pd.to_datetime(start_date)]

            # Prepare data for Backtrader
            data = CustomPandasData(dataname=df, name=ticker)
            cerebro.adddata(data, name=ticker)  # Add the data feed with the ticker name

        # Add the strategy
        cerebro.addstrategy(MultiTickerBacktestStrategy, agent=self)

        # Set initial cash balance
        cerebro.broker.set_cash(cash_balance)

        # Run backtest
        logger.info(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
        cerebro.run()
        logger.info(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
        final_value = cerebro.broker.getvalue()
        # Extract data for plotting
        plot_data = []
        for data in cerebro.datas:
            df = data._dataname  # Access the Pandas DataFrame used in Backtrader
            df['ticker'] = data._name  # Add the ticker name
            plot_data.append(df)

        # Combine all data into a single DataFrame
        combined_df = pd.concat(plot_data)

        # Create an interactive Plotly chart
        fig = go.Figure()

        for ticker in self.tickers:
            ticker_data = combined_df[combined_df['ticker'] == ticker]
            fig.add_trace(go.Candlestick(
                x=ticker_data.index,
                open=ticker_data['open'],
                high=ticker_data['high'],
                low=ticker_data['low'],
                close=ticker_data['close'],
                name=ticker
            ))

        # Add layout details
        fig.update_layout(
            title="Backtesting Results",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            xaxis_rangeslider_visible=False
        )

        # Return the Plotly figure for Streamlit
        return fig, final_value

    def stop_trading(self):
        """
        Stop all trading threads.
        """
        self.running = False
        for thread in self.threads:
            thread.join()
        logger.info("All trading threads stopped.")

class CustomPandasData(bt.feeds.PandasData):
    """
    Custom PandasData class to include additional indicators.
    """
    lines = ('rsi_14', 'macd', 'macd_hist', 'bb_width', 'vwap', 'ema_50', 'sma_20', 'atr', 'adx')
    params = (
        ('rsi_14', None),
        ('macd', None),
        ('macd_hist', None),
        ('bb_width', None),
        ('vwap', None),
        ('ema_50', None),
        ('sma_20', None),
        ('atr', None),
        ('adx', None),
    )

class MultiTickerBacktestStrategy(bt.Strategy):
    """
    Backtesting strategy for multiple tickers using the TradingAgent's decision-making logic.
    """
    params = (
        ('agent', None),
    )

    def __init__(self):
        self.agent = self.params.agent
        self.data_close = {d._name: d.close for d in self.datas}  # Map ticker names to their close prices
        self.local_peaks = {d._name: None for d in self.datas}  # Track local peak close prices for each ticker

    def next(self):
        for data in self.datas:
            ticker = data._name
            close = data.close[0]
            date = data.datetime.date(0)

            # Check if there is an open position
            current_position_size = self.getposition(data).size
            has_open_position = current_position_size > 0

            if has_open_position:
                # Update the local peak close price
                if self.local_peaks[ticker] is None or close > self.local_peaks[ticker]:
                    self.local_peaks[ticker] = close

                # Compare today's close with the local peak close price
                if close < (1 - (sell_off_peak_drop_percent/100)) * self.local_peaks[ticker]:  # Today's close is 2% less than the local peak
                    logger.info(f"Exiting position for {ticker}: Today's close ({close}) is {sell_off_peak_drop_percent}% less than the local peak ({self.local_peaks[ticker]}).")
                    decision = {"action": "SELL", "shares": current_position_size}
                else:
                    # Call the agent's decision-making logic
                    decision = self.agent.make_backtest_decision(ticker, date)
            else:
                # Reset the local peak if no position is open
                self.local_peaks[ticker] = None

                # Call the agent's decision-making logic
                decision = self.agent.make_backtest_decision(ticker, date)

            # Execute the decision
            if decision["action"] == "BUY":
                ## Only buy if there is no open position
                # if not has_open_position:
                    self.buy(data=data, size=decision["shares"])
                    logger.info(f"BUY executed for {ticker}: {decision['shares']} shares at {close}")
                # else:
                #     logger.info(f"BUY skipped for {ticker}: Position already open.")

            elif decision["action"] == "SELL":
                # Only sell up to the number of shares currently owned
                if has_open_position:
                    sell_size = min(decision["shares"], current_position_size)
                    self.sell(data=data, size=sell_size)
                    logger.info(f"SELL executed for {ticker}: {sell_size} shares at {close}")
                else:
                    logger.info(f"SELL skipped for {ticker}: No shares owned to sell.")

if __name__ == "__main__":

    # Example backtesting usage

    # agent = TradingAgent(tickers=["TCOM"], mode='backtesting', risk=0.4, duration=90, cash_balance=100000)
    # start_date = datetime.strptime("2024-5-07", '%Y-%m-%d').date()
    # end_date = datetime.strptime("2024-06-01", '%Y-%m-%d').date()
    # agent.run_backtesting(start_date=start_date, end_date=end_date)
    
    # # Example live trading usage

    agent = TradingAgent(tickers=["AAPL"], mode='live', risk=0.4, duration=7, cash_balance=100000)
    agent.run_paper_trading()
    agent.stop_trading()