import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta


def stock_with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators to a stock price DataFrame.
    """
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        else:
            raise ValueError("DataFrame must have a 'Date' column to set as index.")

    close = df['Close'].squeeze()

    # Technical Indicators
    df['rsi_14'] = ta.rsi(close, length=14)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']

    bb = ta.bbands(close, length=20)
    df['bb_width'] = bb['BBU_20_2.0'] - bb['BBL_20_2.0']

    df['vwap'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])

    df['ema_50'] = ta.ema(close, length=50)
    df['sma_20'] = ta.sma(close, length=20)
    df['atr'] = ta.atr(df['High'], df['Low'], close, length=14)

    adx = ta.adx(high=df['High'], low=df['Low'], close=close, length=14)
    df['adx'] = adx['ADX_14']

    df.dropna(inplace=True)
    return df


def interpret_indicators_window(df: pd.DataFrame, window_size: int = 6) -> float:
    """
    Interpret technical indicators in a rolling window and return a sentiment score.
    """
    df = df.dropna(subset=['macd_hist'])
    recent = df.tail(window_size)
    today = df.iloc[-1]
    score = 0.0

    # Rolling metrics
    rsi_avg = recent['rsi_14'].mean()
    macd_trend = recent['macd'].iloc[-1] - recent['macd'].iloc[0]
    vwap_avg = recent['vwap'].mean()
    close_avg = recent['Close'].mean()

    # Scoring logic
    score += 0.3 if rsi_avg < 35 else -0.3 if rsi_avg > 65 else 0
    score += 0.3 if macd_trend > 0 else -0.3
    score += 0.2 if close_avg > vwap_avg else -0.2
    score += 0.2 if today['macd'] > 0 else -0.2
    score += 0.1 if today['macd_hist'] > 0 else -0.1
    score += 0.1 if today['Close'] > today['ema_50'] else 0
    score += 0.1 if today['Close'] > today['sma_20'] else 0

    atr_mean = df['atr'].rolling(14).mean().iloc[-1]
    score += -0.1 if today['atr'] > atr_mean else 0

    if today['adx'] > 25:
        score += 0.1
    elif today['adx'] < 15:
        score -= 0.1

    if 'volume_avg' in today:
        score += 0.05 if today['Volume'] > today['volume_avg'] else -0.05

    return np.tanh(score)


def indicators_score(df: pd.DataFrame, window_size: int = 6) -> float:
    """
    Entry point to compute indicators score from DataFrame.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        else:
            raise ValueError("DataFrame must have a 'Date' column to set as index.")
    
    df = stock_with_indicators(df)

    return interpret_indicators_window(df, window_size=window_size)

if __name__ == "__main__":
    from data_pipeline.fetch_data import fetch_yahoo_data

    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)  # Fetch 30 days of data
    # Fetch data from Yahoo Finance
    df = fetch_yahoo_data("AAPL", start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
    print(df)
    df_with_indicators = stock_with_indicators(df)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print(df_with_indicators.columns)
    print(df_with_indicators.tail(10))
    print(df_with_indicators.head(10))

    score = interpret_indicators_window(df_with_indicators)
    print(f"Date: {df_with_indicators.index[-1]}, Score: {score}")
