import yfinance as yf
import pandas as pd
import datetime

def fetch_yahoo_data(ticker, start_date, end_date=None):
    """
    Fetches stock data for the given ticker symbol and date range.
    
    Parameters:
    - ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
    - start_date: Start date for fetching data (format: 'YYYY-MM-DD')
    - end_date: End date for fetching data (default: today)
    
    Returns:
    - DataFrame containing stock data with a single row and no index or None if data unavailable
    """
    if end_date is None:
        end_date = datetime.date.today()
        
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        return None

    # Ensure the DataFrame has a single index
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)  # Drop the 2nd level of the multi-index

    # Reset the index to ensure a single index
    data.reset_index(inplace=True)

    return data