from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import datetime
import json
from logger_config import logger

# Download the Vader lexicon only if it hasn't been downloaded yet
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Load Secrets from secrets.json
with open("secrets.json", "r") as secrets_file:
    secrets = json.load(secrets_file)
API_KEY = secrets["news"]["api_key"]

newsapi = NewsApiClient(api_key=API_KEY)  # Get it from newsapi.org

def get_news_sentiment_score(ticker: str, company_name: str, date, days_ago: int = 2):
    """
    Fetches news articles for a given ticker and company name, calculates sentiment scores using Vader,
    and returns an average sentiment score between -1 and 1.
    """
    # convert datetime.date to datetime.datetime
    if isinstance(date, datetime.date) and not isinstance(date, datetime.datetime):
        date = datetime.datetime.combine(date, datetime.datetime.min.time())

    # if date is older than 30 days from today, return 0
    # remove this if you have a paid plan for old news
    if date < (datetime.datetime.now() - datetime.timedelta(days=30)):
        return 0

    try:
        # Define time range
        from_date = date - datetime.timedelta(days=days_ago)
        logger.debug(f"Fetching news sentiment for {ticker} ({company_name}) from {from_date} to {date}")

        # Query news articles
        articles = newsapi.get_everything(
            q=f"{company_name} OR {ticker}",
            from_param=from_date,
            to=date,
            language='en',
            sort_by='publishedAt',
            page_size=30
        )

        sid = SentimentIntensityAnalyzer()

        if not articles['articles']:
            return 0.0  # Neutral if no articles found

        scores = []
        for article in articles['articles']:
            headline = article['title']
            sentiment = sid.polarity_scores(headline)
            scores.append(sentiment['compound'])

        # Return average compound score (-1 = very negative, +1 = very positive)
        return sum(scores) / len(scores)

    except NewsAPIException as e:
        error_message = str(e)
        if "You are trying to request results too far in the past" in error_message:
            logger.error(f"Error fetching news sentiment: {error_message}")
            return 0  # Default sentiment score for this specific error
        else:
            logger.error(f"Error fetching news sentiment: {error_message}")
            raise  # Re-raise unexpected NewsAPI exceptions

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise  # Re-raise unexpected errors

if __name__ == "__main__":
    date ='2025-03-28'
    date = datetime.datetime.strptime(date, '%Y-%m-%d')
    sentiment_score = get_news_sentiment_score(ticker="AAPL", company_name="Apple Inc", date=date)
    print(f"News Sentiment for AAPL: {sentiment_score:.2f}")


# Can add Alpaca Financial News for more news articles and sentiment analysis
#     alpaca_news = AlpacaNewsClient(api_key='your_alpaca_api_key', api_secret='your_alpaca_api_secret')
#     alpaca_articles = alpaca_news.get_news(ticker, from_date, date)
#     for article in alpaca_articles:
#         headline = article['title']
#         sentiment = sid.polarity_scores(headline)
#         scores.append(sentiment['compound'])
#     return sum(scores) / len(scores) if scores else 0.0  # Neutral if no articles found
#     except Exception as e:
#         print(f"Error fetching news sentiment: {e}")
#         return 0.0  # Neutral if an error occurs