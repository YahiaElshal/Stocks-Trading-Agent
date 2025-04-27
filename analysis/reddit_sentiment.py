import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from analysis.utils import get_subreddits_for_ticker # Remove analysis/ if running this as a standalone script
import time
import prawcore
import datetime
from logger_config import logger

def get_reddit_sentiment_score(ticker, date, reddit):
    """
    Fetches Reddit posts for a given ticker and date, calculates sentiment scores using Vader,
    and returns an average sentiment score between -1 and 1.
    """
    # convert datetime.date to datetime.datetime
    if isinstance(date, datetime.date) and not isinstance(date, datetime.datetime):
        date = datetime.datetime.combine(date, datetime.datetime.min.time())

    # if date is older than 30 days from today, return 0
    # remove this if you have a paid plan for old reddit posts
    if date < (datetime.datetime.now() - datetime.timedelta(days=30)):
        return 0
    
    # Get subreddits for the ticker
    subreddits = get_subreddits_for_ticker(ticker)
    if not subreddits:
        raise ValueError(f"No subreddits found for ticker {ticker}")

    # Initialize Vader sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Fetch Reddit posts and calculate sentiment
    sentiment_scores = []
    for subreddit in subreddits:
        try:
            subreddit_instance = reddit.subreddit(subreddit)
            subreddit_instance.id  # This will raise an exception if the subreddit is inaccessible or does not exist
        except prawcore.exceptions.Redirect:
            logger.warning(f"Subreddit '{subreddit}' does not exist. Skipping...")
            continue
        except prawcore.exceptions.Forbidden:
            logger.warning(f"Access to subreddit '{subreddit}' is forbidden. Skipping...")
            continue
        for post in subreddit_instance.search(f"{ticker}", time_filter='month'):
            post_date = pd.to_datetime(post.created_utc, unit='s')
            if post_date <= pd.to_datetime(date):
                sentiment = analyzer.polarity_scores(post.title + " " + post.selftext)
                sentiment_scores.append(sentiment['compound'])

    if sentiment_scores:
        return sum(sentiment_scores) / len(sentiment_scores)
    else:
        return 0  # Neutral sentiment if no posts are found to avoid messing up the decision score


if __name__ == "__main__":

    import json
    
    # Load Secrets from secrets.json
    with open("secrets.json", "r") as secrets_file:
        secrets = json.load(secrets_file)
    REDDIT_CLIENT_ID = secrets["reddit"]["client_id"]
    REDDIT_CLIENT_SECRET = secrets["reddit"]["client_secret"]
    REDDIT_USER_AGENT = secrets["reddit"]["user_agent"]

    # Initialize PRAW 
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_SECRET,
                         client_secret=REDDIT_CLIENT_SECRET,
                         user_agent=REDDIT_USER_AGENT)

    # Example usage
    date ='2025-03-23'
    date = datetime.datetime.strptime(date, '%Y-%m-%d')
    score = get_reddit_sentiment_score('MSFT', date, reddit)
    print(f"Sentiment Score: {score}")