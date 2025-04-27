import subprocess
import os
import csv
import json
from logger_config import logger

SUBREDDIT_FILE = "subreddit_lookup.csv"

# Load model name from config.json
with open("config.json", "r") as config_file:
    config = json.load(config_file)
model_name = config.get("model_name", "gemma3:1b")  # Default to "gemma3:1b" if not specified

def init_subreddits_for_ticker(ticker: str, company: str):
    ticker = ticker.upper()

    # Load existing subreddit data
    existing_data = {}
    if os.path.exists(SUBREDDIT_FILE):
        with open(SUBREDDIT_FILE, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and len(row) >= 2:
                    existing_data[row[0]] = [sub.strip() for sub in row[1:]]

    # If we already have it, return it
    if ticker in existing_data:
        return existing_data[ticker]

    # Prompt the LLM using a clean and precise format
    prompt = (
        f"List the most relevant and active Reddit subreddits, only return subbredits you have proof exist, do not make up any names just because you think they seem right (just subreddit names, no extra text) "
        f"where people discuss financial news, analysis, and opinions about the stock with ticker '{ticker}' and company name '{company}'. "
        f"Only return a comma-separated list like: subreddit1, subreddit2, subreddit3"
    )

    try:
        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            capture_output=True, text=True, check=True
        )
        output = result.stdout.strip()

        # Clean output and split into list
        subreddits = [sub.strip().lstrip("r/") for sub in output.split(",") if sub.strip()]

        # Save new data to file
        with open(SUBREDDIT_FILE, mode="a", newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([ticker] + subreddits)

        return subreddits # Return the new subreddits just for confirmation

    except subprocess.CalledProcessError as e:
        logger.error("Error querying Ollama: %s", e.stderr.strip())
        return []

def get_subreddits_for_ticker(ticker: str):

    """
    read the subreddit file and return the subreddits for the given ticker
    """
    if not os.path.exists(SUBREDDIT_FILE):
        logger.error("Subreddit file does not exist.")
        return []
    subreddits = []
    with open(SUBREDDIT_FILE, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row and row[0].upper() == ticker.upper():
                subreddits = [sub.strip() for sub in row[1:]]
                break
    if not subreddits:
        logger.error(f"No subreddits found for ticker {ticker}.")
    else:
        logger.debug(f"Subreddits for ticker {ticker}: {subreddits}")
    return subreddits

# Example usage
if __name__ == "__main__":
    subs = get_subreddits_for_ticker("AAPL")
    print("Suggested subreddits for AAPL:", subs)
    inits = init_subreddits_for_ticker("V", "VISA Inc.")
    print("Initialized subreddits for Visa:", inits)
