import pandas as pd
import json
from logger_config import logger

# Load configuration from config.json
with open("config.json", "r") as config_file:
    config = json.load(config_file)
prediction_weight = float(config.get("base_weights", {}).get("prediction_weight", 0.15))  # Default to 0.15 if not specified
predictoin_trend_weight = float(config.get("base_weights", {}).get("prediction_trend_weight", 0.15))  # Default to 0.15 if not specified
news_weight = float(config.get("base_weights", {}).get("news_weight", 0.2))  # Default to 0.2 if not specified
reddit_weight = float(config.get("base_weights", {}).get("reddit_weight", 0.2))  # Default to 0.2 if not specified
technical_weight = float(config.get("base_weights", {}).get("technical_weight", 0.3))  # Default to 0.3 if not specified

prediction_aggressiveness_factor = float(config.get("prediction_aggressiveness_factor", 2.5))  # Default to 2.5 if not specified
buy_signal_min_threshold = float(config.get("buy_signal_min_threshold", 0.05))  # Default to 0.05 if not specified
max_percent_per_position = float(config.get("max_percent_per_position", 0.25))  # Default to 0.25 if not specified
confidence_max_risk_multiplier = float(config.get("confidence_max_risk_multiplier", 3.0))  # Default to 3.0 if not specified

def make_trade_decision(ticker, price_now, predicted_price, news_sent, reddit_sent,
                        indicators_score, risk, duration, cash_balance, current_position=0, yesterday_predicted_price=None):
    """
    Make a trade decision based on various inputs.
    """
    # Ensure current_position is numeric
    if not isinstance(current_position, (int, float)):
        raise ValueError(f"Expected current_position to be numeric, got {type(current_position)}")

    # Calculate decision score
    score = decision_score(price_now, predicted_price, news_sent, reddit_sent, indicators_score, risk, duration, yesterday_predicted_price)

    # Determine position change
    position_change = decide_position_size(score, cash_balance, price_now, risk, current_position)

    if position_change > 0:
        action = "BUY"
    elif position_change < 0:
        action = "HOLD"
    else:
        action = "HOLD"

    return {
        "ticker": ticker,
        "action": action,
        "score": score,
        "shares": abs(position_change),
        "confidence": abs(score),
        "price": price_now,
    }

def clean_predicted_price(predicted, current):
    """
    Safeguard against unrealistic price predictions.
    """
    if current < 5 and predicted > 4 * current:
        return None
    elif current >= 5 and predicted > 2 * current:
        return None
    return predicted

def decision_score(price_now, predicted_price, news_sent, reddit_sent, indicators_score, risk, duration, yesterday_predicted_price=None):
    """
    Calculate the decision score based on various factors, including prediction change.
    """
    # Set default weights for each factor
    base_weights = {
        "prediction": prediction_weight,
        "prediction_change": predictoin_trend_weight,
        "news": news_weight,
        "reddit": reddit_weight,
        "technical": technical_weight,
    }

    available_weights = base_weights.copy()

    # Clean predicted price
    predicted_price = clean_predicted_price(predicted_price, price_now)
    if not predicted_price:
        available_weights["prediction"] = 0
        pred_score = 0
    else:
        pred_score = (predicted_price - price_now) / price_now
        pred_score = max(min(pred_score * prediction_aggressiveness_factor, 1), -1)  # Make more aggressive

    # Calculate prediction change score
    if yesterday_predicted_price:
        prediction_change = (predicted_price - yesterday_predicted_price) / yesterday_predicted_price
        # scale to -1 to 1
        prediction_change_score = max(min(prediction_change, 1), -1)
    else:
        available_weights["prediction_change"] = 0
        prediction_change_score = 0

    # Sometimes the sentiment from reddit or news specifically will be 0 bearing no weight,
    # so we need to make sure that the decision score does not use reddit sentiment then,
    # especially for backtesting case of old dates and free plan api
    # Adjust weights if sentiment is zero
    if news_sent == 0:
        available_weights["news"] = 0
    if reddit_sent == 0:
        available_weights["reddit"] = 0

    # Normalize weights
    total_weight = sum(available_weights.values())
    if total_weight == 0:
        return 0

    for k in available_weights:
        available_weights[k] /= total_weight

    # Calculate technical score
    tech_score = indicators_score

    # Calculate decision score
    decision = (
        pred_score * available_weights["prediction"] +
        news_sent * available_weights["news"] +
        reddit_sent * available_weights["reddit"] +
        tech_score * available_weights["technical"] +
        prediction_change_score * available_weights["prediction_change"]
    )

    # Duration multiplier
    duration_multiplier = {
        7: 1.4,
        30: 1.0,
        90: 0.7
    }.get(duration, 1.0)

    decision *= risk * duration_multiplier

    # Debugging: Log the full equation and intermediate values
    logger.debug("\n--- Decision Score Debugging ---")
    logger.debug(f"Price Now: {price_now}")
    logger.debug(f"Predicted Price: {predicted_price}")
    logger.debug(f"Yesterday's Predicted Price: {yesterday_predicted_price}")
    logger.debug(f"Prediction Change Score: {prediction_change_score} (Weight: {available_weights['prediction_change']})")
    logger.debug(f"Predicted Score (pred_score): {pred_score} (Weight: {available_weights['prediction']})")
    logger.debug(f"News Sentiment (news_sent): {news_sent} (Weight: {available_weights['news']})")
    logger.debug(f"Reddit Sentiment (reddit_sent): {reddit_sent} (Weight: {available_weights['reddit']})")
    logger.debug(f"Technical Score (tech_score): {tech_score} (Weight: {available_weights['technical']})")
    logger.debug(f"Risk Level: {risk}")
    logger.debug(f"Duration Multiplier: {duration_multiplier}")
    logger.debug(f"Final Decision Score Equation:")
    logger.debug(f"Decision = ({pred_score} * {available_weights['prediction']}) + "
                 f"({news_sent} * {available_weights['news']}) + "
                 f"({reddit_sent} * {available_weights['reddit']}) + "
                 f"({tech_score} * {available_weights['technical']}) + "
                 f"({prediction_change_score} * {available_weights['prediction_change']})")
    logger.debug(f"Decision *= {risk} * {duration_multiplier}")
    logger.debug(f"Final Decision Score: {decision}")
    logger.debug("--- End Debugging ---\n")

    return decision

def decide_position_size(decision_score, cash_balance, price_now, risk_level, current_position=0):
    """
    Returns number of shares to buy/sell based on confidence and risk.
    """
    min_threshold = buy_signal_min_threshold 
    if abs(decision_score) < min_threshold:
        return 0

    confidence = min(abs(decision_score), 1)

    # Scale value from 0.2 to 3 based on confidence and risk level
    min_risk_scale = 0.2
    risk_multiplier = min_risk_scale + (confidence_max_risk_multiplier - min_risk_scale) * (confidence * risk_level)

    # Calculate maximum investment based on scaled exposure
    general_max_invest = max_percent_per_position  # % of cash
    max_invest = (cash_balance * general_max_invest) * risk_multiplier
    num_shares = int(max_invest // price_now)

    # Exposure and Volatility Adjustment
    if num_shares != 0:
        MAX_ALLOWED_POSITION_PER_STOCK = (general_max_invest * cash_balance) / price_now  

        # Exposure scaling
        exposure_scale = max(0.2, 1 - (current_position / MAX_ALLOWED_POSITION_PER_STOCK))

        # Volatility scaling
        HIGH_VOLATILITY = False  # todo: Can be fetched online from a source like Yahoo Finance
        volatility_scale = 0.7 if HIGH_VOLATILITY else 1.0

        final_scale = exposure_scale * volatility_scale

        num_shares = int(num_shares * final_scale)

    return num_shares if decision_score > 0 else -num_shares