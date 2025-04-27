import streamlit as st
from agent import TradingAgent
import os
from datetime import datetime
import json
# File to store tickers
TICKERS_FILE = "tickers.txt"

# Load existing config
with open("config.json", "r") as f:
    config = json.load(f)

# Function to load tickers from file
def load_tickers():
    if os.path.exists(TICKERS_FILE):
        with open(TICKERS_FILE, "r") as file:
            return [line.strip() for line in file.readlines()]
    return ["AAPL", "V", "EQIX"]  # Default tickers if file doesn't exist

# Function to save tickers to file
def save_tickers(tickers):
    with open(TICKERS_FILE, "w") as file:
        file.writelines(f"{ticker}\n" for ticker in tickers)

# Load tickers
tickers = load_tickers()

# Initialize the agent (singleton pattern to persist state)
if "agent" not in st.session_state:
    st.session_state.agent = None

st.title("Intelligent Trading Agent")

# User inputs
st.sidebar.header("Trading Settings")
trading_mode = st.sidebar.radio("Select Trading Mode", ["Paper Trading", "Backtesting"])

# Text inputs for adding a new ticker and company name
new_ticker = st.sidebar.text_input("Add New Ticker", value="Ticker Symbol")
company_name = st.sidebar.text_input("Add Company Name", value="Company Name")

# Add new ticker functionality
if new_ticker != "Ticker Symbol" and company_name != "Company Name" and new_ticker.strip() and company_name.strip():
    if st.sidebar.button("Add Ticker"):
        with st.spinner("Adding new ticker..."):
            from utils import init_new_ticker  # Import the utility function
            init_new_ticker(new_ticker.strip(), company_name.strip())  # Initialize the new ticker with company name
        tickers.append(new_ticker.strip())  # Add the new ticker to the list
        save_tickers(tickers)  # Save the updated tickers to the file
        st.sidebar.success(f"Ticker '{new_ticker.strip()}' for '{company_name.strip()}' added successfully!")
        
        # Update session state to reflect the new ticker
        st.session_state.tickers = tickers

# Ensure tickers are loaded into session state
if "tickers" not in st.session_state:
    st.session_state.tickers = tickers

# Ticker selection with checkboxes (updated dynamically)
st.sidebar.subheader("Select Tickers to Trade")
selected_tickers = {ticker: st.sidebar.checkbox(ticker, value=False) for ticker in st.session_state.tickers}

# Filter only the checked tickers
checked_tickers = [ticker for ticker, checked in selected_tickers.items() if checked]

risk = st.sidebar.slider("Risk Level (0.1 = Low, 1.0 = High)", min_value=0.1, max_value=1.0, step=0.1, value=0.3)
duration = st.sidebar.selectbox("Intended Trading Duration", ["7 Days", "90 Days"])
cash_balance = st.sidebar.number_input("Initial Cash Balance ($)", min_value=1000, step=100, value=100000)

# Advanced Settings
with st.sidebar.expander("⚙️ Advanced Settings (Optional)", expanded=False):
    st.markdown("Tweak model sensitivity, buying rules, and trading engine behavior.")

    st.subheader("Decision Weights")
    st.text("Aim for all weights to sum to 1.0")
    weight_prediction = st.slider(
        "Prediction Model Weight",
        0.0,
        1.0,
        config.get("base_weights", {}).get("prediction_weight", 0.15),
        0.01,
        help="How much the agent trusts the raw prediction model output."
    )
    weight_trend = st.slider(
        "Prediction Trend Weight",
        0.0,
        1.0,
        config.get("base_weights", {}).get("prediction_trend_weight", 0.15),
        0.01,
        help="How much the agent trusts the predicted trend direction."
    )
    weight_news = st.slider(
        "News Sentiment Weight",
        0.0,
        1.0,
        config.get("base_weights", {}).get("news_weight", 0.2),
        0.01,
        help="How heavily news sentiment analysis affects decisions."
    )
    weight_reddit = st.slider(
        "Reddit Sentiment Weight",
        0.0,
        1.0,
        config.get("base_weights", {}).get("reddit_weight", 0.2),
        0.01,
        help="How much Reddit discussions influence the agent."
    )
    weight_technical = st.slider(
        "Technical Indicators Weight",
        0.0,
        1.0,
        config.get("base_weights", {}).get("technical_indicator_weight", 0.3),
        0.01,
        help="Trust given to traditional technical indicators like RSI, MACD, etc."
    )

    st.subheader("Trading Logic Tweaks")
    prediction_aggressiveness = st.slider(
        "Prediction Aggressiveness Factor",
        0.5,
        5.0,
        config.get("prediction_aggressiveness_factor", 2.5),
        0.1,
        help="Multiplies prediction confidence. Higher = more aggressive trades."
    )
    buy_threshold = st.slider(
        "Minimum Buy Threshold",
        0.0,
        0.5,
        config.get("buy_signal_min_threshold", 0.05),
        0.01,
        help="Minimum confidence score needed before considering a buy."
    )
    position_size_limit = st.slider(
        "Max % of Cash Per Trade",
        0.05,
        1.0,
        config.get("max_percent_per_position", 0.25),
        0.05,
        help="Caps how much cash can go into a single position."
    )
    max_risk_multiplier = st.slider(
        "Max Risk Multiplier",
        0.5,
        5.0,
        config.get("confidence_max_risk_multiplier", 2.0),
        0.1,
        help="Scales maximum position size based on risk/confidence."
    )

    st.subheader("Selling Behavior")
    sell_trigger_drop = st.slider(
        "Sell After Drop (%)",
        1.0,
        10.0,
        config.get("sell_off_peak_drop_percent", 3.0),
        0.1,
        help="Triggers a sell if price drops this % from the last local maximum (Predicted Peak)."
    )

    st.subheader("Order Placement Settings")
    order_buffer_multiplier = st.slider(
        "Order Fill Buffer",
        1.0,
        1.1,
        config.get("limit_order_price_buffer", 1.007),
        0.001,
        format="%.3f",
        help="Adds a small price buffer to help orders fill faster in paper trading (use 1.0 for live trading)."
    )
    price_rounding = st.number_input(
        "Buy Price Rounding (Decimals)",
        min_value=0,
        max_value=5,
        value=config.get("buy_price_rounding", 2),
        help="Number of decimals to round limit order prices."
    )
    hours_bet_decisions = st.number_input(
        "Hours Between Each Decision",
        min_value=1,
        max_value=48,
        value=config.get("hours_bet_decisions",12),
        help="(FOR LIVE TRADING) hours between each decision"
    )
# Before initializing the agent, update and save config
def save_config():
    new_config = {
        "base_weights": {
            "prediction_weight": weight_prediction,
            "prediction_trend_weight": weight_trend,
            "news_weight": weight_news,
            "reddit_weight": weight_reddit,
            "technical_indicator_weight": weight_technical
        },
        "prediction_aggressiveness_factor": prediction_aggressiveness,
        "buy_signal_min_threshold": buy_threshold,
        "max_percent_per_position": position_size_limit,
        "confidence_max_risk_multiplier": max_risk_multiplier,
        "sell_off_peak_drop_percent": sell_trigger_drop,
        "limit_order_price_buffer": order_buffer_multiplier,
        "buy_price_rounding": price_rounding,
        "hours_bet_decisions": hours_bet_decisions
    }
    with open("config.json", "w") as f:
        json.dump(new_config, f, indent=2)

# Run button
if st.sidebar.button("Initialise Agent"):
    if not checked_tickers:
        st.warning("Please select at least one ticker to trade.")
    else:
        # Convert duration to appropriate format
        duration_days = 7 if duration == "7 Days" else 90

        # Save the configuration settings to the config.json file
        save_config()

        # Initialize the agent with only the checked tickers
        st.session_state.agent = TradingAgent(
            tickers=checked_tickers,
            risk=risk,
            duration=duration_days,
            mode=trading_mode,
        )

        st.success(f"Agent initialized for {checked_tickers} with ${cash_balance} cash balance, {risk} risk, and {duration} duration.")

# Display agent status and decision
if st.session_state.agent:
    st.header("Agent Status")
    st.write(f"Ticker: {st.session_state.agent.tickers}")
    st.write(f"Risk Level: {st.session_state.agent.risk}")
    st.write(f"Duration: {st.session_state.agent.duration} days")
    if trading_mode == "Paper Trading":
        st.subheader("Paper Trading Controls")
        if st.button("Start Paper Trading"):
            if st.session_state.agent:
                st.success("Paper trading started!")
                st.info("You can view your dashboard at https://app.alpaca.markets/paper/dashboard/overview ")
                st.info("You can also find the decisions made by the agent in the logs.")
                st.session_state.agent.run_paper_trading()
                st.success("Paper trading finished!")
        if st.button("Stop Paper Trading"):
            if st.session_state.agent:
                st.session_state.agent.stop_trading()
                st.success("Paper trading stopped!")
                st.info("You can view your dashboard at https://app.alpaca.markets/paper/dashboard/overview ")
                st.info("You can also find the decisions made by the agent in the logs.")
                
    elif trading_mode == "Backtesting":
        st.subheader("Backtesting Controls")
        # Input fields for start and end dates
        start_date = st.date_input(
            "Start Date for Backtesting", 
            min_value=datetime(2023, 1, 1).date(),
            max_value=datetime.today().date(), 
            help="Start date must be after January 1, 2023."
        )
        end_date = st.date_input(
            "End Date for Backtesting",
            min_value=datetime(2023, 1, 2).date(),
            max_value=datetime.today().date(),
            help="End date must be before today."
        )

        # convert dates to strings
        start_date = datetime.strptime(str(start_date), '%Y-%m-%d').date()
        end_date = datetime.strptime(str(end_date), '%Y-%m-%d').date()

        # Ensure both dates are provided and valid
        if start_date and end_date and start_date < end_date:
            if st.button("Start Backtesting"):
                st.success(f"Backtesting starting from {start_date} to {end_date}!")

                if st.session_state.agent:
                    fig, portfolio_value = st.session_state.agent.run_backtesting(start_date=start_date, end_date=end_date)

                    # Display the interactive Plotly chart
                    st.plotly_chart(fig, use_container_width=True)
                    st.text("Backtesting finished!")
                    st.text(f"Final Portfolio Value: ${portfolio_value:.2f}")
        else:
            st.warning("Please provide valid start and end dates for backtesting.")

        if st.button("Stop Backtesting") and "chart_plotted" not in st.session_state:
            if st.session_state.agent:
                st.session_state.agent.stop_trading()
                st.success("Backtesting stopped!")
                st.session_state.chart_plotted = True