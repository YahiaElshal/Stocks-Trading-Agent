# Intelligent Trading Agent 

## Description

The Intelligent Trading Agent is a Streamlit-based dashboard agent designed to trade stocks and make informed trading decisions. It leverages machine learning models, sentiment analysis, and technical indicators to predict stock price movements and execute trades. The agent supports both **paper trading** and **backtesting** modes, allowing users to simulate trading strategies or test them on historical data.

The system integrates with APIs like Alpaca, Reddit, and NewsAPI to fetch real-time data and perform sentiment analysis. It also uses Backtrader for backtesting and TensorFlow for predictive modeling.

---
## How to Run

1. **Setup API Keys**  
   - Open the file `template_secrets.json`.
   - Get your:
     - **Alpaca API** credentials ([Get them here](https://alpaca.markets/)).
     - **Reddit API** credentials ([Get them here](https://www.reddit.com/prefs/apps)).
     - **News API** key ([Get it here](https://newsapi.org/)).
   - Insert your credentials into the JSON file.
   - Rename the file to `secrets.json`.

2. **Install Ollama (for LLM support)(only required if you plan to add new tickers)**  
   - Install Ollama by following the instructions [here](https://ollama.com/).
   - After installing Ollama, pull a model you want to use.  
     Example command to pull the default model:
     ```bash
     ollama run gemma3:1b
     ```
   - **Important:**  
     Make sure the model name you pull matches the `model_name` specified inside `config.json`.  
     You can browse available open-source models [here](https://ollama.com/library).

3. **Run the Dashboard**
   - Open the project folder.
   - Launch the dashboard with:
     ```bash
     streamlit run dashboard.py
     ```

4. **(Optionally) You can replace the Alpaca Paper Account credentials with Live Market account to start trading real money**

---
