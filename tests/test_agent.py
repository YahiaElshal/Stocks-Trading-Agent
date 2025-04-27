import unittest
from datetime import datetime
from agent import TradingAgent

class TestTradingAgent(unittest.TestCase):
    def setUp(self):
        self.agent = TradingAgent(tickers=["AAPL"], mode="backtest", risk=0.3, duration=7, cash_balance=100000)

    def test_make_backtest_decision(self):
        decision = self.agent.make_backtest_decision("AAPL", datetime(2025, 4, 1))
        self.assertIn("action", decision)
        self.assertIn("shares", decision)

    def test_run_backtesting(self):
        self.agent.run_backtesting(start_date=datetime(2025, 1, 1), end_date=datetime(2025, 4, 1))
        # No assertion, just ensure no exceptions are raised

if __name__ == "__main__":
    unittest.main()