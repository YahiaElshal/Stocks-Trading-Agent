import unittest
from models.predict import predict_price

class TestPredictPrice(unittest.TestCase):
    def test_predict_price_live(self):
        result = predict_price("AAPL", mode="live")
        self.assertIn("predicted_price", result)

    def test_predict_price_backtest(self):
        result = predict_price("AAPL", date="2025-04-01", mode="backtest")
        self.assertIn("predicted_price", result)

if __name__ == "__main__":
    unittest.main()