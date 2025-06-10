import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream
import config
from data_fetcher import FMPFetcher, NewsFetcher
from sentiment_analyzer import SentimentAnalyzer
import joblib
import numpy as np
import os

class LiveTrader:
    def __init__(self):
        self.api = tradeapi.REST(config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY, config.APCA_BASE_URL)
        self.stream = Stream(config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY, config.APCA_BASE_URL)
        
        self.fmp_fetcher = FMPFetcher()
        self.news_fetcher = NewsFetcher(self.api)
        self.sentiment_analyzer = SentimentAnalyzer()
        
        model_filename = 'investor_model.joblib'
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"Model file '{model_filename}' not found. Run train_model.py first.")
            
        print(f"Loading trained model from '{model_filename}'...")
        self.model = joblib.load(model_filename)
        # We need to know the feature order from training
        self.feature_names = ['Close_Price', 'Analyst_Buy_Ratio', 'News_Sentiment', 'Facial_Sentiment_Code']
        print("Model loaded successfully.")

    async def on_trade_update(self, trade):
        symbol = trade.symbol
        price = trade.price
        print(f"\n--- Trigger: New Trade for {symbol} at ${price} ---")

        ratings = self.fmp_fetcher.get_analyst_ratings(symbol)
        headline = self.news_fetcher.get_latest_headline(symbol)
        sentiment = self.sentiment_analyzer.get_sentiment_score(headline)
        
        # For live trading, we don't have facial sentiment. We can use a neutral value (e.g., 0)
        # or another default. This is an important modeling decision.
        facial_sentiment_code = 0 
        
        live_features = np.array([[
            price,
            ratings.get('buy_ratio', 0),
            sentiment,
            facial_sentiment_code
        ]])
        
        predicted_action = self.model.predict(live_features)[0]
        print(f"[ML Model] Predicted Action: {predicted_action}")
        
        if predicted_action == "BUY":
            print(f"[ACTION] ML Model signals BUY for {symbol}. Submitting order.")
            # self.submit_order(symbol, 10, 'buy')
        elif predicted_action == "SELL":
            print(f"[ACTION] ML Model signals SELL for {symbol}. Submitting order.")
            # self.submit_order(symbol, 10, 'sell')

    def submit_order(self, symbol, qty, side):
        try:
            self.api.submit_order(symbol=symbol, qty=qty, side=side, type='market', time_in_force='day')
            print(f"SUCCESS: Submitted {side} order for {qty} shares of {symbol}.")
        except Exception as e:
            print(f"ERROR: Could not submit order for {symbol}: {e}")

    def run(self):
        print(f"Starting ML-Powered Trading Bot... Monitoring: {config.SYMBOLS_TO_TRADE}")
        self.stream.subscribe_trades(self.on_trade_update, *config.SYMBOLS_TO_TRADE)
        self.stream.run()

if __name__ == "__main__":
    try:
        bot = LiveTrader()
        bot.run()
    except FileNotFoundError as e:
        print(e)
