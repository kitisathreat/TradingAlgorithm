import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream
import config
from data_fetcher import FMPFetcher, NewsFetcher
from sentiment_analyzer import SentimentAnalyzer
import joblib
import numpy as np
import pandas as pd
import os
from tensorflow import keras

class LiveTrader:
    def __init__(self):
        # Initialize APIs and components
        self.api = tradeapi.REST(config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY, config.APCA_BASE_URL)
        self.stream = Stream(config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY, config.APCA_BASE_URL)
        
        self.fmp_fetcher = FMPFetcher()
        self.news_fetcher = NewsFetcher(self.api)
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # --- Load the complete set of trained assets ---
        model_path = 'investor_nn_model.keras'
        preprocessor_path = 'data_preprocessor.joblib'
        encoder_path = 'target_encoder.joblib'

        if not all(os.path.exists(p) for p in [model_path, preprocessor_path, encoder_path]):
            raise FileNotFoundError("One or more required model assets (.keras, .joblib) not found. Run train_model.py first.")
            
        print("Loading neural network model and data preprocessors...")
        self.model = keras.models.load_model(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.target_encoder = joblib.load(encoder_path)
        print("Neural network assets loaded successfully.")

    async def on_trade_update(self, trade):
        """This async function is called on every trade from the Alpaca WebSocket."""
        symbol = trade.symbol
        price = trade.price
        print(f"\n--- Trigger: New Trade for {symbol} at ${price:.2f} ---")

        # 1. Gather live features
        ratings = self.fmp_fetcher.get_analyst_ratings(symbol)
        headline = self.news_fetcher.get_latest_headline(symbol)
        sentiment = self.sentiment_analyzer.get_sentiment_score(headline)
        
        # For live trading, we don't have a live facial reading. We must provide a
        # neutral or default value for this feature. We use 0.
        facial_sentiment_code = 0 
        
        # 2. Assemble features into a DataFrame with the EXACT column names used in training
        live_data_df = pd.DataFrame([{
            'Close_Price': price,
            'Analyst_Buy_Ratio': ratings.get('buy_ratio', 0.0),
            'News_Sentiment': sentiment,
            'Facial_Sentiment_Code': facial_sentiment_code
        }])

        # 3. Preprocess the live data using the loaded preprocessor
        live_features_processed = self.preprocessor.transform(live_data_df)

        # 4. Use the loaded model to predict the action probabilities
        prediction_probabilities = self.model.predict(live_features_processed)[0]
        
        # 5. Decode the prediction to get the final action string
        predicted_index = np.argmax(prediction_probabilities)
        predicted_action = self.target_encoder.categories_[0][predicted_index]
        confidence = prediction_probabilities[predicted_index]
        
        print(f"[NN Model] Predicted Action: {predicted_action} (Confidence: {confidence:.2%})")
        
        # 6. Act on the model's prediction
        if predicted_action == "BUY" and confidence > 0.7: # Optional confidence threshold
            print(f"[ACTION] ML Model signals BUY for {symbol}. Submitting order.")
            # self.submit_order(symbol, 10, 'buy')
        elif predicted_action == "SELL" and confidence > 0.7:
            print(f"[ACTION] ML Model signals SELL for {symbol}. Submitting order.")
            # self.submit_order(symbol, 10, 'sell')

    def submit_order(self, symbol, qty, side):
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            print(f"SUCCESS: Submitted {side} order for {qty} shares of {symbol}.")
        except Exception as e:
            print(f"ERROR: Could not submit order for {symbol}: {e}")

    def run(self):
        """Starts the trading bot."""
        print(f"Starting ML-Powered Trading Bot... Monitoring: {config.SYMBOLS_TO_TRADE}")
        self.stream.subscribe_trades(self.on_trade_update, *config.SYMBOLS_TO_TRADE)
        self.stream.run()

if __name__ == "__main__":
    try:
        bot = LiveTrader()
        bot.run()
    except FileNotFoundError as e:
        print(e)
        exit()