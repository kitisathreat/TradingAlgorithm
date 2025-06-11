# Standard library imports
import os
from datetime import datetime, timedelta

# Third-party imports
import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream
import yfinance as yf

# Local imports
import config
from data_fetcher import FMPFetcher, NewsFetcher
from sentiment_analyzer import SentimentAnalyzer
from decision_engine import (
    TradingModel, SentimentData, PriceData, TradeSignal,
    MarketRegime, RiskLevel, TradingDecision
)

class LiveTrader:
    def __init__(self):
        # Initialize APIs and components
        self.api = tradeapi.REST(config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY, config.APCA_BASE_URL)
        self.stream = Stream(config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY, config.APCA_BASE_URL)
        
        self.fmp_fetcher = FMPFetcher()
        self.news_fetcher = NewsFetcher(self.api)
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Initialize the C++ trading model
        print("Initializing C++ trading engine...")
        self.trading_model = TradingModel()
        
        # Configure the trading model
        self.trading_model.set_risk_parameters(
            max_position_size=0.1,  # Maximum 10% of account per position
            max_drawdown=0.05      # Maximum 5% drawdown
        )
        
        self.trading_model.set_technical_parameters(
            sma_period=20,
            ema_period=20,
            rsi_period=14
        )
        
        self.trading_model.set_sentiment_weights(
            social_weight=0.4,
            analyst_weight=0.4,
            news_weight=0.2
        )
        
        # Initialize historical data cache
        self.price_history = {}
        self.sector_data = {}
        self.vix_data = None
        
        # Load initial historical data
        self._load_historical_data()
        
        print("Trading engine initialized successfully.")

    def _load_historical_data(self):
        """Load historical data for all symbols and VIX"""
        print("Loading historical data...")
        
        # Load data for each symbol
        for symbol in config.SYMBOLS_TO_TRADE:
            try:
                # Get 100 days of historical data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=100)
                
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    # Convert to PriceData format
                    price_data = []
                    for idx, row in hist.iterrows():
                        price_data.append(PriceData(
                            price=row['Close'],
                            volume=row['Volume'],
                            timestamp=idx.timestamp()
                        ))
                    self.price_history[symbol] = price_data
                    
                    # Get sector data (using SPY as proxy for now)
                    spy = yf.Ticker('SPY')
                    spy_hist = spy.history(start=start_date, end=end_date)
                    if not spy_hist.empty:
                        self.sector_data[symbol] = spy_hist['Close'].pct_change().tolist()
                
            except Exception as e:
                print(f"Error loading historical data for {symbol}: {e}")
        
        # Load VIX data
        try:
            vix = yf.Ticker('^VIX')
            vix_hist = vix.history(period='1d')
            if not vix_hist.empty:
                self.vix_data = vix_hist['Close'].iloc[-1] / 100.0  # Convert to decimal
        except Exception as e:
            print(f"Error loading VIX data: {e}")
            self.vix_data = 0.15  # Default to 15% volatility if VIX data unavailable

    def _update_price_history(self, symbol: str, price: float, volume: float):
        """Update price history with new trade data"""
        if symbol in self.price_history:
            # Add new price data
            self.price_history[symbol].append(PriceData(
                price=price,
                volume=volume,
                timestamp=datetime.now().timestamp()
            ))
            
            # Keep only last 100 data points
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol] = self.price_history[symbol][-100:]
            
            # Update sector data (using SPY)
            try:
                spy = yf.Ticker('SPY')
                spy_hist = spy.history(period='1d')
                if not spy_hist.empty:
                    self.sector_data[symbol] = spy_hist['Close'].pct_change().tolist()[-100:]
            except Exception as e:
                print(f"Error updating sector data for {symbol}: {e}")

    async def on_trade_update(self, trade):
        """This async function is called on every trade from the Alpaca WebSocket."""
        symbol = trade.symbol
        price = trade.price
        volume = trade.volume
        print(f"\n--- Trigger: New Trade for {symbol} at ${price:.2f} ---")

        # Update price history
        self._update_price_history(symbol, price, volume)

        # 1. Gather sentiment data
        ratings = self.fmp_fetcher.get_analyst_ratings(symbol)
        headline = self.news_fetcher.get_latest_headline(symbol)
        news_sentiment = self.sentiment_analyzer.get_sentiment_score(headline)
        
        # Create sentiment data structure
        sentiment = SentimentData()
        sentiment.social_sentiment = 0.0  # Placeholder for social sentiment
        sentiment.analyst_sentiment = ratings.get('buy_ratio', 0.0)
        sentiment.news_sentiment = news_sentiment
        sentiment.overall_sentiment = (sentiment.analyst_sentiment + sentiment.news_sentiment) / 2.0

        # 2. Get current position size
        try:
            position = self.api.get_position(symbol)
            current_position_size = float(position.qty) * price / float(self.api.get_account().equity)
        except:
            current_position_size = 0.0

        # 3. Get trading decision from C++ engine
        decision = self.trading_model.get_trading_decision(
            symbol=symbol,
            price_data=self.price_history[symbol],
            sentiment=sentiment,
            sector_data=self.sector_data[symbol],
            vix=self.vix_data,
            account_value=float(self.api.get_account().equity),
            current_position_size=current_position_size
        )

        # 4. Log the decision
        print(f"\n[C++ Engine] Decision for {symbol}:")
        print(f"Signal: {decision.signal}")
        print(f"Confidence: {decision.confidence:.2%}")
        print(f"Position Size: {decision.suggested_position_size:.2%}")
        print(f"Stop Loss: ${decision.stop_loss:.2f}")
        print(f"Take Profit: ${decision.take_profit:.2f}")
        print(f"Reasoning: {decision.reasoning}")

        # 5. Execute trades based on decision
        if decision.confidence > 0.7:  # Only trade with high confidence
            try:
                current_position = float(self.api.get_position(symbol).qty) if self.api.get_position(symbol) else 0
                target_position = (float(self.api.get_account().equity) * decision.suggested_position_size) / price
                
                if decision.signal in [TradeSignal.STRONG_BUY, TradeSignal.WEAK_BUY, TradeSignal.INCREASE_POSITION]:
                    if current_position < target_position:
                        qty_to_buy = round(target_position - current_position)
                        if qty_to_buy > 0:
                            self.submit_order(symbol, qty_to_buy, 'buy', decision.stop_loss, decision.take_profit)
                
                elif decision.signal in [TradeSignal.STRONG_SELL, TradeSignal.WEAK_SELL, TradeSignal.REDUCE_POSITION]:
                    if current_position > 0:
                        qty_to_sell = round(current_position - target_position)
                        if qty_to_sell > 0:
                            self.submit_order(symbol, qty_to_sell, 'sell')
                
                elif decision.signal in [TradeSignal.OPTIONS_BUY, TradeSignal.OPTIONS_SELL]:
                    print(f"[INFO] Options trading not implemented yet for {symbol}")
                
            except Exception as e:
                print(f"Error executing trade for {symbol}: {e}")

    def submit_order(self, symbol, qty, side, stop_loss=None, take_profit=None):
        """Submit an order with optional stop loss and take profit"""
        try:
            # Submit the main order
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            print(f"SUCCESS: Submitted {side} order for {qty} shares of {symbol}")
            
            # If stop loss and take profit are provided, submit OCO order
            if stop_loss and take_profit and side == 'buy':
                # Calculate stop loss and take profit prices
                current_price = float(self.api.get_last_trade(symbol).price)
                stop_price = stop_loss
                limit_price = take_profit
                
                # Submit OCO order
                self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='sell',
                    type='oco',
                    time_in_force='gtc',
                    stop_price=stop_price,
                    limit_price=limit_price
                )
                print(f"SUCCESS: Submitted OCO order for {symbol} with stop at ${stop_price:.2f} and limit at ${limit_price:.2f}")
                
        except Exception as e:
            print(f"ERROR: Could not submit order for {symbol}: {e}")

    def run(self):
        """Starts the trading bot."""
        print(f"Starting C++ Trading Engine... Monitoring: {config.SYMBOLS_TO_TRADE}")
        self.stream.subscribe_trades(self.on_trade_update, *config.SYMBOLS_TO_TRADE)
        self.stream.run()

if __name__ == "__main__":
    try:
        bot = LiveTrader()
        bot.run()
    except Exception as e:
        print(f"Error starting trading bot: {e}")
        exit()