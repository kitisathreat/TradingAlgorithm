# Standard library imports
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

# Third-party imports
import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

# Local imports
import config
try:
    from data_fetcher import FMPFetcher, NewsFetcher
except ImportError:
    import logging
    logging.warning("Could not import FMPFetcher or NewsFetcher. Running in basic mode.")
    class FMPFetcher:
        def get_analyst_ratings(self, symbol):
            logging.warning("FMPFetcher is unavailable. Returning default ratings.")
            return {"buy_ratio": 0.5}
    class NewsFetcher:
        def __init__(self, api=None):
            pass
        def get_latest_headline(self, symbol):
            logging.warning("NewsFetcher is unavailable. Returning default headline.")
            return "No news available."
from sentiment_analyzer import SentimentAnalyzer

# Try to import decision engine, with fallback for development
try:
    from decision_engine import (
        TradingModel, SentimentData, PriceData, TradeSignal,
        MarketRegime, RiskLevel, TradingDecision, TradingEngineError
    )
    DECISION_ENGINE_AVAILABLE = True
except ImportError:
    logging.warning("Decision engine module not found. Using fallback implementation.")
    DECISION_ENGINE_AVAILABLE = False
    
    # Fallback implementations for development
    class TradingEngineError(Exception):
        pass
    
    class MarketRegime:
        TRENDING_UP = "TRENDING_UP"
        TRENDING_DOWN = "TRENDING_DOWN"
        RANGING = "RANGING"
        VOLATILE = "VOLATILE"
        UNKNOWN = "UNKNOWN"
    
    class TradeSignal:
        STRONG_BUY = "STRONG_BUY"
        WEAK_BUY = "WEAK_BUY"
        HOLD = "HOLD"
        REDUCE_POSITION = "REDUCE_POSITION"
        INCREASE_POSITION = "INCREASE_POSITION"
        WEAK_SELL = "WEAK_SELL"
        STRONG_SELL = "STRONG_SELL"
        OPTIONS_BUY = "OPTIONS_BUY"
        OPTIONS_SELL = "OPTIONS_SELL"
    
    class RiskLevel:
        VERY_LOW = "VERY_LOW"
        LOW = "LOW"
        MEDIUM = "MEDIUM"
        HIGH = "HIGH"
        VERY_HIGH = "VERY_HIGH"
    
    class PriceData:
        def __init__(self, price: float, volume: float, timestamp: float):
            self.price = price
            self.volume = volume
            self.timestamp = timestamp
    
    class SentimentData:
        def __init__(self):
            self.social_sentiment = 0.0
            self.analyst_sentiment = 0.0
            self.news_sentiment = 0.0
            self.overall_sentiment = 0.0
    
    class TradingDecision:
        def __init__(self):
            self.signal = TradeSignal.HOLD
            self.confidence = 0.0
            self.suggested_position_size = 0.0
            self.stop_loss = 0.0
            self.take_profit = 0.0
            self.reasoning = ""
    
    class TradingModel:
        def __init__(self):
            logging.warning("Using fallback TradingModel implementation")
        
        def get_trading_decision(self, *args, **kwargs):
            decision = TradingDecision()
            decision.reasoning = "Using fallback implementation"
            return decision
        
        def set_risk_parameters(self, *args, **kwargs):
            pass
        
        def set_technical_parameters(self, *args, **kwargs):
            pass
        
        def set_sentiment_weights(self, *args, **kwargs):
            pass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradingState:
    """Represents the current state of trading for a symbol"""
    symbol: str
    current_position: float
    last_decision: Optional[TradingDecision]
    last_update: datetime
    consecutive_errors: int = 0

class LiveTrader:
    def __init__(self):
        # Initialize APIs and components
        try:
            self.api = tradeapi.REST(config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY, config.APCA_BASE_URL)
            self.stream = Stream(config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY, config.APCA_BASE_URL)
            
            self.fmp_fetcher = FMPFetcher()
            self.news_fetcher = NewsFetcher(self.api)
            self.sentiment_analyzer = SentimentAnalyzer()
            
            # Initialize the C++ trading model with error handling
            logger.info("Initializing C++ trading engine...")
            try:
                self.trading_model = TradingModel()
                
                # Configure the trading model with validation
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
                
                logger.info("Trading engine initialized successfully")
            except TradingEngineError as e:
                logger.error(f"Error initializing trading model: {e}")
                raise
            
            # Initialize trading state tracking
            self.trading_states: Dict[str, TradingState] = {}
            for symbol in config.SYMBOLS_TO_TRADE:
                self.trading_states[symbol] = TradingState(
                    symbol=symbol,
                    current_position=0.0,
                    last_decision=None,
                    last_update=datetime.now()
                )
            
            # Initialize historical data cache
            self.price_history = {}
            self.sector_data = {}
            self.vix_data = None
            
            # Load initial historical data
            self._load_historical_data()
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise

    @contextmanager
    def _error_handling_context(self, symbol: str, operation: str):
        """Context manager for handling errors during trading operations"""
        try:
            yield
            # Reset error count on success
            self.trading_states[symbol].consecutive_errors = 0
        except TradingEngineError as e:
            self.trading_states[symbol].consecutive_errors += 1
            logger.error(f"Trading engine error during {operation} for {symbol}: {e}")
            if self.trading_states[symbol].consecutive_errors >= 3:
                logger.critical(f"Too many consecutive errors for {symbol}, pausing trading")
                # Implement trading pause logic here
        except Exception as e:
            logger.error(f"Unexpected error during {operation} for {symbol}: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _load_historical_data(self):
        """Load historical data for all symbols and VIX with retry logic"""
        logger.info("Loading historical data...")
        
        for symbol in config.SYMBOLS_TO_TRADE:
            try:
                # Get 100 days of historical data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=100)
                
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    # Convert to PriceData format with validation
                    price_data = []
                    for idx, row in hist.iterrows():
                        try:
                            price_data.append(PriceData(
                                price=float(row['Close']),
                                volume=float(row['Volume']),
                                timestamp=float(idx.timestamp())
                            ))
                        except (ValueError, TypeError) as e:
                            logger.error(f"Error converting price data for {symbol}: {e}")
                            continue
                    
                    self.price_history[symbol] = price_data
                    
                    # Get sector data (using SPY as proxy for now)
                    spy = yf.Ticker('SPY')
                    spy_hist = spy.history(start=start_date, end=end_date)
                    if not spy_hist.empty:
                        self.sector_data[symbol] = spy_hist['Close'].pct_change().fillna(0).tolist()
                
            except Exception as e:
                logger.error(f"Error loading historical data for {symbol}: {e}")
                raise
        
        # Load VIX data
        try:
            vix = yf.Ticker('^VIX')
            vix_hist = vix.history(period='1d')
            if not vix_hist.empty:
                self.vix_data = float(vix_hist['Close'].iloc[-1]) / 100.0  # Convert to decimal
            else:
                logger.warning("No VIX data available, using default value")
                self.vix_data = 0.15  # Default to 15% volatility
        except Exception as e:
            logger.error(f"Error loading VIX data: {e}")
            self.vix_data = 0.15  # Default to 15% volatility

    def _update_price_history(self, symbol: str, price: float, volume: float):
        """Update price history with new trade data"""
        try:
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            # Create new price data point with validation
            new_price_data = PriceData(
                price=float(price),
                volume=float(volume),
                timestamp=float(datetime.now().timestamp())
            )
            
            # Update history (keep last 1000 points)
            self.price_history[symbol].append(new_price_data)
            if len(self.price_history[symbol]) > 1000:
                self.price_history[symbol] = self.price_history[symbol][-1000:]
            
            # Update trading state
            self.trading_states[symbol].last_update = datetime.now()
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error updating price history for {symbol}: {e}")
            raise

    async def on_trade_update(self, trade):
        """Handle new trade updates with enhanced error handling"""
        symbol = trade.symbol
        price = trade.price
        volume = trade.volume
        logger.info(f"\n--- New Trade for {symbol} at ${price:.2f} ---")

        with self._error_handling_context(symbol, "trade update"):
            # Update price history
            self._update_price_history(symbol, price, volume)

            # 1. Gather sentiment data with error handling
            try:
                ratings = self.fmp_fetcher.get_analyst_ratings(symbol)
                headline = self.news_fetcher.get_latest_headline(symbol)
                news_sentiment = self.sentiment_analyzer.get_sentiment_score(headline)
                
                sentiment = SentimentData()
                sentiment.social_sentiment = 0.0  # Placeholder for social sentiment
                sentiment.analyst_sentiment = float(ratings.get('buy_ratio', 0.0))
                sentiment.news_sentiment = float(news_sentiment)
                sentiment.overall_sentiment = (sentiment.analyst_sentiment + sentiment.news_sentiment) / 2.0
            except Exception as e:
                logger.error(f"Error gathering sentiment data for {symbol}: {e}")
                # Use neutral sentiment as fallback
                sentiment = SentimentData()

            # 2. Get current position size
            try:
                position = self.api.get_position(symbol)
                current_position_size = float(position.qty) * price / float(self.api.get_account().equity)
            except Exception as e:
                logger.warning(f"Error getting position for {symbol}: {e}")
                current_position_size = 0.0

            # 3. Get trading decision from C++ engine
            try:
                decision = self.trading_model.get_trading_decision(
                    symbol=symbol,
                    price_data=self.price_history[symbol],
                    sentiment=sentiment,
                    sector_data=self.sector_data[symbol],
                    vix=self.vix_data,
                    account_value=float(self.api.get_account().equity),
                    current_position_size=current_position_size
                )
                
                # Update trading state
                self.trading_states[symbol].last_decision = decision
                self.trading_states[symbol].current_position = current_position_size

                # Log the decision
                logger.info(f"\nDecision for {symbol}:")
                logger.info(f"Signal: {decision.signal}")
                logger.info(f"Confidence: {decision.confidence:.2%}")
                logger.info(f"Position Size: {decision.suggested_position_size:.2%}")
                logger.info(f"Stop Loss: ${decision.stop_loss:.2f}")
                logger.info(f"Take Profit: ${decision.take_profit:.2f}")
                logger.info(f"Reasoning: {decision.reasoning}")

                # 4. Execute trades based on decision
                if decision.confidence > 0.7:  # Only trade with high confidence
                    await self._execute_trade_decision(symbol, decision, price)

            except TradingEngineError as e:
                logger.error(f"Trading engine error for {symbol}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in trading decision for {symbol}: {e}")
                raise

    async def _execute_trade_decision(self, symbol: str, decision: TradingDecision, current_price: float):
        """Execute trading decisions with enhanced error handling"""
        try:
            current_position = float(self.api.get_position(symbol).qty) if self.api.get_position(symbol) else 0
            target_position = (float(self.api.get_account().equity) * decision.suggested_position_size) / current_price
            
            if decision.signal in [TradeSignal.STRONG_BUY, TradeSignal.WEAK_BUY, TradeSignal.INCREASE_POSITION]:
                if current_position < target_position:
                    qty_to_buy = round(target_position - current_position)
                    if qty_to_buy > 0:
                        await self.submit_order(symbol, qty_to_buy, 'buy', decision.stop_loss, decision.take_profit)
            
            elif decision.signal in [TradeSignal.STRONG_SELL, TradeSignal.WEAK_SELL, TradeSignal.REDUCE_POSITION]:
                if current_position > 0:
                    qty_to_sell = round(current_position - target_position)
                    if qty_to_sell > 0:
                        await self.submit_order(symbol, qty_to_sell, 'sell')
            
            elif decision.signal in [TradeSignal.OPTIONS_BUY, TradeSignal.OPTIONS_SELL]:
                logger.info(f"Options trading not implemented yet for {symbol}")
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def submit_order(self, symbol: str, qty: int, side: str, stop_loss: Optional[float] = None, 
                          take_profit: Optional[float] = None):
        """Submit an order with retry logic and enhanced error handling"""
        try:
            # Validate inputs
            if qty <= 0:
                raise ValueError("Order quantity must be positive")
            if side not in ['buy', 'sell']:
                raise ValueError("Order side must be 'buy' or 'sell'")
            
            # Submit the main order
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            logger.info(f"SUCCESS: Submitted {side} order for {qty} shares of {symbol}")
            
            # If stop loss and take profit are provided, submit OCO order
            if stop_loss and take_profit and side == 'buy':
                # Validate stop loss and take profit
                if stop_loss >= current_price:
                    raise ValueError("Stop loss must be below current price")
                if take_profit <= current_price:
                    raise ValueError("Take profit must be above current price")
                
                # Submit OCO order
                self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='sell',
                    type='oco',
                    time_in_force='gtc',
                    stop_price=stop_loss,
                    limit_price=take_profit
                )
                logger.info(f"SUCCESS: Submitted OCO order for {symbol} with stop at ${stop_loss:.2f} and limit at ${take_profit:.2f}")
                
        except Exception as e:
            logger.error(f"ERROR: Could not submit order for {symbol}: {e}")
            raise

    def run(self):
        """Start the trading bot with error handling"""
        try:
            logger.info(f"Starting C++ Trading Engine... Monitoring: {config.SYMBOLS_TO_TRADE}")
            self.stream.subscribe_trades(self.on_trade_update, *config.SYMBOLS_TO_TRADE)
            self.stream.run()
        except Exception as e:
            logger.critical(f"Fatal error in trading bot: {e}")
            raise

if __name__ == "__main__":
    try:
        trader = LiveTrader()
        trader.run()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        raise