import unittest
import sys
import os
import logging
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import decision engine, with fallback for testing
try:
    from decision_engine import (
        TradingModel, SentimentData, PriceData, TradeSignal,
        MarketRegime, RiskLevel, TradingDecision, TradingEngineError
    )
    DECISION_ENGINE_AVAILABLE = True
except ImportError:
    logging.warning("Decision engine module not found. Using fallback implementation for tests.")
    DECISION_ENGINE_AVAILABLE = False
    
    # Import fallback implementations from live_trader
    from live_trader import (
        TradingModel, SentimentData, PriceData, TradeSignal,
        MarketRegime, RiskLevel, TradingDecision, TradingEngineError
    )

from live_trader import LiveTrader, TradingState

class TestTradingIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DECISION_ENGINE_AVAILABLE:
            logging.warning("Running tests with fallback decision engine implementation")
    
    def setUp(self):
        self.trader = LiveTrader()
        # Add test setup code here
    
    def test_trading_decision(self):
        # Create test data
        price_data = [
            PriceData(price=100.0, volume=1000.0, timestamp=datetime.now().timestamp()),
            PriceData(price=101.0, volume=1100.0, timestamp=(datetime.now() + timedelta(minutes=1)).timestamp())
        ]
        
        sentiment = SentimentData()
        sentiment.analyst_sentiment = 0.7
        sentiment.news_sentiment = 0.6
        sentiment.overall_sentiment = 0.65
        
        # Test trading decision
        decision = self.trader.trading_model.get_trading_decision(
            symbol="AAPL",
            price_data=price_data,
            sentiment=sentiment,
            sector_data=[0.01, 0.02, 0.03],
            vix=0.15,
            account_value=100000.0,
            current_position_size=0.0
        )
        
        self.assertIsInstance(decision, TradingDecision)
        self.assertIsInstance(decision.signal, (str, TradeSignal))
        self.assertGreaterEqual(decision.confidence, 0.0)
        self.assertLessEqual(decision.confidence, 1.0)
        
        if not DECISION_ENGINE_AVAILABLE:
            self.assertEqual(decision.reasoning, "Using fallback implementation")
    
    def test_sentiment_analysis(self):
        # Test sentiment analysis integration
        sentiment = self.trader.sentiment_analyzer.get_sentiment_score("AAPL reports strong earnings")
        self.assertIsInstance(sentiment, float)
        self.assertGreaterEqual(sentiment, -1.0)
        self.assertLessEqual(sentiment, 1.0)
    
    def test_market_data_fetching(self):
        # Test market data fetching
        try:
            self.trader._load_historical_data()
            self.assertGreater(len(self.trader.price_history), 0)
        except Exception as e:
            if not DECISION_ENGINE_AVAILABLE:
                self.skipTest("Skipping market data test with fallback implementation")
            else:
                raise

    @unittest.skipIf(not DECISION_ENGINE_AVAILABLE, "Decision engine not available")
    def test_trading_model_initialization(self):
        """Test proper initialization of the trading model"""
        assert self.trader.trading_model is not None
        # Test that the model can be used
        assert isinstance(self.trader.trading_model, TradingModel)

    @unittest.skipIf(not DECISION_ENGINE_AVAILABLE, "Decision engine not available")
    def test_invalid_risk_parameters(self):
        """Test validation of risk parameters"""
        with self.assertRaises(TradingEngineError):
            self.trader.trading_model.set_risk_parameters(max_position_size=1.5, max_drawdown=0.05)
        
        with self.assertRaises(TradingEngineError):
            self.trader.trading_model.set_risk_parameters(max_position_size=0.1, max_drawdown=-0.1)

    @unittest.skipIf(not DECISION_ENGINE_AVAILABLE, "Decision engine not available")
    def test_invalid_technical_parameters(self):
        """Test validation of technical parameters"""
        with self.assertRaises(TradingEngineError):
            self.trader.trading_model.set_technical_parameters(sma_period=0, ema_period=20, rsi_period=14)
        
        with self.assertRaises(TradingEngineError):
            self.trader.trading_model.set_technical_parameters(sma_period=20, ema_period=-5, rsi_period=14)

    @unittest.skipIf(not DECISION_ENGINE_AVAILABLE, "Decision engine not available")
    def test_invalid_sentiment_weights(self):
        """Test validation of sentiment weights"""
        with self.assertRaises(TradingEngineError):
            self.trader.trading_model.set_sentiment_weights(social_weight=-0.1, analyst_weight=0.6, news_weight=0.5)
        
        with self.assertRaises(TradingEngineError):
            self.trader.trading_model.set_sentiment_weights(social_weight=0.4, analyst_weight=0.4, news_weight=0.4)  # Sum > 1

    @unittest.skipIf(not DECISION_ENGINE_AVAILABLE, "Decision engine not available")
    def test_trading_decision_validation(self):
        """Test validation of trading decision inputs"""
        # Test empty symbol
        with self.assertRaises(TradingEngineError):
            self.trader.trading_model.get_trading_decision(
                symbol="",
                price_data=[],
                sentiment=SentimentData(),
                sector_data=[0.1, 0.2, 0.3],
                vix=0.15,
                account_value=100000.0,
                current_position_size=0.0
            )
        
        # Test empty price data
        with self.assertRaises(TradingEngineError):
            self.trader.trading_model.get_trading_decision(
                symbol="AAPL",
                price_data=[],
                sentiment=SentimentData(),
                sector_data=[0.1, 0.2, 0.3],
                vix=0.15,
                account_value=100000.0,
                current_position_size=0.0
            )
        
        # Test invalid account value
        with self.assertRaises(TradingEngineError):
            self.trader.trading_model.get_trading_decision(
                symbol="AAPL",
                price_data=[],
                sentiment=SentimentData(),
                sector_data=[0.1, 0.2, 0.3],
                vix=0.15,
                account_value=-1000.0,
                current_position_size=0.0
            )

    @unittest.skipIf(not DECISION_ENGINE_AVAILABLE, "Decision engine not available")
    def test_trading_decision_output(self):
        """Test the output of trading decisions"""
        decision = self.trader.trading_model.get_trading_decision(
            symbol="AAPL",
            price_data=[],
            sentiment=SentimentData(),
            sector_data=[0.1, 0.2, 0.3],
            vix=0.15,
            account_value=100000.0,
            current_position_size=0.0
        )
        
        # Verify decision structure
        self.assertIsInstance(decision, TradingDecision)
        self.assertIsInstance(decision.signal, TradeSignal)
        self.assertGreaterEqual(decision.confidence, 0.0)
        self.assertLessEqual(decision.confidence, 1.0)
        self.assertLess(decision.stop_loss, decision.take_profit)
        self.assertIsInstance(decision.reasoning, str)
        self.assertGreater(len(decision.reasoning), 0)

    @unittest.skipIf(not DECISION_ENGINE_AVAILABLE, "Decision engine not available")
    def test_price_data_conversion(self):
        """Test conversion of Python data to C++ PriceData"""
        # Test valid data
        valid_data = [
            {"price": 100.0, "volume": 1000.0, "timestamp": datetime.now().timestamp()},
            {"price": 101.0, "volume": 2000.0, "timestamp": datetime.now().timestamp()}
        ]
        
        # Test invalid data types
        invalid_data = [
            {"price": "100.0", "volume": 1000.0, "timestamp": datetime.now().timestamp()},
            {"price": 101.0, "volume": "2000.0", "timestamp": datetime.now().timestamp()}
        ]
        
        # Test missing fields
        incomplete_data = [
            {"price": 100.0, "volume": 1000.0},
            {"price": 101.0, "timestamp": datetime.now().timestamp()}
        ]
        
        # Test valid data
        decision = self.trader.trading_model.get_trading_decision(
            symbol="AAPL",
            price_data=valid_data,
            sentiment=SentimentData(),
            sector_data=[0.1, 0.2],
            vix=0.15,
            account_value=100000.0,
            current_position_size=0.0
        )
        self.assertIsInstance(decision, TradingDecision)
        
        # Test invalid data types
        with self.assertRaises(TradingEngineError):
            self.trader.trading_model.get_trading_decision(
                symbol="AAPL",
                price_data=invalid_data,
                sentiment=SentimentData(),
                sector_data=[0.1, 0.2],
                vix=0.15,
                account_value=100000.0,
                current_position_size=0.0
            )
        
        # Test missing fields
        with self.assertRaises(TradingEngineError):
            self.trader.trading_model.get_trading_decision(
                symbol="AAPL",
                price_data=incomplete_data,
                sentiment=SentimentData(),
                sector_data=[0.1, 0.2],
                vix=0.15,
                account_value=100000.0,
                current_position_size=0.0
            )

    @unittest.skipIf(not DECISION_ENGINE_AVAILABLE, "Decision engine not available")
    def test_trading_state_management(self):
        """Test the trading state management functionality"""
        state = TradingState(
            symbol="AAPL",
            current_position=0.0,
            last_decision=None,
            last_update=datetime.now()
        )
        
        self.assertEqual(state.symbol, "AAPL")
        self.assertEqual(state.current_position, 0.0)
        self.assertIsNone(state.last_decision)
        self.assertIsInstance(state.last_update, datetime)
        self.assertEqual(state.consecutive_errors, 0)

    @unittest.skipIf(not DECISION_ENGINE_AVAILABLE, "Decision engine not available")
    def test_market_regime_detection(self):
        """Test market regime detection in different scenarios"""
        # Test with trending up data
        up_trend_data = [
            PriceData(
                price=100.0 * (1 + 0.001 * i),
                volume=np.random.uniform(1000, 10000),
                timestamp=(datetime.now() - timedelta(days=i)).timestamp()
            ) for i in range(100)
        ]
        
        decision_up = self.trader.trading_model.get_trading_decision(
            symbol="AAPL",
            price_data=up_trend_data,
            sentiment=SentimentData(),
            sector_data=[0.1, 0.2, 0.3],
            vix=0.15,
            account_value=100000.0,
            current_position_size=0.0
        )
        
        # Test with trending down data
        down_trend_data = [
            PriceData(
                price=100.0 * (1 - 0.001 * i),
                volume=np.random.uniform(1000, 10000),
                timestamp=(datetime.now() - timedelta(days=i)).timestamp()
            ) for i in range(100)
        ]
        
        decision_down = self.trader.trading_model.get_trading_decision(
            symbol="AAPL",
            price_data=down_trend_data,
            sentiment=SentimentData(),
            sector_data=[0.1, 0.2, 0.3],
            vix=0.15,
            account_value=100000.0,
            current_position_size=0.0
        )
        
        # Verify that decisions are different for different market regimes
        self.assertNotEqual(decision_up.signal, decision_down.signal)
        self.assertNotEqual(decision_up.confidence, decision_down.confidence)

    @unittest.skipIf(not DECISION_ENGINE_AVAILABLE, "Decision engine not available")
    def test_risk_management(self):
        """Test risk management functionality"""
        # Test with high volatility
        high_vol_data = [
            PriceData(
                price=100.0 * (1 + np.random.normal(0, 0.05)),
                volume=np.random.uniform(1000, 10000),
                timestamp=(datetime.now() - timedelta(days=i)).timestamp()
            ) for i in range(100)
        ]
        
        decision_high_vol = self.trader.trading_model.get_trading_decision(
            symbol="AAPL",
            price_data=high_vol_data,
            sentiment=SentimentData(),
            sector_data=[0.1, 0.2, 0.3],
            vix=0.30,  # High VIX
            account_value=100000.0,
            current_position_size=0.0
        )
        
        # Test with low volatility
        low_vol_data = [
            PriceData(
                price=100.0 * (1 + np.random.normal(0, 0.01)),
                volume=np.random.uniform(1000, 10000),
                timestamp=(datetime.now() - timedelta(days=i)).timestamp()
            ) for i in range(100)
        ]
        
        decision_low_vol = self.trader.trading_model.get_trading_decision(
            symbol="AAPL",
            price_data=low_vol_data,
            sentiment=SentimentData(),
            sector_data=[0.1, 0.2, 0.3],
            vix=0.10,  # Low VIX
            account_value=100000.0,
            current_position_size=0.0
        )
        
        # Verify that position sizes are adjusted for risk
        self.assertLessEqual(decision_high_vol.suggested_position_size, decision_low_vol.suggested_position_size)

if __name__ == '__main__':
    unittest.main() 