import pytest
import sys
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict

# Add the project root to the Python path
sys.path.insert(0, '..')

# Import the trading components
from decision_engine import (
    TradingModel, SentimentData, PriceData, TradeSignal,
    MarketRegime, RiskLevel, TradingDecision, TradingEngineError
)
from live_trader import LiveTrader, TradingState

class TestTradingIntegration:
    @pytest.fixture
    def trading_model(self):
        """Fixture providing a configured TradingModel instance"""
        model = TradingModel()
        model.set_risk_parameters(max_position_size=0.1, max_drawdown=0.05)
        model.set_technical_parameters(sma_period=20, ema_period=20, rsi_period=14)
        model.set_sentiment_weights(social_weight=0.4, analyst_weight=0.4, news_weight=0.2)
        return model

    @pytest.fixture
    def sample_price_data(self) -> List[PriceData]:
        """Fixture providing sample price data for testing"""
        # Generate 100 days of sample price data
        base_price = 100.0
        volatility = 0.02
        dates = [datetime.now() - timedelta(days=x) for x in range(100)]
        
        prices = []
        current_price = base_price
        
        for date in reversed(dates):
            # Generate random price movement
            change = np.random.normal(0, volatility)
            current_price *= (1 + change)
            
            # Generate random volume
            volume = np.random.uniform(1000, 10000)
            
            prices.append(PriceData(
                price=current_price,
                volume=volume,
                timestamp=date.timestamp()
            ))
        
        return prices

    @pytest.fixture
    def sample_sentiment(self) -> SentimentData:
        """Fixture providing sample sentiment data"""
        return SentimentData(
            social_sentiment=0.6,
            analyst_sentiment=0.7,
            news_sentiment=0.5,
            overall_sentiment=0.6
        )

    def test_trading_model_initialization(self, trading_model):
        """Test proper initialization of the trading model"""
        assert trading_model is not None
        # Test that the model can be used
        assert isinstance(trading_model, TradingModel)

    def test_invalid_risk_parameters(self, trading_model):
        """Test validation of risk parameters"""
        with pytest.raises(TradingEngineError):
            trading_model.set_risk_parameters(max_position_size=1.5, max_drawdown=0.05)
        
        with pytest.raises(TradingEngineError):
            trading_model.set_risk_parameters(max_position_size=0.1, max_drawdown=-0.1)

    def test_invalid_technical_parameters(self, trading_model):
        """Test validation of technical parameters"""
        with pytest.raises(TradingEngineError):
            trading_model.set_technical_parameters(sma_period=0, ema_period=20, rsi_period=14)
        
        with pytest.raises(TradingEngineError):
            trading_model.set_technical_parameters(sma_period=20, ema_period=-5, rsi_period=14)

    def test_invalid_sentiment_weights(self, trading_model):
        """Test validation of sentiment weights"""
        with pytest.raises(TradingEngineError):
            trading_model.set_sentiment_weights(social_weight=-0.1, analyst_weight=0.6, news_weight=0.5)
        
        with pytest.raises(TradingEngineError):
            trading_model.set_sentiment_weights(social_weight=0.4, analyst_weight=0.4, news_weight=0.4)  # Sum > 1

    def test_trading_decision_validation(self, trading_model, sample_price_data, sample_sentiment):
        """Test validation of trading decision inputs"""
        # Test empty symbol
        with pytest.raises(TradingEngineError):
            trading_model.get_trading_decision(
                symbol="",
                price_data=sample_price_data,
                sentiment=sample_sentiment,
                sector_data=[0.1, 0.2, 0.3],
                vix=0.15,
                account_value=100000.0,
                current_position_size=0.0
            )
        
        # Test empty price data
        with pytest.raises(TradingEngineError):
            trading_model.get_trading_decision(
                symbol="AAPL",
                price_data=[],
                sentiment=sample_sentiment,
                sector_data=[0.1, 0.2, 0.3],
                vix=0.15,
                account_value=100000.0,
                current_position_size=0.0
            )
        
        # Test invalid account value
        with pytest.raises(TradingEngineError):
            trading_model.get_trading_decision(
                symbol="AAPL",
                price_data=sample_price_data,
                sentiment=sample_sentiment,
                sector_data=[0.1, 0.2, 0.3],
                vix=0.15,
                account_value=-1000.0,
                current_position_size=0.0
            )

    def test_trading_decision_output(self, trading_model, sample_price_data, sample_sentiment):
        """Test the output of trading decisions"""
        decision = trading_model.get_trading_decision(
            symbol="AAPL",
            price_data=sample_price_data,
            sentiment=sample_sentiment,
            sector_data=[0.1, 0.2, 0.3],
            vix=0.15,
            account_value=100000.0,
            current_position_size=0.0
        )
        
        # Verify decision structure
        assert isinstance(decision, TradingDecision)
        assert isinstance(decision.signal, TradeSignal)
        assert 0.0 <= decision.confidence <= 1.0
        assert 0.0 <= decision.suggested_position_size <= 1.0
        assert decision.stop_loss < decision.take_profit
        assert isinstance(decision.reasoning, str)
        assert len(decision.reasoning) > 0

    def test_price_data_conversion(self, trading_model):
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
        decision = trading_model.get_trading_decision(
            symbol="AAPL",
            price_data=valid_data,
            sentiment=SentimentData(),
            sector_data=[0.1, 0.2],
            vix=0.15,
            account_value=100000.0,
            current_position_size=0.0
        )
        assert isinstance(decision, TradingDecision)
        
        # Test invalid data types
        with pytest.raises(TradingEngineError):
            trading_model.get_trading_decision(
                symbol="AAPL",
                price_data=invalid_data,
                sentiment=SentimentData(),
                sector_data=[0.1, 0.2],
                vix=0.15,
                account_value=100000.0,
                current_position_size=0.0
            )
        
        # Test missing fields
        with pytest.raises(TradingEngineError):
            trading_model.get_trading_decision(
                symbol="AAPL",
                price_data=incomplete_data,
                sentiment=SentimentData(),
                sector_data=[0.1, 0.2],
                vix=0.15,
                account_value=100000.0,
                current_position_size=0.0
            )

    def test_trading_state_management(self):
        """Test the trading state management functionality"""
        state = TradingState(
            symbol="AAPL",
            current_position=0.0,
            last_decision=None,
            last_update=datetime.now()
        )
        
        assert state.symbol == "AAPL"
        assert state.current_position == 0.0
        assert state.last_decision is None
        assert isinstance(state.last_update, datetime)
        assert state.consecutive_errors == 0

    def test_market_regime_detection(self, trading_model, sample_price_data):
        """Test market regime detection in different scenarios"""
        # Test with trending up data
        up_trend_data = sample_price_data.copy()
        for i in range(len(up_trend_data)):
            up_trend_data[i].price = 100.0 * (1 + 0.001 * i)
        
        decision_up = trading_model.get_trading_decision(
            symbol="AAPL",
            price_data=up_trend_data,
            sentiment=SentimentData(),
            sector_data=[0.1, 0.2, 0.3],
            vix=0.15,
            account_value=100000.0,
            current_position_size=0.0
        )
        
        # Test with trending down data
        down_trend_data = sample_price_data.copy()
        for i in range(len(down_trend_data)):
            down_trend_data[i].price = 100.0 * (1 - 0.001 * i)
        
        decision_down = trading_model.get_trading_decision(
            symbol="AAPL",
            price_data=down_trend_data,
            sentiment=SentimentData(),
            sector_data=[0.1, 0.2, 0.3],
            vix=0.15,
            account_value=100000.0,
            current_position_size=0.0
        )
        
        # Verify that decisions are different for different market regimes
        assert decision_up.signal != decision_down.signal or decision_up.confidence != decision_down.confidence

    def test_risk_management(self, trading_model, sample_price_data, sample_sentiment):
        """Test risk management functionality"""
        # Test with high volatility
        high_vol_data = sample_price_data.copy()
        for i in range(len(high_vol_data)):
            high_vol_data[i].price = 100.0 * (1 + np.random.normal(0, 0.05))
        
        decision_high_vol = trading_model.get_trading_decision(
            symbol="AAPL",
            price_data=high_vol_data,
            sentiment=sample_sentiment,
            sector_data=[0.1, 0.2, 0.3],
            vix=0.30,  # High VIX
            account_value=100000.0,
            current_position_size=0.0
        )
        
        # Test with low volatility
        low_vol_data = sample_price_data.copy()
        for i in range(len(low_vol_data)):
            low_vol_data[i].price = 100.0 * (1 + np.random.normal(0, 0.01))
        
        decision_low_vol = trading_model.get_trading_decision(
            symbol="AAPL",
            price_data=low_vol_data,
            sentiment=sample_sentiment,
            sector_data=[0.1, 0.2, 0.3],
            vix=0.10,  # Low VIX
            account_value=100000.0,
            current_position_size=0.0
        )
        
        # Verify that position sizes are adjusted for risk
        assert decision_high_vol.suggested_position_size <= decision_low_vol.suggested_position_size

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 