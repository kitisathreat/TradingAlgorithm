"""
Bridge module to connect the neural network model with the C++ trading engine
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add the orchestrator directory to Python path
REPO_ROOT = Path(__file__).parent.parent
ORCHESTRATOR_PATH = REPO_ROOT / "_2_Orchestrator_And_ML_Python"
sys.path.append(str(ORCHESTRATOR_PATH))

# Import the model trainer
from interactive_training_app.backend.model_trainer import ModelTrainer, TradingNeuralNetwork

# Import C++ bindings
try:
    from decision_engine import (
        TradingModel, NeuralNetworkInsights, SentimentData,
        MarketRegime, TradeSignal, RiskLevel
    )
    CPP_BINDINGS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"C++ bindings not available: {e}")
    CPP_BINDINGS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelBridge:
    """
    Bridge class to connect the neural network model with the C++ trading engine
    """
    
    def __init__(self):
        self.model_trainer = ModelTrainer()
        self.trading_model = TradingModel() if CPP_BINDINGS_AVAILABLE else None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize the bridge and load the trained model"""
        try:
            # Check if model is trained
            model_state = self.model_trainer.get_model_state()
            if not model_state['is_trained']:
                logger.error("Neural network model is not trained")
                return False
            
            # Initialize C++ trading model if available
            if not CPP_BINDINGS_AVAILABLE:
                logger.error("C++ bindings not available")
                return False
            
            # Set up trading model parameters
            self.trading_model.set_risk_parameters(max_position_size=1.0, max_drawdown=0.1)
            self.trading_model.set_technical_parameters(sma_period=20, ema_period=20, rsi_period=14)
            self.trading_model.set_sentiment_weights(social_weight=0.3, analyst_weight=0.4, news_weight=0.3)
            self.trading_model.set_neural_network_weight(0.4)  # Give significant weight to NN predictions
            
            self.is_initialized = True
            logger.info("Model bridge initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing model bridge: {e}")
            return False
    
    def _map_to_market_regime(self, prediction: str) -> MarketRegime:
        """Map neural network prediction to market regime"""
        if prediction == "BUY":
            return MarketRegime.TRENDING_UP
        elif prediction == "SELL":
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.RANGING
    
    def _create_neural_network_insights(
        self,
        prediction: str,
        confidence: float,
        features: Dict
    ) -> NeuralNetworkInsights:
        """Create NeuralNetworkInsights from model prediction"""
        # Calculate trend strength based on technical indicators
        trend_strength = 0.0
        if 'price_change_20d' in features:
            trend_strength = features['price_change_20d']
        
        return NeuralNetworkInsights(
            predicted_price_change=trend_strength,
            trend_strength=abs(trend_strength),
            predicted_regime=self._map_to_market_regime(prediction),
            confidence=confidence
        )
    
    def get_trading_decision(
        self,
        symbol: str,
        features: Dict,
        sentiment_data: Optional[Dict] = None,
        sector_data: Optional[list] = None,
        vix: float = 20.0,
        account_value: float = 100000.0,
        current_position: float = 0.0
    ) -> Dict:
        """
        Get trading decision by combining neural network predictions with C++ engine
        """
        if not self.is_initialized:
            return {'error': 'Model bridge not initialized'}
        
        try:
            # Get neural network prediction
            nn_prediction = self.model_trainer.make_prediction(features)
            if 'error' in nn_prediction:
                return nn_prediction
            
            # Create neural network insights
            nn_insights = self._create_neural_network_insights(
                prediction=nn_prediction['prediction'],
                confidence=nn_prediction['confidence'],
                features=features
            )
            
            # Create sentiment data
            sentiment = SentimentData(
                social_sentiment=sentiment_data.get('social_sentiment', 0.0) if sentiment_data else 0.0,
                analyst_sentiment=sentiment_data.get('analyst_sentiment', 0.0) if sentiment_data else 0.0,
                news_sentiment=sentiment_data.get('news_sentiment', 0.0) if sentiment_data else 0.0,
                overall_sentiment=sentiment_data.get('overall_sentiment', 0.0) if sentiment_data else 0.0
            )
            
            # Convert price data to format expected by C++ engine
            price_data = [{
                'price': features['current_price'],
                'volume': features['volume'],
                'timestamp': features.get('timestamp', 0.0)
            }]
            
            # Get trading decision from C++ engine
            decision = self.trading_model.get_trading_decision(
                symbol=symbol,
                price_data=price_data,
                sentiment=sentiment,
                sector_data=sector_data or [],
                vix=vix,
                account_value=account_value,
                current_position_size=current_position,
                nn_insights=nn_insights
            )
            
            # Convert decision to dictionary
            return {
                'signal': decision.signal.name,
                'confidence': float(decision.confidence),
                'position_size': float(decision.suggested_position_size),
                'stop_loss': float(decision.stop_loss),
                'take_profit': float(decision.take_profit),
                'reasoning': decision.reasoning,
                'neural_network_prediction': nn_prediction['prediction'],
                'neural_network_confidence': float(nn_prediction['confidence'])
            }
            
        except Exception as e:
            logger.error(f"Error getting trading decision: {e}")
            return {'error': str(e)}
    
    def get_model_status(self) -> Dict:
        """Get status of both neural network and C++ engine"""
        return {
            'neural_network': self.model_trainer.get_model_state(),
            'cpp_engine_available': CPP_BINDINGS_AVAILABLE,
            'bridge_initialized': self.is_initialized,
            'training_stats': self.model_trainer.get_training_stats()
        } 