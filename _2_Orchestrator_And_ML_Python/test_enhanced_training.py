#!/usr/bin/env python3
"""
Test script for enhanced training features
Tests multiple neural network models and additive training data
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add the orchestrator path
ORCHESTRATOR_PATH = Path(__file__).parent
sys.path.append(str(ORCHESTRATOR_PATH))

from interactive_training_app.backend.model_trainer import ModelTrainer

def test_enhanced_training():
    """Test the enhanced training features"""
    print("🧪 Testing Enhanced Training Features")
    print("=" * 50)
    
    try:
        # Test 1: Initialize with different model types
        print("\n1️⃣ Testing Model Initialization:")
        model_types = ["simple", "standard", "deep", "lstm", "ensemble"]
        
        for model_type in model_types:
            trainer = ModelTrainer(model_type=model_type)
            print(f"   ✅ {model_type.capitalize()} model initialized")
            print(f"      - Model type: {trainer.model_type}")
            print(f"      - Training examples: {len(trainer.training_examples)}")
        
        # Test 2: Model information
        print("\n2️⃣ Testing Model Information:")
        trainer = ModelTrainer(model_type="standard")
        available_models = trainer.get_available_models()
        
        for model_type, info in available_models.items():
            print(f"   📊 {info['name']}:")
            print(f"      - Description: {info['description']}")
            print(f"      - Best for: {info['best_for']}")
            print(f"      - Training time: {info['training_time']}")
            print(f"      - Complexity: {info['complexity']}")
        
        # Test 3: Model changing
        print("\n3️⃣ Testing Model Changing:")
        trainer = ModelTrainer(model_type="simple")
        print(f"   Initial model: {trainer.model_type}")
        
        success = trainer.change_model_type("deep")
        print(f"   Changed to deep: {'✅ Success' if success else '❌ Failed'}")
        print(f"   Current model: {trainer.model_type}")
        
        # Test 4: Additive training data
        print("\n4️⃣ Testing Additive Training Data:")
        
        # Create sample training examples
        sample_examples = [
            {
                'timestamp': datetime.now().isoformat(),
                'technical_features': {
                    'symbol': 'AAPL',
                    'current_price': 150.0,
                    'volume': 1000000,
                    'rsi': 65.0,
                    'macd': 0.5,
                    'bb_position': 0.6,
                    'volatility': 0.02,
                    'volume_ratio': 1.2,
                    'price_change_1d': 0.01,
                    'price_change_5d': 0.05,
                    'price_change_20d': 0.15,
                    'return_5d': 5.0,
                    'return_20d': 15.0,
                    'return_60d': 25.0,
                    'volatility_20d': 20.0,
                    'max_drawdown': -10.0
                },
                'sentiment_analysis': {
                    'sentiment_score': 0.3,
                    'positive': 0.6,
                    'negative': 0.3,
                    'keywords': ['bullish', 'momentum', 'support'],
                    'confidence': 'medium'
                },
                'user_decision': 'BUY'
            },
            {
                'timestamp': datetime.now().isoformat(),
                'technical_features': {
                    'symbol': 'MSFT',
                    'current_price': 300.0,
                    'volume': 2000000,
                    'rsi': 35.0,
                    'macd': -0.3,
                    'bb_position': 0.3,
                    'volatility': 0.03,
                    'volume_ratio': 0.8,
                    'price_change_1d': -0.02,
                    'price_change_5d': -0.08,
                    'price_change_20d': -0.12,
                    'return_5d': -8.0,
                    'return_20d': -12.0,
                    'return_60d': -5.0,
                    'volatility_20d': 25.0,
                    'max_drawdown': -15.0
                },
                'sentiment_analysis': {
                    'sentiment_score': -0.2,
                    'positive': 0.3,
                    'negative': 0.7,
                    'keywords': ['bearish', 'resistance', 'decline'],
                    'confidence': 'high'
                },
                'user_decision': 'SELL'
            }
        ]
        
        # Add examples to trainer
        trainer.training_examples = sample_examples
        print(f"   Added {len(sample_examples)} training examples")
        print(f"   Total examples: {len(trainer.training_examples)}")
        
        # Test 5: Training data preparation
        print("\n5️⃣ Testing Training Data Preparation:")
        X, y = trainer.prepare_training_data()
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Labels shape: {y.shape}")
        print(f"   Feature vector size: {X.shape[1] if X.size > 0 else 0}")
        print(f"   Unique labels: {set(y) if y.size > 0 else 'None'}")
        
        # Test 6: Model state management
        print("\n6️⃣ Testing Model State Management:")
        trainer.save_model_state(
            is_trained=True,
            training_examples=len(trainer.training_examples),
            model_accuracy=0.85,
            model_type=trainer.model_type
        )
        print(f"   Model state saved")
        
        model_state = trainer.get_model_state()
        print(f"   Model state loaded:")
        print(f"      - Is trained: {model_state.get('is_trained', False)}")
        print(f"      - Training examples: {model_state.get('training_examples', 0)}")
        print(f"      - Accuracy: {model_state.get('model_accuracy', 0):.2%}")
        print(f"      - Model type: {model_state.get('model_type', 'Unknown')}")
        
        # Test 7: Training statistics
        print("\n7️⃣ Testing Training Statistics:")
        stats = trainer.get_training_stats()
        print(f"   Total examples: {stats.get('total_examples', 0)}")
        print(f"   Symbols trained: {stats.get('symbols_trained', 0)}")
        
        decision_dist = stats.get('decision_distribution', {})
        print(f"   Decision distribution: {decision_dist}")
        
        sentiment_dist = stats.get('sentiment_distribution', {})
        print(f"   Sentiment analysis: {sentiment_dist}")
        
        # Test 8: Current model info
        print("\n8️⃣ Testing Current Model Info:")
        model_info = trainer.get_current_model_info()
        print(f"   Model type: {model_info.get('model_type', 'Unknown')}")
        print(f"   Input dimension: {model_info.get('input_dim', 0)}")
        print(f"   Is trained: {model_info.get('is_trained', False)}")
        print(f"   Total parameters: {model_info.get('total_params', 0):,}")
        
        print("\n✅ All enhanced training tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def test_training_process():
    """Test the complete training process"""
    print("\n🔄 Testing Complete Training Process")
    print("=" * 50)
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(model_type="simple")
        print(f"✅ Trainer initialized with {trainer.model_type} model")
        
        # Add training examples
        sample_examples = [
            {
                'timestamp': datetime.now().isoformat(),
                'technical_features': {
                    'symbol': 'TSLA',
                    'current_price': 200.0,
                    'volume': 1500000,
                    'rsi': 70.0,
                    'macd': 0.8,
                    'bb_position': 0.8,
                    'volatility': 0.04,
                    'volume_ratio': 1.5,
                    'price_change_1d': 0.03,
                    'price_change_5d': 0.12,
                    'price_change_20d': 0.25,
                    'return_5d': 12.0,
                    'return_20d': 25.0,
                    'return_60d': 40.0,
                    'volatility_20d': 30.0,
                    'max_drawdown': -8.0
                },
                'sentiment_analysis': {
                    'sentiment_score': 0.6,
                    'positive': 0.8,
                    'negative': 0.2,
                    'keywords': ['bullish', 'breakout', 'momentum'],
                    'confidence': 'high'
                },
                'user_decision': 'BUY'
            }
        ]
        
        trainer.training_examples = sample_examples
        print(f"✅ Added {len(sample_examples)} training example")
        
        # Test training (with minimal epochs for testing)
        print("🔄 Starting training process...")
        success = trainer.train_neural_network(epochs=5)  # Minimal epochs for testing
        
        if success:
            print("✅ Training completed successfully!")
            
            # Test prediction
            print("🔮 Testing prediction...")
            features = sample_examples[0]['technical_features']
            prediction = trainer.make_prediction(features)
            print(f"   Prediction result: {prediction}")
        else:
            print("❌ Training failed")
            
    except Exception as e:
        print(f"❌ Error in training process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_training()
    test_training_process() 