"""Tests for training.training_loop."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_2_Orchestrator_And_ML_Python"))
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock

def _make_df():
    n = 30
    dates = pd.date_range(end=datetime.utcnow(), periods=n, freq="B")
    c = 100 + np.cumsum(np.random.randn(n))
    return pd.DataFrame({"Open": c, "High": c+1, "Low": c-1,
                          "Close": c, "Volume": 1e6*np.ones(n)}, index=dates)

def _make_mock_trainer(n=0):
    t = MagicMock()
    t.training_examples = [{"decision": "BUY"}] * n
    t.symbols = ["AAPL", "MSFT", "GOOG", "TSLA", "META"]
    t._calculate_performance_metrics.return_value = {"rsi": 0.5}
    t.analyze_sentiment_and_keywords.return_value = {"sentiment": 0.1}
    t.add_training_example.return_value = True
    t.model_metadata = {"min_epochs": 50}
    return t

def test_progress_returns_tuple():
    from training.training_loop import TrainingLoopController
    c, r = TrainingLoopController(_make_mock_trainer(10), 50).progress()
    assert c == 10 and r == 50

def test_not_ready_below_threshold():
    from training.training_loop import TrainingLoopController
    assert not TrainingLoopController(_make_mock_trainer(49), 50).is_ready_to_train()

def test_ready_at_threshold():
    from training.training_loop import TrainingLoopController
    assert TrainingLoopController(_make_mock_trainer(50), 50).is_ready_to_train()

def test_decision_counts_all_keys():
    from training.training_loop import TrainingLoopController
    counts = TrainingLoopController(_make_mock_trainer(0), 50).decision_counts()
    assert {"BUY", "SELL", "HOLD"} <= set(counts.keys())

def test_class_balance_warning_fires():
    from training.training_loop import TrainingLoopController
    t = MagicMock()
    t.model_metadata = {}
    t.training_examples = [{"decision": "BUY"}]*40 + [{"decision": "SELL"}]*40
    assert any("HOLD" in w for w in TrainingLoopController(t, 50).class_balance_warnings())

def test_submit_decision_calls_trainer():
    from training.training_loop import TrainingLoopController
    t = _make_mock_trainer(0)
    TrainingLoopController(t, 50).submit_decision(_make_df(), "BUY", "bullish", "AAPL")
    t.add_training_example.assert_called_once()

def test_submit_invalid_decision_rejected():
    from training.training_loop import TrainingLoopController
    assert not TrainingLoopController(_make_mock_trainer(0), 50).submit_decision(
        _make_df(), "MAYBE", "hmm", "X")

def test_submit_empty_reasoning_rejected():
    from training.training_loop import TrainingLoopController
    assert not TrainingLoopController(_make_mock_trainer(0), 50).submit_decision(
        _make_df(), "BUY", "   ", "AAPL")
