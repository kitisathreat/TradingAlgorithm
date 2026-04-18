"""Tests for inference.trade_suggester."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_2_Orchestrator_And_ML_Python"))
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from datetime import datetime

def _mock_trainer(decision="BUY", confidence=0.8):
    t = MagicMock()
    t.model_metadata = {"trained": True}
    t._calculate_performance_metrics.return_value = {f"f{i}": 0.1 for i in range(20)}
    t.make_prediction.return_value = {"prediction": decision, "confidence": confidence, "error": None}
    t.user_dir = Path("/tmp")
    return t

def _make_df(n=30):
    dates = pd.date_range(end=datetime.utcnow(), periods=n, freq="B")
    n = len(dates)
    c = 100 + np.arange(n, dtype=float)
    return pd.DataFrame({"Open": c, "High": c+1, "Low": c-1, "Close": c,
                          "Volume": 1e6*np.ones(n)}, index=dates)

def test_suggestion_has_required_fields(inference_params_default):
    from inference.trade_suggester import TradeSuggester
    sug = TradeSuggester(_mock_trainer(), inference_params_default).suggest("AAPL", _make_df())
    assert hasattr(sug, "decision") and hasattr(sug, "confidence") and hasattr(sug, "shares")

def test_low_confidence_forces_hold():
    from inference.trade_suggester import TradeSuggester, InferenceParams
    sug = TradeSuggester(_mock_trainer(confidence=0.5), InferenceParams(confidence_threshold=0.95)).suggest("AAPL", _make_df())
    assert sug.decision == "HOLD"

def test_high_confidence_not_held():
    from inference.trade_suggester import TradeSuggester, InferenceParams
    sug = TradeSuggester(_mock_trainer(decision="BUY", confidence=0.9),
                         InferenceParams(confidence_threshold=0.5)).suggest("AAPL", _make_df())
    assert sug.decision == "BUY"

def test_risk_tolerance_scales_shares():
    from inference.trade_suggester import TradeSuggester, InferenceParams
    from paper_trading.portfolio import Portfolio
    p = Portfolio.__new__(Portfolio)
    p.cash = 100_000.0
    p.positions = {}
    p.trades = []
    p.starting_cash = 100_000.0
    t = _mock_trainer(decision="BUY", confidence=0.9)
    s_low = TradeSuggester(t, InferenceParams(risk_tolerance=0.1, confidence_threshold=0.0)).suggest("AAPL", _make_df(), portfolio=p)
    s_high = TradeSuggester(t, InferenceParams(risk_tolerance=1.0, confidence_threshold=0.0)).suggest("AAPL", _make_df(), portfolio=p)
    assert s_high.shares >= s_low.shares

def test_fundamental_weight_blending():
    from inference.trade_suggester import TradeSuggester, InferenceParams
    kpis = {k: 1.0 for k in ["gross_margin","operating_margin","net_margin",
                               "revenue_growth_yoy","roe","current_ratio","fcf_to_capex"]}
    t = _mock_trainer(decision="BUY", confidence=0.5)
    s_no = TradeSuggester(t, InferenceParams(fundamental_weight=0.0, confidence_threshold=0.0)).suggest("AAPL", _make_df(), fundamental_kpis=kpis)
    s_fund = TradeSuggester(t, InferenceParams(fundamental_weight=1.0, confidence_threshold=0.0)).suggest("AAPL", _make_df(), fundamental_kpis=kpis)
    assert s_fund.confidence >= s_no.confidence
