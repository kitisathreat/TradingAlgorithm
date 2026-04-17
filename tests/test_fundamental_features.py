"""Tests for FundamentalFeatureExtractor."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_2_Orchestrator_And_ML_Python"))
import pytest
import pandas as pd

def test_all_keys_present(mock_financial_model):
    from fundamental.fundamental_features import FundamentalFeatureExtractor
    kpis = FundamentalFeatureExtractor().extract(mock_financial_model, "TEST")
    assert set(kpis.keys()) == set(FundamentalFeatureExtractor.KEYS)

def test_values_normalized(mock_financial_model):
    from fundamental.fundamental_features import FundamentalFeatureExtractor
    kpis = FundamentalFeatureExtractor().extract(mock_financial_model, "TEST")
    for k, v in kpis.items():
        assert -1.0 <= v <= 1.0, f"{k}={v} out of [-1,1]"

def test_gross_margin_positive(mock_financial_model):
    from fundamental.fundamental_features import FundamentalFeatureExtractor
    kpis = FundamentalFeatureExtractor().extract(mock_financial_model, "TEST")
    assert kpis["gross_margin"] > 0

def test_missing_income_returns_zeros():
    from fundamental.fundamental_features import FundamentalFeatureExtractor
    kpis = FundamentalFeatureExtractor().extract({}, "EMPTY")
    assert all(v == 0.0 for v in kpis.values())

def test_revenue_growth_positive(mock_financial_model):
    from fundamental.fundamental_features import FundamentalFeatureExtractor
    kpis = FundamentalFeatureExtractor().extract(mock_financial_model, "TEST")
    assert kpis["revenue_growth_yoy"] > 0

def test_clamp_handles_extreme_values():
    from fundamental.fundamental_features import FundamentalFeatureExtractor
    assert FundamentalFeatureExtractor._clamp(999.0) == 1.0
    assert FundamentalFeatureExtractor._clamp(-999.0) == -1.0
    assert FundamentalFeatureExtractor._clamp(float("nan")) == 0.0
