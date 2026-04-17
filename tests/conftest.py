"""Shared pytest fixtures."""
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def sample_ohlcv_df():
    np.random.seed(42)
    n = 30
    dates = pd.date_range(end=datetime.utcnow(), periods=n, freq="B")
    close = 100.0 + np.cumsum(np.random.randn(n))
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    return pd.DataFrame({"Open": close - np.random.rand(n) * 0.5,
                         "High": high, "Low": low, "Close": close,
                         "Volume": np.random.randint(1_000_000, 5_000_000, n).astype(float)},
                        index=dates)

@pytest.fixture
def tmp_user_dir(tmp_path):
    uname = "testuser"
    base = tmp_path / "data" / "users" / uname
    for sub in ("model", "trades", "training", "fundamentals_cache"):
        (base / sub).mkdir(parents=True)
    return base

@pytest.fixture
def user_store_tmp(tmp_path):
    yaml_path = tmp_path / "users.yaml"
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "_2_Orchestrator_And_ML_Python"))
    from auth.user_store import UserStore
    return UserStore(path=yaml_path)

@pytest.fixture
def empty_portfolio(tmp_user_dir):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "_2_Orchestrator_And_ML_Python"))
    from paper_trading.portfolio import Portfolio
    return Portfolio.from_username("testuser", user_root=tmp_user_dir.parent)

@pytest.fixture
def inference_params_default():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "_2_Orchestrator_And_ML_Python"))
    from inference.trade_suggester import InferenceParams
    return InferenceParams()

@pytest.fixture
def mock_financial_model():
    idx = pd.date_range("2022-01-01", periods=2, freq="YE")
    income = pd.DataFrame({
        "Revenues": [90_000.0, 100_000.0],
        "CostOfRevenue": [55_000.0, 60_000.0],
        "OperatingIncomeLoss": [18_000.0, 20_000.0],
        "NetIncomeLoss": [12_000.0, 15_000.0],
    }, index=idx)
    balance = pd.DataFrame({
        "AssetsCurrent": [50_000.0, 45_000.0],
        "LiabilitiesCurrent": [25_000.0, 22_000.0],
        "StockholdersEquity": [80_000.0, 70_000.0],
    }, index=idx)
    cash = pd.DataFrame({
        "NetCashProvidedByUsedInOperatingActivities": [22_000.0, 20_000.0],
        "PaymentsToAcquirePropertyPlantAndEquipment": [10_000.0, 9_000.0],
    }, index=idx)
    return {"annual_income_statement": income, "annual_balance_sheet": balance, "annual_cash_flow": cash}
