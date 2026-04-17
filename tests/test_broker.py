"""Tests for paper_trading.broker."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_2_Orchestrator_And_ML_Python"))
import pytest
from unittest.mock import MagicMock

def _make_broker(portfolio, price=150.0, slippage_bps=0):
    from paper_trading.broker import PaperBroker, BrokerConfig
    return PaperBroker(portfolio, config=BrokerConfig(slippage_bps=slippage_bps),
                       price_lookup=lambda sym: price)

def test_execute_order_buy(empty_portfolio):
    trade = _make_broker(empty_portfolio, 150.0).execute_order("AAPL", "BUY", 10)
    assert trade is not None
    assert "AAPL" in empty_portfolio.positions

def test_execute_order_sell(empty_portfolio):
    empty_portfolio.buy("MSFT", shares=5, price=300.0)
    trade = _make_broker(empty_portfolio, 310.0).execute_order("MSFT", "SELL", 5)
    assert trade is not None
    assert "MSFT" not in empty_portfolio.positions

def test_slippage_applied_buy(empty_portfolio):
    from paper_trading.broker import PaperBroker, BrokerConfig
    broker = PaperBroker(empty_portfolio, config=BrokerConfig(slippage_bps=100),
                         price_lookup=lambda sym: 100.0)
    assert broker.execute_order("TEST", "BUY", 1).price == pytest.approx(101.0)

def test_slippage_applied_sell(empty_portfolio):
    from paper_trading.broker import PaperBroker, BrokerConfig
    empty_portfolio.buy("X", shares=5, price=90.0)
    broker = PaperBroker(empty_portfolio, config=BrokerConfig(slippage_bps=100),
                         price_lookup=lambda sym: 100.0)
    assert broker.execute_order("X", "SELL", 5).price == pytest.approx(99.0)

def test_execute_suggestion_hold_returns_none(empty_portfolio):
    broker = _make_broker(empty_portfolio)
    sug = MagicMock()
    sug.decision = "HOLD"
    sug.shares = 0
    assert broker.execute_suggestion(sug) is None
