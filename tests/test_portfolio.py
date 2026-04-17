"""Tests for paper_trading.portfolio."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_2_Orchestrator_And_ML_Python"))
import pytest

def test_buy_decreases_cash(empty_portfolio):
    p = empty_portfolio
    initial = p.cash
    p.buy("AAPL", shares=10, price=150.0)
    assert p.cash == pytest.approx(initial - 10 * 150.0)

def test_buy_creates_position(empty_portfolio):
    p = empty_portfolio
    p.buy("AAPL", shares=10, price=150.0)
    assert "AAPL" in p.positions
    assert p.positions["AAPL"].shares == 10

def test_sell_removes_position(empty_portfolio):
    p = empty_portfolio
    p.buy("AAPL", shares=10, price=150.0)
    p.sell("AAPL", shares=10, price=160.0)
    assert "AAPL" not in p.positions

def test_partial_sell_reduces_shares(empty_portfolio):
    p = empty_portfolio
    p.buy("AAPL", shares=10, price=150.0)
    p.sell("AAPL", shares=4, price=155.0)
    assert p.positions["AAPL"].shares == pytest.approx(6.0)

def test_equity_includes_positions(empty_portfolio):
    p = empty_portfolio
    p.buy("AAPL", shares=10, price=150.0)
    assert p.equity(lambda sym: 160.0) == pytest.approx(p.cash + 10 * 160.0)

def test_save_and_reload(empty_portfolio):
    p = empty_portfolio
    p.buy("MSFT", shares=5, price=300.0)
    p.save()
    from paper_trading.portfolio import Portfolio
    p2 = Portfolio._load(p.storage_path)
    assert "MSFT" in p2.positions
    assert p2.positions["MSFT"].shares == 5

def test_cannot_sell_more_than_held(empty_portfolio):
    from paper_trading.portfolio import InsufficientSharesError
    empty_portfolio.buy("TSLA", shares=5, price=200.0)
    with pytest.raises(InsufficientSharesError):
        empty_portfolio.sell("TSLA", shares=10, price=200.0)

def test_buy_with_insufficient_cash(empty_portfolio):
    from paper_trading.portfolio import InsufficientFundsError
    with pytest.raises(InsufficientFundsError):
        empty_portfolio.buy("BRK.A", shares=1, price=1_000_000.0)
