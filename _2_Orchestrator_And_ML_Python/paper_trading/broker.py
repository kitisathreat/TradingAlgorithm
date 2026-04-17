"""Simulated order execution."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from .portfolio import Portfolio, Trade

logger = logging.getLogger(__name__)


@dataclass
class BrokerConfig:
    slippage_bps: float = 0.0
    commission_per_trade: float = 0.0


class PaperBroker:
    def __init__(self, portfolio: Portfolio,
                 price_lookup: Optional[Callable[[str], float]] = None,
                 config: Optional[BrokerConfig] = None):
        self.portfolio = portfolio
        self.config = config or BrokerConfig()
        self._price_lookup = price_lookup or self._default_price_lookup

    def execute_order(self, symbol: str, side: str, shares: float,
                      source: str = "manual", reasoning: str = "") -> Trade:
        price = self._apply_slippage(self._price_lookup(symbol), side)
        if side.upper() == "BUY":
            return self.portfolio.buy(symbol, shares, price,
                                      commission=self.config.commission_per_trade,
                                      source=source, reasoning=reasoning)
        elif side.upper() == "SELL":
            return self.portfolio.sell(symbol, shares, price,
                                       commission=self.config.commission_per_trade,
                                       source=source, reasoning=reasoning)
        raise ValueError(f"Unknown side: {side}")

    def execute_suggestion(self, suggestion, reasoning: str = "") -> Optional[Trade]:
        if suggestion.decision == "HOLD" or suggestion.shares <= 0:
            return None
        return self.execute_order(suggestion.symbol, suggestion.decision, suggestion.shares,
                                  source="model_suggestion",
                                  reasoning=reasoning or suggestion.reasoning)

    def _apply_slippage(self, price: float, side: str) -> float:
        adj = price * self.config.slippage_bps / 10_000.0
        return price + adj if side.upper() == "BUY" else price - adj

    @staticmethod
    def _default_price_lookup(symbol: str) -> float:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        fast = getattr(ticker, "fast_info", None)
        if fast is not None:
            price = fast.get("last_price") or fast.get("lastPrice")
            if price:
                return float(price)
        hist = ticker.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        raise RuntimeError(f"Could not resolve price for {symbol}")
