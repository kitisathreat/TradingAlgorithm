"""Virtual-currency portfolio tracking for paper trading."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    shares: float
    avg_cost: float

    def market_value(self, price: float) -> float:
        return self.shares * price

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.avg_cost) * self.shares


@dataclass
class Trade:
    timestamp: str
    symbol: str
    side: str
    shares: float
    price: float
    commission: float = 0.0
    source: str = "manual"
    reasoning: str = ""

    @classmethod
    def now(cls, symbol: str, side: str, shares: float, price: float, **kw) -> "Trade":
        return cls(timestamp=datetime.now(timezone.utc).isoformat(),
                   symbol=symbol.upper(), side=side.upper(),
                   shares=shares, price=price, **kw)


class InsufficientFundsError(ValueError):
    pass


class InsufficientSharesError(ValueError):
    pass


@dataclass
class Portfolio:
    username: str
    starting_cash: float = 100_000.0
    cash: float = 100_000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    equity_history: List[Dict[str, float]] = field(default_factory=list)
    storage_path: Optional[Path] = None

    @classmethod
    def from_username(cls, username: str, starting_cash: float = 100_000.0,
                      user_root: Path = Path("data") / "users") -> "Portfolio":
        path = user_root / username / "trades" / "portfolio.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return cls._load(path)
        p = cls(username=username, starting_cash=starting_cash, cash=starting_cash, storage_path=path)
        p.save()
        return p

    @classmethod
    def _load(cls, path: Path) -> "Portfolio":
        with path.open("r") as f:
            raw = json.load(f)
        positions = {sym: Position(**pos) for sym, pos in raw.get("positions", {}).items()}
        trades = [Trade(**t) for t in raw.get("trades", [])]
        return cls(username=raw.get("username", ""), starting_cash=raw.get("starting_cash", 100_000.0),
                   cash=raw.get("cash", 100_000.0), positions=positions, trades=trades,
                   equity_history=raw.get("equity_history", []), storage_path=path)

    def save(self) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"username": self.username, "starting_cash": self.starting_cash,
                "cash": self.cash,
                "positions": {s: asdict(p) for s, p in self.positions.items()},
                "trades": [asdict(t) for t in self.trades],
                "equity_history": self.equity_history}
        with self.storage_path.open("w") as f:
            json.dump(data, f, indent=2)

    def buy(self, symbol: str, shares: float, price: float,
            commission: float = 0.0, source: str = "manual", reasoning: str = "") -> Trade:
        symbol = symbol.upper()
        if shares <= 0:
            raise ValueError("shares must be positive")
        cost = shares * price + commission
        if cost > self.cash + 1e-9:
            raise InsufficientFundsError(f"Need ${cost:,.2f} but only have ${self.cash:,.2f}")
        self.cash -= cost
        existing = self.positions.get(symbol)
        if existing is None:
            self.positions[symbol] = Position(symbol=symbol, shares=shares, avg_cost=price)
        else:
            total_shares = existing.shares + shares
            existing.avg_cost = (existing.avg_cost * existing.shares + price * shares) / total_shares
            existing.shares = total_shares
        trade = Trade.now(symbol, "BUY", shares, price, commission=commission,
                          source=source, reasoning=reasoning)
        self.trades.append(trade)
        self.save()
        return trade

    def sell(self, symbol: str, shares: float, price: float,
             commission: float = 0.0, source: str = "manual", reasoning: str = "") -> Trade:
        symbol = symbol.upper()
        if shares <= 0:
            raise ValueError("shares must be positive")
        existing = self.positions.get(symbol)
        if existing is None or existing.shares < shares - 1e-9:
            have = existing.shares if existing else 0
            raise InsufficientSharesError(f"Can't sell {shares} {symbol}, only own {have}")
        self.cash += shares * price - commission
        existing.shares -= shares
        if existing.shares <= 1e-9:
            del self.positions[symbol]
        trade = Trade.now(symbol, "SELL", shares, price, commission=commission,
                          source=source, reasoning=reasoning)
        self.trades.append(trade)
        self.save()
        return trade

    def equity(self, price_lookup: Callable[[str], float]) -> float:
        total = self.cash
        for sym, pos in self.positions.items():
            try:
                total += pos.market_value(price_lookup(sym))
            except Exception:
                total += pos.market_value(pos.avg_cost)
        return total

    def total_return(self, price_lookup: Callable[[str], float]) -> float:
        if self.starting_cash == 0:
            return 0.0
        return (self.equity(price_lookup) - self.starting_cash) / self.starting_cash

    def max_drawdown(self) -> float:
        if not self.equity_history:
            return 0.0
        peak = -float("inf")
        max_dd = 0.0
        for snap in self.equity_history:
            val = snap.get("equity", 0.0)
            peak = max(peak, val)
            if peak > 0:
                max_dd = max(max_dd, (peak - val) / peak)
        return max_dd

    def sharpe_ratio(self, price_lookup: Callable[[str], float], risk_free: float = 0.02) -> float:
        if len(self.equity_history) < 2:
            return 0.0
        equities = [s["equity"] for s in self.equity_history]
        returns = [(b - a) / a for a, b in zip(equities, equities[1:]) if a > 0]
        if not returns:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / max(1, len(returns) - 1)
        std = math.sqrt(var)
        return 0.0 if std == 0 else (mean - risk_free / 252) / std
