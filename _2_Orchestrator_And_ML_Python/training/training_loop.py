"""Iterative training-example collection loop."""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class StockPrompt:
    symbol: str
    stock_df: Any
    stock_info: Dict[str, Any]
    prompt_id: str


class TrainingLoopController:
    DEFAULT_MIN_REQUIRED = 50
    CLASS_BALANCE_FLOOR = 0.15

    def __init__(self, trainer, min_required: Optional[int] = None):
        self.trainer = trainer
        self.min_required = min_required or self._infer_min_required(trainer)
        self._seen_prompts: set[str] = set()

    def progress(self) -> Tuple[int, int]:
        return len(self.trainer.training_examples), self.min_required

    def is_ready_to_train(self) -> bool:
        return len(self.trainer.training_examples) >= self.min_required

    def decision_counts(self) -> Dict[str, int]:
        counts = Counter(ex.get("decision", "UNKNOWN") for ex in self.trainer.training_examples)
        for key in ("BUY", "SELL", "HOLD"):
            counts.setdefault(key, 0)
        return dict(counts)

    def class_balance_warnings(self) -> List[str]:
        total = sum(self.decision_counts().values())
        if total < 10:
            return []
        warnings = []
        for cls, count in self.decision_counts().items():
            pct = count / total
            if pct < self.CLASS_BALANCE_FLOOR:
                warnings.append(
                    f"Only {count}/{total} ({pct:.0%}) of decisions are {cls} \u2014 "
                    "the model may overfit to the dominant classes."
                )
        return warnings

    def next_training_prompt(self, days: int = 30, max_tries: int = 5) -> Optional[StockPrompt]:
        symbols = list(getattr(self.trainer, "symbols", []))
        random.shuffle(symbols)
        for symbol in symbols[:max_tries * 3]:
            prompt_id = self._prompt_id(symbol, days)
            if prompt_id in self._seen_prompts:
                continue
            stock_info, stock_df = self._fetch_stock(symbol, days)
            if stock_df is None or stock_df.empty:
                continue
            self._seen_prompts.add(prompt_id)
            return StockPrompt(symbol=symbol, stock_df=stock_df,
                               stock_info=stock_info, prompt_id=prompt_id)
        return None

    def reset_session_seen(self) -> None:
        self._seen_prompts.clear()

    def submit_decision(self, stock_df, decision: str, reasoning: str,
                        symbol: str = "", fundamental_kpis: Optional[Dict[str, float]] = None) -> bool:
        decision = decision.upper()
        if decision not in ("BUY", "SELL", "HOLD"):
            logger.warning("Invalid decision: %s", decision)
            return False
        if not reasoning or not reasoning.strip():
            logger.warning("Empty reasoning for %s", symbol)
            return False
        try:
            features = self.trainer._calculate_performance_metrics(stock_df)
            if fundamental_kpis:
                features = dict(features)
                features.update(fundamental_kpis)
            sentiment = self.trainer.analyze_sentiment_and_keywords(reasoning)
            ok = self.trainer.add_training_example(features, sentiment, decision)
            return bool(ok)
        except Exception as exc:
            logger.exception("Could not submit training example: %s", exc)
            return False

    def _fetch_stock(self, symbol: str, days: int) -> Tuple[Dict, Any]:
        try:
            import yfinance as yf
        except ImportError:
            return {}, None
        try:
            end = datetime.utcnow()
            start = end - timedelta(days=days * 2)
            df = yf.Ticker(symbol).history(start=start, end=end)
            if df is None or df.empty:
                return {}, None
            df = df.tail(days)
            return ({"current_price": float(df["Close"].iloc[-1]),
                     "volume": float(df["Volume"].iloc[-1])}, df)
        except Exception as exc:
            logger.warning("yfinance failed for %s: %s", symbol, exc)
            return {}, None

    @staticmethod
    def _prompt_id(symbol: str, days: int) -> str:
        bucket = datetime.utcnow().strftime("%Y-%m-%d")
        return f"{symbol}:{days}:{bucket}"

    @staticmethod
    def _infer_min_required(trainer) -> int:
        meta = getattr(trainer, "model_metadata", {}) or {}
        min_epochs = int(meta.get("min_epochs", 50))
        return max(50, min_epochs)
