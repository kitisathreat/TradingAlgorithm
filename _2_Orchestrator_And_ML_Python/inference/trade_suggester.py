"""Inference-time trade suggestion with tunable sliders."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class InferenceParams:
    risk_tolerance: float = 0.5
    information_amount: float = 1.0
    fundamental_weight: float = 0.3
    confidence_threshold: float = 0.6
    max_position_pct: float = 0.10
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15
    technical_weight: float = 0.4
    sentiment_weight: float = 0.3

    @classmethod
    def load(cls, path: Path) -> "InferenceParams":
        if path.exists():
            try:
                return cls(**json.loads(path.read_text()))
            except Exception as exc:
                logger.warning("Couldn't load inference params from %s: %s", path, exc)
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))


@dataclass
class TradeSuggestion:
    symbol: str
    decision: str
    confidence: float
    shares: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str
    feature_snapshot: Dict[str, Any] = field(default_factory=dict)


class TradeSuggester:
    def __init__(self, trainer, params: Optional[InferenceParams] = None):
        self.trainer = trainer
        self.params = params or InferenceParams()

    def suggest(self, symbol: str, stock_df,
                fundamental_kpis: Optional[Dict[str, float]] = None,
                portfolio=None) -> TradeSuggestion:
        p = self.params
        features = self.trainer._calculate_performance_metrics(stock_df)
        current_price = self._latest_price(stock_df)
        effective_features = self._apply_information_filter(features, fundamental_kpis, p.information_amount)
        nn_prediction = self.trainer.make_prediction(effective_features)
        if isinstance(nn_prediction, dict) and nn_prediction.get("error"):
            return TradeSuggestion(symbol=symbol, decision="HOLD", confidence=0.0, shares=0,
                                   entry_price=current_price, stop_loss=0.0, take_profit=0.0,
                                   reasoning=f"Model error: {nn_prediction['error']}",
                                   feature_snapshot=effective_features)
        nn_decision = nn_prediction.get("prediction") or nn_prediction.get("predicted_decision") or "HOLD"
        nn_conf = float(nn_prediction.get("confidence", 0.0))
        fundamental_score = self._score_fundamentals(fundamental_kpis) if fundamental_kpis else 0.0
        if p.fundamental_weight > 0 and fundamental_kpis:
            blended_conf = (1 - p.fundamental_weight) * nn_conf + p.fundamental_weight * abs(fundamental_score)
        else:
            blended_conf = nn_conf
        if blended_conf < p.confidence_threshold:
            return TradeSuggestion(symbol=symbol, decision="HOLD", confidence=blended_conf, shares=0,
                                   entry_price=current_price, stop_loss=0.0, take_profit=0.0,
                                   reasoning=f"Confidence {blended_conf:.1%} below threshold {p.confidence_threshold:.1%} \u2192 HOLD.",
                                   feature_snapshot=effective_features)
        if fundamental_kpis and p.fundamental_weight > 0.5:
            if nn_decision == "BUY" and fundamental_score < -0.3:
                return TradeSuggestion(symbol=symbol, decision="HOLD", confidence=blended_conf, shares=0,
                                       entry_price=current_price, stop_loss=0.0, take_profit=0.0,
                                       reasoning="NN bullish but fundamentals are negative \u2014 abstaining.",
                                       feature_snapshot=effective_features)
            if nn_decision == "SELL" and fundamental_score > 0.3:
                return TradeSuggestion(symbol=symbol, decision="HOLD", confidence=blended_conf, shares=0,
                                       entry_price=current_price, stop_loss=0.0, take_profit=0.0,
                                       reasoning="NN bearish but fundamentals are positive \u2014 abstaining.",
                                       feature_snapshot=effective_features)
        shares = self._size_position(nn_decision, blended_conf, current_price, portfolio)
        if nn_decision == "BUY":
            stop = current_price * (1 - p.stop_loss_pct)
            target = current_price * (1 + p.take_profit_pct)
        elif nn_decision == "SELL":
            stop = current_price * (1 + p.stop_loss_pct)
            target = current_price * (1 - p.take_profit_pct)
        else:
            stop = target = 0.0
        reasoning = (f"NN: {nn_decision} ({nn_conf:.1%} conf). "
                     f"Fundamental score: {fundamental_score:+.2f}. "
                     f"Blended: {blended_conf:.1%}. "
                     f"Risk tolerance {p.risk_tolerance:.2f} \u2192 {shares:.2f} shares.")
        return TradeSuggestion(symbol=symbol, decision=nn_decision, confidence=blended_conf,
                               shares=shares, entry_price=current_price, stop_loss=stop,
                               take_profit=target, reasoning=reasoning,
                               feature_snapshot=effective_features)

    @staticmethod
    def _latest_price(stock_df) -> float:
        try:
            return float(stock_df["Close"].iloc[-1])
        except Exception:
            return 0.0

    @staticmethod
    def _apply_information_filter(features, fundamental_kpis, information_amount):
        effective = dict(features)
        if information_amount < 0.5:
            for k in list(effective):
                if "sentiment" in k.lower():
                    effective[k] = 0.0
        if information_amount < 0.75 and fundamental_kpis:
            for k in fundamental_kpis:
                if k in effective:
                    effective[k] = 0.0
        elif fundamental_kpis:
            effective.update(fundamental_kpis)
        return effective

    @staticmethod
    def _score_fundamentals(kpis) -> float:
        if not kpis:
            return 0.0
        weights = {"net_margin": 0.15, "operating_margin": 0.15, "gross_margin": 0.10,
                   "revenue_growth_yoy": 0.25, "roe": 0.20, "current_ratio": 0.05,
                   "fcf_to_capex": 0.10}
        score = total_w = 0.0
        for key, w in weights.items():
            if key in kpis:
                score += kpis[key] * w
                total_w += w
        return 0.0 if total_w == 0 else max(-1.0, min(1.0, score / total_w))

    def _size_position(self, decision, confidence, current_price, portfolio) -> float:
        if decision == "HOLD" or current_price <= 0:
            return 0.0
        p = self.params
        cash = max(0.0, portfolio.cash) if portfolio is not None else 100_000.0
        budget = cash * p.max_position_pct * p.risk_tolerance * confidence
        return round(budget / current_price, 4) if budget > 0 else 0.0
