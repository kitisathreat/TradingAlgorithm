#pragma once

#include <string>

// Enum to define the clear trading signals
enum class TradeSignal {
    HOLD,
    BUY,
    SELL
};

// This class encapsulates the logic of a rule-based trading model.
class TradingModel {
public:
    // The core decision-making function.
    TradeSignal get_trading_decision(
        const std::string& symbol,
        double current_price,
        double sentiment_score,
        double analyst_buy_ratio
    );
}; 
