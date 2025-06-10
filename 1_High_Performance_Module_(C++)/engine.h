#pragma once

#include <string>

// Enum to define the clear trading signals
enum class TradeSignal {
    HOLD,
    BUY,
    SELL
// this'll probably need some edits when I add the additional categorization, but that's more complicated 
};

// Rule-based trading model logic. simple for now, complications later.
class TradingModel {
public:
    // The core decision-making function.
    TradeSignal get_trading_decision(
        const std::string& symbol,
        double current_price,
        double sentiment_score,
        double analyst_buy_ratio
    // Definitely need to add more decisionmaking functionality. Rn it's simple, maybe I could train it to do a neural network? Idk if there's good tensorflow integration with C
    );
}; 
