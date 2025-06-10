#include "engine.h"
#include <iostream>

TradeSignal TradingModel::get_trading_decision(
    const std::string& symbol,
    double current_price,
    double sentiment_score,
    double analyst_buy_ratio)
{
    // --- Example Rule-Based Trading Logic ---
    std::cout << "[C++ Engine] Analyzing " << symbol
        << ": Price=" << current_price
        << ", Sentiment=" << sentiment_score
        << ", AnalystBuyRatio=" << analyst_buy_ratio << std::endl;

    bool strong_sentiment = sentiment_score > 0.5;
    bool strong_analyst_backing = analyst_buy_ratio > 0.7;

    if (strong_sentiment && strong_analyst_backing) {
        std::cout << "[C++ Engine] Decision: BUY" << std::endl;
        return TradeSignal::BUY;
    }

    if (sentiment_score < -0.5) {
        std::cout << "[C++ Engine] Decision: SELL" << std::endl;
        return TradeSignal::SELL;
    }

    std::cout << "[C++ Engine] Decision: HOLD" << std::endl;
    return TradeSignal::HOLD;
}