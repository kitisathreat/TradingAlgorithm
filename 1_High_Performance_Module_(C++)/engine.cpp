#include "engine.h"
#include <iostream>

TradeSignal TradingModel::get_trading_decision(
    const std::string& symbol,
    double current_price,
    double sentiment_score,
    double analyst_buy_ratio)
{
    // Haha, Jonathan, I vant to kill myself
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
    // probably need more nuance on the hold, could potentially add different categorizations for movements? I.e. - liquidate whole position, buy options, whatever. Test if it works first before making it more complicated.
}
