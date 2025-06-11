#include "engine.h"
#include <iostream>
#include <numeric>
#include <cmath>
#include <algorithm>

// TechnicalAnalyzer Implementation
TechnicalAnalyzer::TechnicalAnalyzer(int sma_period, int ema_period, 
                                   int rsi_period, int macd_fast, 
                                   int macd_slow, int macd_signal,
                                   int bb_period, double bb_std)
    : sma_period_(sma_period), ema_period_(ema_period),
      rsi_period_(rsi_period), macd_fast_(macd_fast),
      macd_slow_(macd_slow), macd_signal_(macd_signal),
      bb_period_(bb_period), bb_std_(bb_std) {}

double TechnicalAnalyzer::calculate_sma(const std::vector<PriceData>& data, int period) {
    if (data.size() < period) return 0.0;
    
    double sum = 0.0;
    for (int i = data.size() - period; i < data.size(); ++i) {
        sum += data[i].price;
    }
    return sum / period;
}

double TechnicalAnalyzer::calculate_ema(const std::vector<PriceData>& data, int period) {
    if (data.size() < period) return 0.0;
    
    double multiplier = 2.0 / (period + 1);
    double ema = calculate_sma(data, period); // Start with SMA
    
    for (int i = data.size() - period; i < data.size(); ++i) {
        ema = (data[i].price - ema) * multiplier + ema;
    }
    return ema;
}

double TechnicalAnalyzer::calculate_rsi(const std::vector<PriceData>& data, int period) {
    if (data.size() < period + 1) return 50.0; // Neutral RSI
    
    std::vector<double> gains, losses;
    for (int i = 1; i < data.size(); ++i) {
        double change = data[i].price - data[i-1].price;
        gains.push_back(std::max(change, 0.0));
        losses.push_back(std::max(-change, 0.0));
    }
    
    double avg_gain = std::accumulate(gains.end() - period, gains.end(), 0.0) / period;
    double avg_loss = std::accumulate(losses.end() - period, losses.end(), 0.0) / period;
    
    if (avg_loss == 0.0) return 100.0;
    double rs = avg_gain / avg_loss;
    return 100.0 - (100.0 / (1.0 + rs));
}

std::tuple<double, double, double> TechnicalAnalyzer::calculate_macd(
    const std::vector<PriceData>& data) {
    
    double ema_fast = calculate_ema(data, macd_fast_);
    double ema_slow = calculate_ema(data, macd_slow_);
    double macd = ema_fast - ema_slow;
    
    // Calculate signal line (EMA of MACD)
    std::vector<PriceData> macd_data;
    for (const auto& price : data) {
        PriceData macd_point;
        macd_point.price = macd;
        macd_point.timestamp = price.timestamp;
        macd_data.push_back(macd_point);
    }
    double signal = calculate_ema(macd_data, macd_signal_);
    
    return std::make_tuple(macd, signal, macd - signal);
}

std::tuple<double, double, double> TechnicalAnalyzer::calculate_bollinger_bands(
    const std::vector<PriceData>& data) {
    
    double middle_band = calculate_sma(data, bb_period_);
    
    // Calculate standard deviation
    double sum = 0.0;
    for (int i = data.size() - bb_period_; i < data.size(); ++i) {
        sum += std::pow(data[i].price - middle_band, 2);
    }
    double std_dev = std::sqrt(sum / bb_period_);
    
    double upper_band = middle_band + (bb_std_ * std_dev);
    double lower_band = middle_band - (bb_std_ * std_dev);
    
    return std::make_tuple(upper_band, middle_band, lower_band);
}

double TechnicalAnalyzer::calculate_atr(const std::vector<PriceData>& data, int period) {
    if (data.size() < 2) return 0.0;
    
    std::vector<double> true_ranges;
    for (int i = 1; i < data.size(); ++i) {
        double high_low = std::abs(data[i].price - data[i-1].price);
        double high_close = std::abs(data[i].price - data[i-1].price);
        double low_close = std::abs(data[i-1].price - data[i-1].price);
        true_ranges.push_back(std::max({high_low, high_close, low_close}));
    }
    
    return calculate_sma(std::vector<PriceData>(true_ranges.begin(), true_ranges.end()), period);
}

TechnicalIndicators TechnicalAnalyzer::calculate_indicators(
    const std::vector<PriceData>& price_data) {
    
    TechnicalIndicators indicators;
    
    // Calculate all indicators
    indicators.sma_20 = calculate_sma(price_data, sma_period_);
    indicators.sma_50 = calculate_sma(price_data, 50);
    indicators.ema_20 = calculate_ema(price_data, ema_period_);
    indicators.rsi_14 = calculate_rsi(price_data, rsi_period_);
    
    auto [macd, signal, histogram] = calculate_macd(price_data);
    indicators.macd = macd;
    indicators.macd_signal = signal;
    indicators.macd_histogram = histogram;
    
    auto [upper, middle, lower] = calculate_bollinger_bands(price_data);
    indicators.upper_band = upper;
    indicators.middle_band = middle;
    indicators.lower_band = lower;
    
    indicators.atr = calculate_atr(price_data);
    
    return indicators;
}

// MarketAnalyzer Implementation
MarketAnalyzer::MarketAnalyzer() {}

MarketRegime MarketAnalyzer::detect_market_regime(const std::vector<PriceData>& price_data) {
    if (price_data.size() < 50) return MarketRegime::UNKNOWN;
    
    // Calculate trend using linear regression
    std::vector<double> x(price_data.size());
    std::iota(x.begin(), x.end(), 0);
    std::vector<double> y;
    for (const auto& data : price_data) {
        y.push_back(data.price);
    }
    
    double n = x.size();
    double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
    double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
    double sum_xy = 0.0;
    double sum_xx = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        sum_xy += x[i] * y[i];
        sum_xx += x[i] * x[i];
    }
    
    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    
    // Calculate volatility
    double volatility = 0.0;
    for (size_t i = 1; i < price_data.size(); ++i) {
        volatility += std::abs(price_data[i].price - price_data[i-1].price);
    }
    volatility /= price_data.size();
    
    // Determine regime
    if (volatility > 0.02) return MarketRegime::VOLATILE;
    if (slope > 0.001) return MarketRegime::TRENDING_UP;
    if (slope < -0.001) return MarketRegime::TRENDING_DOWN;
    return MarketRegime::RANGING;
}

double MarketAnalyzer::calculate_market_breadth(
    const std::vector<double>& sector_performance) {
    
    int advancing = 0;
    for (double perf : sector_performance) {
        if (perf > 0) advancing++;
    }
    return static_cast<double>(advancing) / sector_performance.size();
}

double MarketAnalyzer::calculate_sector_correlation(
    const std::vector<PriceData>& price_data,
    const std::vector<double>& sector_data) {
    
    if (price_data.size() != sector_data.size() || price_data.size() < 2) {
        return 0.0;
    }
    
    std::vector<double> price_returns, sector_returns;
    for (size_t i = 1; i < price_data.size(); ++i) {
        price_returns.push_back((price_data[i].price - price_data[i-1].price) / price_data[i-1].price);
        sector_returns.push_back(sector_data[i] - sector_data[i-1]);
    }
    
    double mean_price = std::accumulate(price_returns.begin(), price_returns.end(), 0.0) / price_returns.size();
    double mean_sector = std::accumulate(sector_returns.begin(), sector_returns.end(), 0.0) / sector_returns.size();
    
    double covariance = 0.0;
    double var_price = 0.0;
    double var_sector = 0.0;
    
    for (size_t i = 0; i < price_returns.size(); ++i) {
        covariance += (price_returns[i] - mean_price) * (sector_returns[i] - mean_sector);
        var_price += std::pow(price_returns[i] - mean_price, 2);
        var_sector += std::pow(sector_returns[i] - mean_sector, 2);
    }
    
    covariance /= price_returns.size();
    var_price /= price_returns.size();
    var_sector /= price_returns.size();
    
    return covariance / std::sqrt(var_price * var_sector);
}

MarketContext MarketAnalyzer::analyze_market_context(
    const std::vector<PriceData>& price_data,
    const std::vector<double>& sector_data,
    double vix) {
    
    MarketContext context;
    context.regime = detect_market_regime(price_data);
    context.sector_correlation = calculate_sector_correlation(price_data, sector_data);
    context.market_breadth = calculate_market_breadth(sector_data);
    context.vix = vix;
    context.sector_performance = std::accumulate(sector_data.end() - 5, sector_data.end(), 0.0) / 5;
    
    return context;
}

// RiskManager Implementation
RiskManager::RiskManager(double max_position_size, double max_drawdown)
    : max_position_size_(max_position_size), max_drawdown_(max_drawdown) {}

double RiskManager::calculate_volatility(const std::vector<PriceData>& price_data) {
    if (price_data.size() < 2) return 0.0;
    
    std::vector<double> returns;
    for (size_t i = 1; i < price_data.size(); ++i) {
        returns.push_back((price_data[i].price - price_data[i-1].price) / price_data[i-1].price);
    }
    
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double variance = 0.0;
    
    for (double ret : returns) {
        variance += std::pow(ret - mean, 2);
    }
    variance /= returns.size();
    
    return std::sqrt(variance);
}

double RiskManager::calculate_max_drawdown(const std::vector<PriceData>& price_data) {
    if (price_data.empty()) return 0.0;
    
    double peak = price_data[0].price;
    double max_drawdown = 0.0;
    
    for (const auto& data : price_data) {
        if (data.price > peak) {
            peak = data.price;
        }
        double drawdown = (peak - data.price) / peak;
        max_drawdown = std::max(max_drawdown, drawdown);
    }
    
    return max_drawdown;
}

double RiskManager::calculate_sharpe_ratio(const std::vector<PriceData>& price_data) {
    if (price_data.size() < 2) return 0.0;
    
    std::vector<double> returns;
    for (size_t i = 1; i < price_data.size(); ++i) {
        returns.push_back((price_data[i].price - price_data[i-1].price) / price_data[i-1].price);
    }
    
    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double volatility = calculate_volatility(price_data);
    
    // Assuming risk-free rate of 0.02 (2%)
    return (mean_return - 0.02) / volatility;
}

RiskMetrics RiskManager::calculate_risk_metrics(
    const std::vector<PriceData>& price_data,
    double current_position_size) {
    
    RiskMetrics metrics;
    metrics.volatility = calculate_volatility(price_data);
    metrics.max_drawdown = calculate_max_drawdown(price_data);
    metrics.sharpe_ratio = calculate_sharpe_ratio(price_data);
    metrics.position_size = current_position_size;
    metrics.risk_level = determine_risk_level(metrics);
    
    return metrics;
}

RiskLevel RiskManager::determine_risk_level(const RiskMetrics& metrics) {
    if (metrics.volatility < 0.01 && metrics.max_drawdown < 0.05) {
        return RiskLevel::VERY_LOW;
    } else if (metrics.volatility < 0.02 && metrics.max_drawdown < 0.1) {
        return RiskLevel::LOW;
    } else if (metrics.volatility < 0.03 && metrics.max_drawdown < 0.15) {
        return RiskLevel::MEDIUM;
    } else if (metrics.volatility < 0.04 && metrics.max_drawdown < 0.2) {
        return RiskLevel::HIGH;
    } else {
        return RiskLevel::VERY_HIGH;
    }
}

double RiskManager::calculate_position_size(
    const RiskMetrics& metrics,
    double account_value,
    double confidence) {
    
    double base_size = max_position_size_ * account_value;
    
    // Adjust based on risk level
    double risk_multiplier = 1.0;
    switch (metrics.risk_level) {
        case RiskLevel::VERY_LOW: risk_multiplier = 1.0; break;
        case RiskLevel::LOW: risk_multiplier = 0.8; break;
        case RiskLevel::MEDIUM: risk_multiplier = 0.6; break;
        case RiskLevel::HIGH: risk_multiplier = 0.4; break;
        case RiskLevel::VERY_HIGH: risk_multiplier = 0.2; break;
    }
    
    // Adjust based on confidence
    double confidence_multiplier = 0.5 + (confidence * 0.5);
    
    return base_size * risk_multiplier * confidence_multiplier;
}

std::pair<double, double> RiskManager::calculate_stop_loss_take_profit(
    const PriceData& current_price,
    const TechnicalIndicators& indicators,
    const RiskMetrics& metrics) {
    
    double atr = indicators.atr;
    double stop_loss = current_price.price - (2.0 * atr);
    double take_profit = current_price.price + (3.0 * atr);
    
    // Adjust based on risk level
    switch (metrics.risk_level) {
        case RiskLevel::VERY_LOW:
            stop_loss = current_price.price - (1.5 * atr);
            take_profit = current_price.price + (2.0 * atr);
            break;
        case RiskLevel::LOW:
            stop_loss = current_price.price - (2.0 * atr);
            take_profit = current_price.price + (3.0 * atr);
            break;
        case RiskLevel::MEDIUM:
            stop_loss = current_price.price - (2.5 * atr);
            take_profit = current_price.price + (4.0 * atr);
            break;
        case RiskLevel::HIGH:
            stop_loss = current_price.price - (3.0 * atr);
            take_profit = current_price.price + (5.0 * atr);
            break;
        case RiskLevel::VERY_HIGH:
            stop_loss = current_price.price - (3.5 * atr);
            take_profit = current_price.price + (6.0 * atr);
            break;
    }
    
    return std::make_pair(stop_loss, take_profit);
}

// TradingModel Implementation
TradingModel::TradingModel()
    : technical_analyzer_(std::make_unique<TechnicalAnalyzer>()),
      market_analyzer_(std::make_unique<MarketAnalyzer>()),
      risk_manager_(std::make_unique<RiskManager>()) {}

void TradingModel::set_risk_parameters(double max_position_size, double max_drawdown) {
    risk_manager_ = std::make_unique<RiskManager>(max_position_size, max_drawdown);
}

void TradingModel::set_technical_parameters(int sma_period, int ema_period, int rsi_period) {
    technical_analyzer_ = std::make_unique<TechnicalAnalyzer>(sma_period, ema_period, rsi_period);
}

void TradingModel::set_sentiment_weights(double social_weight, double analyst_weight, double news_weight) {
    // Normalize weights
    double total = social_weight + analyst_weight + news_weight;
    social_weight /= total;
    analyst_weight /= total;
    news_weight /= total;
}

double TradingModel::calculate_decision_confidence(
    const TechnicalIndicators& indicators,
    const SentimentData& sentiment,
    const MarketContext& context,
    const RiskMetrics& risk) {
    
    double technical_score = 0.0;
    // RSI scoring
    if (indicators.rsi_14 < 30) technical_score += 0.3;
    else if (indicators.rsi_14 > 70) technical_score -= 0.3;
    
    // MACD scoring
    if (indicators.macd > indicators.macd_signal) technical_score += 0.2;
    else technical_score -= 0.2;
    
    // Bollinger Bands scoring
    double bb_position = (indicators.upper_band - indicators.lower_band) / indicators.middle_band;
    if (bb_position < 0.1) technical_score += 0.2;
    else if (bb_position > 0.3) technical_score -= 0.2;
    
    // Sentiment scoring
    double sentiment_score = (sentiment.social_sentiment + 
                            sentiment.analyst_sentiment + 
                            sentiment.news_sentiment) / 3.0;
    
    // Market context scoring
    double context_score = 0.0;
    switch (context.regime) {
        case MarketRegime::TRENDING_UP: context_score = 0.3; break;
        case MarketRegime::TRENDING_DOWN: context_score = -0.3; break;
        case MarketRegime::RANGING: context_score = 0.0; break;
        case MarketRegime::VOLATILE: context_score = -0.2; break;
        case MarketRegime::UNKNOWN: context_score = 0.0; break;
    }
    
    // Risk scoring
    double risk_score = 0.0;
    switch (risk.risk_level) {
        case RiskLevel::VERY_LOW: risk_score = 0.2; break;
        case RiskLevel::LOW: risk_score = 0.1; break;
        case RiskLevel::MEDIUM: risk_score = 0.0; break;
        case RiskLevel::HIGH: risk_score = -0.1; break;
        case RiskLevel::VERY_HIGH: risk_score = -0.2; break;
    }
    
    // Combine all scores with weights
    double final_score = (technical_score * technical_weight_) +
                        (sentiment_score * sentiment_weight_) +
                        (context_score * market_context_weight_) +
                        (risk_score * risk_weight_);
    
    // Normalize to 0-1 range
    return (final_score + 1.0) / 2.0;
}

TradeSignal TradingModel::determine_signal(
    double confidence,
    const TechnicalIndicators& indicators,
    const MarketContext& context) {
    
    // Strong signals require high confidence
    if (confidence > 0.8) {
        if (indicators.rsi_14 < 30 && indicators.macd > indicators.macd_signal) {
            return TradeSignal::STRONG_BUY;
        }
        if (indicators.rsi_14 > 70 && indicators.macd < indicators.macd_signal) {
            return TradeSignal::STRONG_SELL;
        }
    }
    
    // Weak signals with moderate confidence
    if (confidence > 0.6) {
        if (indicators.rsi_14 < 40 && indicators.macd > indicators.macd_signal) {
            return TradeSignal::WEAK_BUY;
        }
        if (indicators.rsi_14 > 60 && indicators.macd < indicators.macd_signal) {
            return TradeSignal::WEAK_SELL;
        }
    }
    
    // Position management signals
    if (confidence > 0.7) {
        if (context.regime == MarketRegime::TRENDING_UP) {
            return TradeSignal::INCREASE_POSITION;
        }
        if (context.regime == MarketRegime::TRENDING_DOWN) {
            return TradeSignal::REDUCE_POSITION;
        }
    }
    
    // Options signals in volatile markets
    if (context.regime == MarketRegime::VOLATILE && confidence > 0.75) {
        if (indicators.rsi_14 < 30) {
            return TradeSignal::OPTIONS_BUY;
        }
        if (indicators.rsi_14 > 70) {
            return TradeSignal::OPTIONS_SELL;
        }
    }
    
    return TradeSignal::HOLD;
}

std::string TradingModel::generate_reasoning(
    const TradeSignal& signal,
    const TechnicalIndicators& indicators,
    const SentimentData& sentiment,
    const MarketContext& context,
    const RiskMetrics& risk) {
    
    std::string reasoning;
    
    // Add technical analysis reasoning
    reasoning += "Technical Analysis: ";
    if (indicators.rsi_14 < 30) reasoning += "Oversold (RSI: " + std::to_string(indicators.rsi_14) + "). ";
    else if (indicators.rsi_14 > 70) reasoning += "Overbought (RSI: " + std::to_string(indicators.rsi_14) + "). ";
    
    if (indicators.macd > indicators.macd_signal) reasoning += "MACD bullish. ";
    else reasoning += "MACD bearish. ";
    
    // Add sentiment reasoning
    reasoning += "Sentiment: ";
    if (sentiment.overall_sentiment > 0.6) reasoning += "Strongly positive. ";
    else if (sentiment.overall_sentiment > 0.4) reasoning += "Moderately positive. ";
    else if (sentiment.overall_sentiment < -0.6) reasoning += "Strongly negative. ";
    else if (sentiment.overall_sentiment < -0.4) reasoning += "Moderately negative. ";
    else reasoning += "Neutral. ";
    
    // Add market context reasoning
    reasoning += "Market Context: ";
    switch (context.regime) {
        case MarketRegime::TRENDING_UP: reasoning += "Strong uptrend. "; break;
        case MarketRegime::TRENDING_DOWN: reasoning += "Strong downtrend. "; break;
        case MarketRegime::RANGING: reasoning += "Sideways market. "; break;
        case MarketRegime::VOLATILE: reasoning += "High volatility. "; break;
        case MarketRegime::UNKNOWN: reasoning += "Unclear market regime. "; break;
    }
    
    // Add risk assessment
    reasoning += "Risk Level: ";
    switch (risk.risk_level) {
        case RiskLevel::VERY_LOW: reasoning += "Very low risk. "; break;
        case RiskLevel::LOW: reasoning += "Low risk. "; break;
        case RiskLevel::MEDIUM: reasoning += "Medium risk. "; break;
        case RiskLevel::HIGH: reasoning += "High risk. "; break;
        case RiskLevel::VERY_HIGH: reasoning += "Very high risk. "; break;
    }
    
    return reasoning;
}

TradingDecision TradingModel::get_trading_decision(
    const std::string& symbol,
    const std::vector<PriceData>& price_data,
    const SentimentData& sentiment,
    const std::vector<double>& sector_data,
    double vix,
    double account_value,
    double current_position_size) {
    
    // Calculate all indicators and metrics
    TechnicalIndicators indicators = technical_analyzer_->calculate_indicators(price_data);
    MarketContext context = market_analyzer_->analyze_market_context(price_data, sector_data, vix);
    RiskMetrics risk = risk_manager_->calculate_risk_metrics(price_data, current_position_size);
    
    // Calculate decision confidence
    double confidence = calculate_decision_confidence(indicators, sentiment, context, risk);
    
    // Determine trading signal
    TradeSignal signal = determine_signal(confidence, indicators, context);
    
    // Calculate position size and stop levels
    double position_size = risk_manager_->calculate_position_size(risk, account_value, confidence);
    auto [stop_loss, take_profit] = risk_manager_->calculate_stop_loss_take_profit(
        price_data.back(), indicators, risk);
    
    // Generate reasoning
    std::string reasoning = generate_reasoning(signal, indicators, sentiment, context, risk);
    
    // Log the decision
    std::cout << "[C++ Engine] Analyzing " << symbol << std::endl;
    std::cout << "[C++ Engine] Signal: " << static_cast<int>(signal) << std::endl;
    std::cout << "[C++ Engine] Confidence: " << confidence << std::endl;
    std::cout << "[C++ Engine] Position Size: " << position_size << std::endl;
    std::cout << "[C++ Engine] Stop Loss: " << stop_loss << std::endl;
    std::cout << "[C++ Engine] Take Profit: " << take_profit << std::endl;
    std::cout << "[C++ Engine] Reasoning: " << reasoning << std::endl;
    
    // Return the complete trading decision
    return TradingDecision{
        signal,
        confidence,
        position_size,
        stop_loss,
        take_profit,
        reasoning
    };
}
