#pragma once

#include <string>
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>
#include <unordered_map>

// Market regime types
enum class MarketRegime {
    TRENDING_UP,
    TRENDING_DOWN,
    RANGING,
    VOLATILE,
    UNKNOWN
};

// More granular trading signals
enum class TradeSignal {
    STRONG_BUY,
    WEAK_BUY,
    HOLD,
    REDUCE_POSITION,
    INCREASE_POSITION,
    WEAK_SELL,
    STRONG_SELL,
    OPTIONS_BUY,
    OPTIONS_SELL
};

// Risk levels for position sizing
enum class RiskLevel {
    VERY_LOW,
    LOW,
    MEDIUM,
    HIGH,
    VERY_HIGH
};

// Structure to hold price data
struct PriceData {
    double price;
    double volume;
    double timestamp;
};

// Structure to hold technical indicators
struct TechnicalIndicators {
    double sma_20;
    double sma_50;
    double ema_20;
    double rsi_14;
    double macd;
    double macd_signal;
    double macd_histogram;
    double upper_band;
    double lower_band;
    double middle_band;
    double atr;
};

// Structure to hold market context
struct MarketContext {
    MarketRegime regime;
    double sector_correlation;
    double market_breadth;
    double vix;
    double sector_performance;
};

// Structure to hold risk metrics
struct RiskMetrics {
    double volatility;
    double max_drawdown;
    double sharpe_ratio;
    double position_size;
    RiskLevel risk_level;
};

// Structure to hold sentiment data
struct SentimentData {
    double social_sentiment;
    double analyst_sentiment;
    double news_sentiment;
    double overall_sentiment;
};

// Structure to hold neural network insights
struct NeuralNetworkInsights {
    double predicted_price_change;
    double trend_strength;
    MarketRegime predicted_regime;
    double confidence;
};

// Structure to hold the final trading decision
struct TradingDecision {
    TradeSignal signal;
    double confidence;
    double suggested_position_size;
    double stop_loss;
    double take_profit;
    std::string reasoning;
};

// Technical Analysis class
class TechnicalAnalyzer {
public:
    TechnicalAnalyzer(int sma_period = 20, int ema_period = 20, 
                     int rsi_period = 14, int macd_fast = 12, 
                     int macd_slow = 26, int macd_signal = 9,
                     int bb_period = 20, double bb_std = 2.0);

    TechnicalIndicators calculate_indicators(const std::vector<PriceData>& price_data);
    double calculate_sma(const std::vector<PriceData>& data, int period);
    double calculate_ema(const std::vector<PriceData>& data, int period);
    double calculate_rsi(const std::vector<PriceData>& data, int period);
    std::tuple<double, double, double> calculate_macd(const std::vector<PriceData>& data);
    std::tuple<double, double, double> calculate_bollinger_bands(const std::vector<PriceData>& data);
    double calculate_atr(const std::vector<PriceData>& data, int period = 14);

private:
    int sma_period_;
    int ema_period_;
    int rsi_period_;
    int macd_fast_;
    int macd_slow_;
    int macd_signal_;
    int bb_period_;
    double bb_std_;
};

// Market Analysis class
class MarketAnalyzer {
public:
    MarketAnalyzer();
    MarketContext analyze_market_context(const std::vector<PriceData>& price_data,
                                       const std::vector<double>& sector_data,
                                       double vix);
    MarketRegime detect_market_regime(const std::vector<PriceData>& price_data);
    double calculate_market_breadth(const std::vector<double>& sector_performance);
    double calculate_sector_correlation(const std::vector<PriceData>& price_data,
                                      const std::vector<double>& sector_data);
};

// Risk Manager class
class RiskManager {
public:
    RiskManager(double max_position_size = 1.0, double max_drawdown = 0.1);
    RiskMetrics calculate_risk_metrics(const std::vector<PriceData>& price_data,
                                     double current_position_size);
    double calculate_position_size(const RiskMetrics& metrics,
                                 double account_value,
                                 double confidence);
    std::pair<double, double> calculate_stop_loss_take_profit(const PriceData& current_price,
                                                            const TechnicalIndicators& indicators,
                                                            const RiskMetrics& metrics);
    RiskLevel determine_risk_level(const RiskMetrics& metrics);

private:
    double max_position_size_;
    double max_drawdown_;
    double calculate_volatility(const std::vector<PriceData>& price_data);
    double calculate_max_drawdown(const std::vector<PriceData>& price_data);
    double calculate_sharpe_ratio(const std::vector<PriceData>& price_data);
};

// Main Trading Model class
class TradingModel {
public:
    TradingModel();
    
    // Main decision making function with enhanced parameters
    TradingDecision get_trading_decision(
        const std::string& symbol,
        const std::vector<PriceData>& price_data,
        const SentimentData& sentiment,
        const std::vector<double>& sector_data,
        double vix,
        double account_value,
        double current_position_size,
        const NeuralNetworkInsights& nn_insights = NeuralNetworkInsights{}  // Optional parameter
    );

    // Configuration methods
    void set_risk_parameters(double max_position_size, double max_drawdown);
    void set_technical_parameters(int sma_period, int ema_period, int rsi_period);
    void set_sentiment_weights(double social_weight, double analyst_weight, double news_weight);
    void set_neural_network_weight(double weight);  // New method to set NN weight

private:
    std::unique_ptr<TechnicalAnalyzer> technical_analyzer_;
    std::unique_ptr<MarketAnalyzer> market_analyzer_;
    std::unique_ptr<RiskManager> risk_manager_;
    
    // Weights for different components
    double technical_weight_ = 0.35;  // Adjusted to make room for NN
    double sentiment_weight_ = 0.25;  // Adjusted
    double market_context_weight_ = 0.20;
    double risk_weight_ = 0.10;
    double neural_network_weight_ = 0.10;  // New weight for NN insights

    // Helper methods
    double calculate_decision_confidence(const TechnicalIndicators& indicators,
                                       const SentimentData& sentiment,
                                       const MarketContext& context,
                                       const RiskMetrics& risk,
                                       const NeuralNetworkInsights& nn_insights);  // Updated
    
    TradeSignal determine_signal(double confidence,
                               const TechnicalIndicators& indicators,
                               const MarketContext& context,
                               const NeuralNetworkInsights& nn_insights);  // Updated
    
    std::string generate_reasoning(const TradeSignal& signal,
                                 const TechnicalIndicators& indicators,
                                 const SentimentData& sentiment,
                                 const MarketContext& context,
                                 const RiskMetrics& risk,
                                 const NeuralNetworkInsights& nn_insights);  // Updated
}; 
