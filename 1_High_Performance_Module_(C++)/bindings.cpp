#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "engine.h"

namespace py = pybind11;

// Custom exception class for trading engine errors
class TradingEngineError : public std::runtime_error {
public:
    explicit TradingEngineError(const std::string& msg) : std::runtime_error(msg) {}
};

// Type-safe conversion functions with error handling
std::vector<PriceData> list_to_price_data(const py::list& data) {
    try {
        std::vector<PriceData> result;
        result.reserve(data.size());  // Pre-allocate for better performance
        
        for (const auto& item : data) {
            if (!py::isinstance<py::dict>(item)) {
                throw TradingEngineError("Price data must be a list of dictionaries");
            }
            
            py::dict price_dict = item.cast<py::dict>();
            
            // Validate required keys
            if (!price_dict.contains("price") || !price_dict.contains("volume") || !price_dict.contains("timestamp")) {
                throw TradingEngineError("Price data dictionary must contain 'price', 'volume', and 'timestamp'");
            }
            
            // Type-safe conversion with validation
            PriceData price;
            try {
                price.price = price_dict["price"].cast<double>();
                price.volume = price_dict["volume"].cast<double>();
                price.timestamp = price_dict["timestamp"].cast<double>();
                
                // Validate values
                if (price.price <= 0) throw TradingEngineError("Price must be positive");
                if (price.volume < 0) throw TradingEngineError("Volume cannot be negative");
                if (price.timestamp < 0) throw TradingEngineError("Timestamp must be positive");
                
            } catch (const py::cast_error&) {
                throw TradingEngineError("Invalid data types in price dictionary");
            }
            
            result.push_back(std::move(price));
        }
        return result;
    } catch (const TradingEngineError& e) {
        throw;  // Re-throw our custom exceptions
    } catch (const std::exception& e) {
        throw TradingEngineError(std::string("Error converting price data: ") + e.what());
    }
}

// Type-safe conversion for doubles with validation
std::vector<double> list_to_doubles(const py::list& data) {
    try {
        std::vector<double> result;
        result.reserve(data.size());
        
        for (const auto& item : data) {
            try {
                double value = item.cast<double>();
                if (std::isnan(value) || std::isinf(value)) {
                    throw TradingEngineError("Invalid numeric value in data");
                }
                result.push_back(value);
            } catch (const py::cast_error&) {
                throw TradingEngineError("All values must be numeric");
            }
        }
        return result;
    } catch (const TradingEngineError& e) {
        throw;
    } catch (const std::exception& e) {
        throw TradingEngineError(std::string("Error converting numeric data: ") + e.what());
    }
}

PYBIND11_MODULE(decision_engine, m) {
    m.doc() = "A high-performance C++ decision engine for trading with advanced technical analysis, market context, and risk management.";

    // Register custom exception
    py::register_exception<TradingEngineError>(m, "TradingEngineError");

    // Enums
    py::enum_<MarketRegime>(m, "MarketRegime")
        .value("TRENDING_UP", MarketRegime::TRENDING_UP)
        .value("TRENDING_DOWN", MarketRegime::TRENDING_DOWN)
        .value("RANGING", MarketRegime::RANGING)
        .value("VOLATILE", MarketRegime::VOLATILE)
        .value("UNKNOWN", MarketRegime::UNKNOWN)
        .export_values();

    py::enum_<TradeSignal>(m, "TradeSignal")
        .value("STRONG_BUY", TradeSignal::STRONG_BUY)
        .value("WEAK_BUY", TradeSignal::WEAK_BUY)
        .value("HOLD", TradeSignal::HOLD)
        .value("REDUCE_POSITION", TradeSignal::REDUCE_POSITION)
        .value("INCREASE_POSITION", TradeSignal::INCREASE_POSITION)
        .value("WEAK_SELL", TradeSignal::WEAK_SELL)
        .value("STRONG_SELL", TradeSignal::STRONG_SELL)
        .value("OPTIONS_BUY", TradeSignal::OPTIONS_BUY)
        .value("OPTIONS_SELL", TradeSignal::OPTIONS_SELL)
        .export_values();

    py::enum_<RiskLevel>(m, "RiskLevel")
        .value("VERY_LOW", RiskLevel::VERY_LOW)
        .value("LOW", RiskLevel::LOW)
        .value("MEDIUM", RiskLevel::MEDIUM)
        .value("HIGH", RiskLevel::HIGH)
        .value("VERY_HIGH", RiskLevel::VERY_HIGH)
        .export_values();

    // Structs
    py::class_<PriceData>(m, "PriceData")
        .def(py::init<>())
        .def(py::init([](double price, double volume, double timestamp) {
            if (price <= 0) throw TradingEngineError("Price must be positive");
            if (volume < 0) throw TradingEngineError("Volume cannot be negative");
            if (timestamp < 0) throw TradingEngineError("Timestamp must be positive");
            return PriceData{price, volume, timestamp};
        }), py::arg("price"), py::arg("volume"), py::arg("timestamp"),
            "Initialize PriceData with validation")
        .def_readwrite("price", &PriceData::price)
        .def_readwrite("volume", &PriceData::volume)
        .def_readwrite("timestamp", &PriceData::timestamp)
        .def("__repr__", [](const PriceData& p) {
            return "PriceData(price=" + std::to_string(p.price) + 
                   ", volume=" + std::to_string(p.volume) + 
                   ", timestamp=" + std::to_string(p.timestamp) + ")";
        });

    py::class_<TechnicalIndicators>(m, "TechnicalIndicators")
        .def(py::init<>())
        .def_readwrite("sma_20", &TechnicalIndicators::sma_20)
        .def_readwrite("sma_50", &TechnicalIndicators::sma_50)
        .def_readwrite("ema_20", &TechnicalIndicators::ema_20)
        .def_readwrite("rsi_14", &TechnicalIndicators::rsi_14)
        .def_readwrite("macd", &TechnicalIndicators::macd)
        .def_readwrite("macd_signal", &TechnicalIndicators::macd_signal)
        .def_readwrite("macd_histogram", &TechnicalIndicators::macd_histogram)
        .def_readwrite("upper_band", &TechnicalIndicators::upper_band)
        .def_readwrite("lower_band", &TechnicalIndicators::lower_band)
        .def_readwrite("middle_band", &TechnicalIndicators::middle_band)
        .def_readwrite("atr", &TechnicalIndicators::atr);

    py::class_<MarketContext>(m, "MarketContext")
        .def(py::init<>())
        .def_readwrite("regime", &MarketContext::regime)
        .def_readwrite("sector_correlation", &MarketContext::sector_correlation)
        .def_readwrite("market_breadth", &MarketContext::market_breadth)
        .def_readwrite("vix", &MarketContext::vix)
        .def_readwrite("sector_performance", &MarketContext::sector_performance);

    py::class_<RiskMetrics>(m, "RiskMetrics")
        .def(py::init<>())
        .def_readwrite("volatility", &RiskMetrics::volatility)
        .def_readwrite("max_drawdown", &RiskMetrics::max_drawdown)
        .def_readwrite("sharpe_ratio", &RiskMetrics::sharpe_ratio)
        .def_readwrite("position_size", &RiskMetrics::position_size)
        .def_readwrite("risk_level", &RiskMetrics::risk_level);

    py::class_<SentimentData>(m, "SentimentData")
        .def(py::init<>())
        .def_readwrite("social_sentiment", &SentimentData::social_sentiment)
        .def_readwrite("analyst_sentiment", &SentimentData::analyst_sentiment)
        .def_readwrite("news_sentiment", &SentimentData::news_sentiment)
        .def_readwrite("overall_sentiment", &SentimentData::overall_sentiment);

    py::class_<TradingDecision>(m, "TradingDecision")
        .def(py::init<>())
        .def_readwrite("signal", &TradingDecision::signal)
        .def_readwrite("confidence", &TradingDecision::confidence)
        .def_readwrite("suggested_position_size", &TradingDecision::suggested_position_size)
        .def_readwrite("stop_loss", &TradingDecision::stop_loss)
        .def_readwrite("take_profit", &TradingDecision::take_profit)
        .def_readwrite("reasoning", &TradingDecision::reasoning);

    // Add NeuralNetworkInsights struct
    py::class_<NeuralNetworkInsights>(m, "NeuralNetworkInsights")
        .def(py::init<>())
        .def_readwrite("predicted_price_change", &NeuralNetworkInsights::predicted_price_change)
        .def_readwrite("trend_strength", &NeuralNetworkInsights::trend_strength)
        .def_readwrite("predicted_regime", &NeuralNetworkInsights::predicted_regime)
        .def_readwrite("confidence", &NeuralNetworkInsights::confidence);

    // Main TradingModel class
    py::class_<TradingModel>(m, "TradingModel")
        .def(py::init<>())
        .def("get_trading_decision", [](TradingModel& self,
                                      const std::string& symbol,
                                      const py::list& price_data,
                                      const SentimentData& sentiment,
                                      const py::list& sector_data,
                                      double vix,
                                      double account_value,
                                      double current_position_size,
                                      const NeuralNetworkInsights& nn_insights = NeuralNetworkInsights{}) {
            try {
                // Validate inputs
                if (symbol.empty()) throw TradingEngineError("Symbol cannot be empty");
                if (price_data.empty()) throw TradingEngineError("Price data cannot be empty");
                if (account_value <= 0) throw TradingEngineError("Account value must be positive");
                if (current_position_size < 0) throw TradingEngineError("Current position size cannot be negative");
                if (vix < 0) throw TradingEngineError("VIX cannot be negative");

                // Convert and validate data
                auto price_data_vec = list_to_price_data(price_data);
                auto sector_data_vec = list_to_doubles(sector_data);

                // Make the trading decision
                return self.get_trading_decision(
                    symbol,
                    std::move(price_data_vec),
                    sentiment,
                    std::move(sector_data_vec),
                    vix,
                    account_value,
                    current_position_size,
                    nn_insights
                );
            } catch (const TradingEngineError& e) {
                throw;  // Re-throw our custom exceptions
            } catch (const std::exception& e) {
                throw TradingEngineError(std::string("Error in get_trading_decision: ") + e.what());
            }
        }, "Makes a trading decision based on comprehensive market analysis and neural network insights",
           py::arg("symbol"),
           py::arg("price_data"),
           py::arg("sentiment"),
           py::arg("sector_data"),
           py::arg("vix"),
           py::arg("account_value"),
           py::arg("current_position_size"),
           py::arg("nn_insights") = NeuralNetworkInsights{})
        .def("set_risk_parameters", [](TradingModel& self, double max_position_size, double max_drawdown) {
            if (max_position_size <= 0 || max_position_size > 1.0) {
                throw TradingEngineError("Max position size must be between 0 and 1");
            }
            if (max_drawdown <= 0 || max_drawdown > 1.0) {
                throw TradingEngineError("Max drawdown must be between 0 and 1");
            }
            self.set_risk_parameters(max_position_size, max_drawdown);
        }, "Set risk management parameters")
        .def("set_technical_parameters", [](TradingModel& self, int sma_period, int ema_period, int rsi_period) {
            if (sma_period <= 0 || ema_period <= 0 || rsi_period <= 0) {
                throw TradingEngineError("Technical indicator periods must be positive");
            }
            self.set_technical_parameters(sma_period, ema_period, rsi_period);
        }, "Set technical analysis parameters")
        .def("set_sentiment_weights", [](TradingModel& self, double social_weight, double analyst_weight, double news_weight) {
            if (social_weight < 0 || analyst_weight < 0 || news_weight < 0) {
                throw TradingEngineError("Sentiment weights cannot be negative");
            }
            double total = social_weight + analyst_weight + news_weight;
            if (std::abs(total - 1.0) > 1e-6) {
                throw TradingEngineError("Sentiment weights must sum to 1.0");
            }
            self.set_sentiment_weights(social_weight, analyst_weight, news_weight);
        }, "Set weights for different sentiment sources")
        .def("set_neural_network_weight", [](TradingModel& self, double weight) {
            if (weight < 0 || weight > 1.0) {
                throw TradingEngineError("Neural network weight must be between 0 and 1");
            }
            self.set_neural_network_weight(weight);
        }, "Set the weight for neural network insights in decision making");

    // TechnicalAnalyzer class
    py::class_<TechnicalAnalyzer>(m, "TechnicalAnalyzer")
        .def(py::init<int, int, int, int, int, int, int, double>(),
             py::arg("sma_period") = 20,
             py::arg("ema_period") = 20,
             py::arg("rsi_period") = 14,
             py::arg("macd_fast") = 12,
             py::arg("macd_slow") = 26,
             py::arg("macd_signal") = 9,
             py::arg("bb_period") = 20,
             py::arg("bb_std") = 2.0)
        .def("calculate_indicators", [](TechnicalAnalyzer& self, const py::list& price_data) {
            return self.calculate_indicators(list_to_price_data(price_data));
        }, "Calculate all technical indicators");

    // MarketAnalyzer class
    py::class_<MarketAnalyzer>(m, "MarketAnalyzer")
        .def(py::init<>())
        .def("analyze_market_context", [](MarketAnalyzer& self,
                                        const py::list& price_data,
                                        const py::list& sector_data,
                                        double vix) {
            return self.analyze_market_context(
                list_to_price_data(price_data),
                list_to_doubles(sector_data),
                vix
            );
        }, "Analyze market context and regime");

    // RiskManager class
    py::class_<RiskManager>(m, "RiskManager")
        .def(py::init<double, double>(),
             py::arg("max_position_size") = 1.0,
             py::arg("max_drawdown") = 0.1)
        .def("calculate_risk_metrics", [](RiskManager& self,
                                        const py::list& price_data,
                                        double current_position_size) {
            return self.calculate_risk_metrics(
                list_to_price_data(price_data),
                current_position_size
            );
        }, "Calculate risk metrics")
        .def("calculate_position_size", &RiskManager::calculate_position_size,
             "Calculate suggested position size")
        .def("calculate_stop_loss_take_profit", &RiskManager::calculate_stop_loss_take_profit,
             "Calculate stop loss and take profit levels");
}
