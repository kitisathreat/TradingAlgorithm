#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "engine.h"

namespace py = pybind11;

// Helper function to convert Python list to vector of PriceData
std::vector<PriceData> list_to_price_data(const py::list& data) {
    std::vector<PriceData> result;
    for (const auto& item : data) {
        py::dict price_dict = item.cast<py::dict>();
        PriceData price;
        price.price = price_dict["price"].cast<double>();
        price.volume = price_dict["volume"].cast<double>();
        price.timestamp = price_dict["timestamp"].cast<double>();
        result.push_back(price);
    }
    return result;
}

// Helper function to convert Python list to vector of doubles
std::vector<double> list_to_doubles(const py::list& data) {
    std::vector<double> result;
    for (const auto& item : data) {
        result.push_back(item.cast<double>());
    }
    return result;
}

PYBIND11_MODULE(decision_engine, m) {
    m.doc() = "A high-performance C++ decision engine for trading with advanced technical analysis, market context, and risk management.";

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
        .def_readwrite("price", &PriceData::price)
        .def_readwrite("volume", &PriceData::volume)
        .def_readwrite("timestamp", &PriceData::timestamp);

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
                                      double current_position_size) {
            return self.get_trading_decision(
                symbol,
                list_to_price_data(price_data),
                sentiment,
                list_to_doubles(sector_data),
                vix,
                account_value,
                current_position_size
            );
        }, "Makes a trading decision based on comprehensive market analysis")
        .def("set_risk_parameters", &TradingModel::set_risk_parameters,
             "Set risk management parameters")
        .def("set_technical_parameters", &TradingModel::set_technical_parameters,
             "Set technical analysis parameters")
        .def("set_sentiment_weights", &TradingModel::set_sentiment_weights,
             "Set weights for different sentiment sources");

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
