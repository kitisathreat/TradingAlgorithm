#include <pybind11/pybind11.h>
#include "engine.h"

namespace py = pybind11;


PYBIND11_MODULE(decision_engine, m) {
    m.doc() = "A high-performance C++ decision engine for trading.";

    py::enum_<TradeSignal>(m, "TradeSignal")
        .value("HOLD", TradeSignal::HOLD)
        .value("BUY", TradeSignal::BUY)
        .value("SELL", TradeSignal::SELL)
        .export_values();

    py::class_<TradingModel>(m, "TradingModel")
        .def(py::init<>()) // Expose the constructor
        .def("get_trading_decision", &TradingModel::get_trading_decision, "Makes a trade decision based on market data");
}
