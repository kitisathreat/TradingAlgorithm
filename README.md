# TradingAlgorithm: High-Performance Hybrid Trading System

## Overview

**TradingAlgorithm** is a hybrid algorithmic trading platform that combines the speed and efficiency of C++ with the flexibility and intelligence of Python. The system is designed for both research and live trading, leveraging advanced technical analysis, machine learning, and real-time data processing. The architecture enables rapid decision-making for live markets while supporting robust model training and interactive human-in-the-loop workflows.

---

## Repository Structure

```
TradingAlgorithm/
├── 1_High_Performance_Module_(C++)/         # C++ core: high-speed trading engine, pybind11 bindings
├── 2_Orchestrator_And_ML_(Python)/          # Python orchestration, ML, deployment, and UI
│   ├── deployment_services/                 # FastAPI and related deployment services
│   ├── interactive_training_app/            # Interactive training app (backend/frontend)
│   ├── tests/                               # Integration and unit tests
├── 3_Networking_and_User_Input/             # (Reserved for networking/user input modules)
├── README.md                                # Project documentation
├── TradingAlgo.sln                          # Solution file for C++ build
```

---

## Main Components

### 1. High-Performance Module (C++)
- **Purpose:** Implements the core trading decision engine for ultra-fast, low-latency calculations.
- **Key Features:**
  - Technical analysis (SMA, EMA, RSI, etc.)
  - Market regime detection
  - Risk management logic
  - Exposed to Python via `pybind11` bindings
- **Implementation:**
  - `engine.cpp`/`engine.h`: Core trading logic, data structures, and algorithms
  - `bindings.cpp`: Exposes C++ classes and functions to Python, with robust type safety and error handling
  - `CMakeLists.txt`: Build configuration for compiling the C++ module as a Python extension

### 2. Orchestrator and Machine Learning (Python)
- **Purpose:** Coordinates data flow, model training, live trading, and user interaction.
- **Key Features:**
  - Loads and preprocesses historical and live market data
  - Trains neural networks for pattern recognition and signal generation
  - Integrates with the C++ engine for real-time trading decisions
  - Manages risk, order execution, and logging
- **Implementation:**
  - `live_trader.py`: Main trading bot, integrates C++ engine, handles trading loop, error handling, and logging
  - `train_model.py`: Model training pipeline for ML components
  - `market_analyzer.py`: Market data analysis and feature extraction
  - `tests/`: Comprehensive integration/unit tests for C++/Python boundary and trading logic

### 3. Deployment Services & Interactive Training
- **Purpose:** Provides APIs and user interfaces for monitoring, training, and human-in-the-loop workflows.
- **Key Features:**
  - FastAPI server for facial sentiment analysis and logging
  - Interactive web app for training and feedback
- **Implementation:**
  - `deployment_services/`: FastAPI and related deployment scripts
  - `interactive_training_app/`: Full-stack app (backend: Python, frontend: JS/React or similar)

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- C++17 compiler (GCC, Clang, or MSVC)
- [pybind11](https://github.com/pybind/pybind11)
- Required Python packages (see below)

### Building the C++ Module
1. Install `pybind11` and a compatible compiler.
2. From the `1_High_Performance_Module_(C++)` directory, build the Python extension:
   ```sh
   mkdir build && cd build
   cmake ..
   make
   ```
3. Ensure the resulting module (e.g., `decision_engine.*.pyd`/`.so`) is accessible to the Python code (copy or symlink as needed).

### Python Dependencies
Install required packages:
```sh
pip install -r 2_Orchestrator_And_ML_(Python)/requirements.txt
```

### Running the System
1. **Prepare Data:** Download and preprocess historical market data as needed.
2. **Start Deployment Services:**
   ```sh
   cd 2_Orchestrator_And_ML_(Python)/deployment_services
   uvicorn pi_server:app --reload
   ```
3. **Interactive Training (optional):**
   ```sh
   cd 2_Orchestrator_And_ML_(Python)/interactive_training_app
   # Start backend and frontend as per app instructions
   ```
4. **Train Model:**
   ```sh
   python 2_Orchestrator_And_ML_(Python)/train_model.py
   ```
5. **Run Live Trading Bot:**
   ```sh
   python 2_Orchestrator_And_ML_(Python)/live_trader.py
   ```

---

## Key Implementation Details

- **C++/Python Integration:**
  - The C++ engine is exposed to Python using `pybind11`, with custom exception handling and type-safe data conversion.
  - Python code calls the C++ engine for every trading decision, passing validated data structures.
- **Error Handling:**
  - Both C++ and Python layers include robust error handling and logging.
  - Custom exceptions are raised for invalid input, and all errors are logged for traceability.
- **Testing:**
  - Integration and unit tests are provided in `2_Orchestrator_And_ML_(Python)/tests/`.
  - Tests cover C++/Python boundary, trading logic, and error scenarios.
- **Extensibility:**
  - Modular design allows for easy extension of technical indicators, ML models, and deployment services.

---

## Need-to-Know Topics

- **API Keys & Secrets:**
  - Store sensitive credentials in environment variables or a `.env` file (never commit secrets to version control).
- **Performance:**
  - The C++ module is designed for low-latency, high-throughput environments. Ensure it is compiled with optimizations enabled.
- **Customization:**
  - You can add new technical indicators, ML models, or data sources by extending the relevant modules.
- **Error Logs:**
  - All trading and system errors are logged to `trading_bot.log` for debugging and audit purposes.
- **Testing:**
  - Run `pytest` in the `tests/` directory to verify integration and logic before deploying to production.

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes, improvements, or new features.

---

## License

This project is licensed under the MIT License.