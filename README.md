# TradingAlgorithm: Hybrid Trading System with ML Integration

## Overview

**TradingAlgorithm** is a sophisticated trading platform that combines high-performance C++ for core trading logic with Python for machine learning and orchestration. The system is designed to be both powerful for experienced developers and accessible for those new to algorithmic trading.

### Key Features
- High-speed trading engine in C++ for real-time market analysis
- Machine learning integration for pattern recognition and prediction
- Interactive web interface for monitoring and control
- Comprehensive logging and error handling
- Modular design for easy extension and customization

## Project Structure

```
TradingAlgorithm/
├── 0_Launchers/                        # Quick-start scripts and launchers
├── 1_High_Performance_Module_(C++)/    # Core trading engine (C++)
│   ├── src/                           # Source code
│   └── tests/                         # C++ unit tests
├── 2_Orchestrator_And_ML_Python/       # Python orchestration and ML
│   ├── logs/                          # Trading and system logs
│   ├── tests/                         # Python tests
│   ├── main.py                        # Main entry point
│   └── requirements.txt               # Python dependencies
├── 3_Networking_and_User_Input/        # Web interface and API
│   └── web_interface/                 # Streamlit-based UI
├── docs/                              # Documentation
└── .github/                           # CI/CD workflows
```

## Getting Started

### For Beginners

1. **Setup Environment**
   - Install Python 3.9 (required for compatibility)
   - Run the setup script in `0_Launchers/setup_environment.bat`
   - This will create a virtual environment and install all dependencies

2. **Quick Start**
   - Run `0_Launchers/start_trading.bat` to launch the system
   - Access the web interface at `http://localhost:8501`
   - Monitor trading activity in the logs directory

3. **Key Files to Understand**
   - `2_Orchestrator_And_ML_Python/main.py`: Main trading logic
   - `3_Networking_and_User_Input/web_interface/`: Web dashboard
   - `docs/project_structure.txt`: Detailed component documentation

### For Developers

1. **Development Setup**
   ```powershell
   # Create and activate virtual environment
   python -m venv venv
   .\venv\Scripts\activate

   # Install dependencies
   pip install -r 2_Orchestrator_And_ML_Python/requirements.txt
   ```

2. **Building C++ Module**
   ```powershell
   cd 1_High_Performance_Module_(C++)
   mkdir build
   cd build
   cmake ..
   cmake --build . --config Release
   ```

3. **Running Tests**
   ```powershell
   # Python tests
   cd 2_Orchestrator_And_ML_Python
   python -m pytest tests/

   # C++ tests
   cd 1_High_Performance_Module_(C++)
   cd build
   ctest
   ```

## Component Details

### 1. High-Performance Module (C++)
- **Purpose**: Ultra-fast market analysis and trading decisions
- **Key Components**:
  - Technical analysis indicators (SMA, EMA, RSI)
  - Market regime detection
  - Risk management algorithms
- **Integration**: Exposed to Python via pybind11 bindings

### 2. Python Orchestrator
- **Purpose**: Coordinates trading operations and ML integration
- **Key Features**:
  - Market data processing
  - ML model training and inference
  - Trading strategy execution
  - Comprehensive logging
- **Main Files**:
  - `main.py`: Core trading logic
  - `market_analyzer.py`: Market data analysis
  - `model_trainer.py`: ML model training

### 3. Web Interface
- **Purpose**: User-friendly monitoring and control
- **Features**:
  - Real-time trading dashboard
  - Performance metrics
  - Strategy configuration
  - Log viewer

## Development Guidelines

### Adding New Features
1. **Technical Indicators**
   - Add C++ implementation in `1_High_Performance_Module_(C++)/src/`
   - Create Python bindings in `bindings.cpp`
   - Add tests in respective test directories

2. **ML Models**
   - Implement in `2_Orchestrator_And_ML_Python/`
   - Follow existing model architecture
   - Add validation tests

3. **UI Components**
   - Add to `3_Networking_and_User_Input/web_interface/`
   - Follow Streamlit best practices
   - Include error handling

### Best Practices
- Use type hints in Python code
- Follow C++17 standards
- Write tests for new features
- Update documentation
- Use environment variables for secrets
- Follow existing logging patterns

## Troubleshooting

Common issues and solutions are documented in:
- `docs/streamlit_cloud_troubleshooting.md`
- `docs/setup_guide.md`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Run tests
5. Submit pull request

## License

MIT License - See LICENSE file for details