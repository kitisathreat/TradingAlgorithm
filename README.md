# TradingAlgo
# Hybrid C++/Python Trading Bot with ML Decision Engine

This repository contains the complete source code for a sophisticated, multi-component algorithmic trading system. The project is designed to learn an investor's decision-making process through an interactive training session that includes computer vision, train a machine learning model based on that expert data, and then deploy that model for live, real-time trading.

The architecture is distributed, leveraging different hardware and technologies for their specific strengths: a low-power Raspberry Pi for orchestration, a powerful desktop for intensive computer vision tasks, and a main development environment for logic and model training.

### Core Features

* **Hybrid Performance:** Combines Python for rapid development and API integration with a C++ module for high-performance, rule-based logic.
* **Machine Learning Core:** Uses `scikit-learn` to train a RandomForest model that learns and replicates an expert investor's trading style.
* **Interactive Training:** A web-based interface allows an expert investor to make decisions on historical data, creating a high-quality training dataset.
* **Multi-Modal Data Capture:** The training session captures not only the investor's explicit "BUY/SELL/HOLD" choice but also their implicit facial sentiment via computer vision (`deepface`).
* **Distributed Architecture:** Optimized for real-world deployment with three distinct components:
    1.  **Orchestration Server:** A lightweight FastAPI server designed to run perpetually on a Raspberry Pi.
    2.  **Vision Service:** An offloaded FastAPI service to run on a powerful PC for computationally heavy facial analysis.
    3.  **Live Trader & Trainer:** The main scripts for model training and live execution.
* **Pluggable Data Sources:** The system is designed to use premium news sources like Polygon.io, with a graceful fallback to the free Alpaca News API.

### Tech Stack

* **Backend & ML:** Python, FastAPI, scikit-learn, Pandas, `pybind11`
* **High-Performance Module:** C++
* **Frontend:** HTML, CSS, JavaScript (no framework)
* **Computer Vision:** `deepface`, OpenCV
* **APIs & Services:** Alpaca (Trading & Data), Financial Modeling Prep (Fundamentals), Polygon.io (Premium News)
* **Deployment:** Gunicorn, Nginx, systemd (for Raspberry Pi)
* **Development Environment:** Visual Studio 2022

### Project Structure
TradingBotSolution/
│
├── 1_High_Performance_Module_(C++)/  # C++ Project for rule-based engine
│   ├── engine.h
│   ├── engine.cpp
│   └── bindings.cpp
│
└── 2_Orchestrator_And_ML_(Python)/   # Main Python Project
├── .env                          # For local API keys (DO NOT COMMIT)
├── requirements.txt              # Python dependencies
├── config.py                     # Loads configuration
├── data_fetcher.py               # Fetches data from APIs
├── sentiment_analyzer.py         # Performs sentiment analysis
├── train_model.py                # SCRIPT 1: Trains and saves the ML model
├── live_trader.py                # SCRIPT 2: Runs the live bot
│
└── interactive_training_app/       # Sub-folder for the training tool
├── backend/
│   ├── pi_server.py
│   └── historical_data.csv
└── frontend/
├── index.html
├── style.css
└── script.js

### Setup and Installation

Follow these steps to set up the development environment on your main PC.

#### 1. Prerequisites

* Visual Studio 2022 with the following workloads:
    * **Python development**
    * **Desktop development with C++** (ensure C++ MFC for v143 build tools is included)
* Python 3.9+
* Git

#### 2. Clone the Repository
"```bash"

#### 3. Configure the C++ Module
Open TradingBotSolution.sln in Visual Studio.
Download or clone the pybind11 repository.
In Visual Studio, right-click the 1_High_Performance_Module_(C++) project -> Properties.
Go to Configuration Properties -> C/C++ -> General.
In Additional Include Directories, add the path to the pybind11/include folder.

