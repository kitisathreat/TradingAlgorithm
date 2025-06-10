# Hybrid Trading Bot with ML Decision Engine

This repository contains the complete source code for a sophisticated, hybrid C++/Python algorithmic trading system. The project is designed to capture expert knowledge through an interactive training interface, train a machine learning model based on that expertise, and deploy the resulting model for live, real-time trading decisions.

The architecture is distributed across three main components: a core development project in Visual Studio, a lightweight server for interactive data collection (designed for a Raspberry Pi), and a high-performance computer vision service for facial sentiment analysis.

---

## System Architecture

The project operates in two distinct phases: an offline **Training Phase** to build the model and an online **Inference Phase** for live trading.

#### **Phase 1: Interactive Training Data Collection**

#### **Phase 2: Live Trading**

## Visual Studio Project Structure

The solution is managed within Visual Studio and organized into two main projects.

* `TradingBotSolution.sln`: The master solution file that holds both projects.

### 1. `1_High_Performance_Module_(C++)`
A C++ project that compiles into a Python module (`.pyd`). It serves as a high-speed, rule-based decision engine or can be used for performance-critical feature calculations.

-   `/Header Files/engine.h`: Defines the `TradingModel` class and `TradeSignal` enum.
-   `/Source Files/engine.cpp`: Implements the C++ logic.
-   `/Source Files/bindings.cpp`: Uses `pybind11` to create the Python interface.

### 2. `2_Orchestrator_And_ML_(Python)`
The main Python project containing all the application logic, training scripts, and deployment services.

-   `/deployment_services/`: Contains standalone services.
    -   `vision_service.py`: The FastAPI server for facial analysis, meant to run on a powerful PC.
-   `/interactive_training_app/`: The complete application for the investor training session.
    -   `/backend/pi_server.py`: The FastAPI server to be deployed on the Raspberry Pi.
    -   `/backend/historical_data.csv`: Sample historical data to present to the investor.
    -   `/frontend/`: Contains the `index.html`, `style.css`, and `script.js` for the web interface.
-   `train_model.py`: **Script 1.** Loads the data collected from the interactive session and trains a `scikit-learn` model.
-   `live_trader.py`: **Script 2.** The main entry point for the live trading bot, which loads and uses the trained model.
-   `.env`: A local, untracked file for storing secret API keys.
-   `requirements.txt`: A list of all required Python packages.

---

## Setup and Installation

Follow these steps to set up the development environment on your main PC.

### 1. Prerequisites
-   **Visual Studio 2022** with the following workloads installed:
    -   `Python development`
    -   `Desktop development with C++` (ensure the "C++ MFC..." component is included for DLL templates)
-   **Python 3.9+**
-   **Git LFS** (for handling large model files). Download from [https://git-lfs.github.com/](https://git-lfs.github.com/) and run `git lfs install` in your terminal once per machine.

### 2. API Key Setup
-   Create a file named `.env` in the root of the `2_Orchestrator_And_ML_(Python)` project.
-   Add your secret keys to this file. It is ignored by Git and will not be pushed to GitHub.
    ```ini
    # .env file content
    APCA_API_KEY_ID="YOUR_ALPACA_KEY"
    APCA_API_SECRET_KEY="YOUR_ALPACA_SECRET"
    FMP_API_KEY="YOUR_FMP_KEY"
    POLYGON_API_KEY="YOUR_POLYGON_KEY" # Optional
    ```

### 3. Python Dependencies
-   Navigate to the `2_Orchestrator_And_ML_(Python)` directory in a terminal.
-   Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
-   Install all required packages:
    ```bash
    pip install -r requirements.txt
    ```

### 4. C++ Module Compilation
-   Open the `TradingBotSolution.sln` file in Visual Studio.
-   Set the solution configuration to `Release` and `x64`.
-   Right-click on the `1_High_Performance_Module_(C++)` project and select **Build**.
-   This will create a `decision_engine.pyd` file in the build output directory. Copy this `.pyd` file into the root of the `2_Orchestrator_And_ML_(Python)` project so Python can import it.

---

## Workflow: How to Use the System

Follow this workflow from data collection to live trading.

### Step 1: Prepare Historical Data
-   Edit the `interactive_training_app/backend/historical_data.csv` file. Populate it with the historical scenarios you want to present to your expert investor.

### Step 2: Run Deployment Services
-   **On your powerful PC:** Run the vision service to handle facial analysis.
    ```bash
    # In a terminal on your PC:
    uvicorn deployment_services.vision_service:app --host 0.0.0.0 --port 8001
    ```
-   **On your Raspberry Pi:** Run the interactive server.
    ```bash
    # In a terminal on your Pi (after copying the files):
    gunicorn -w 2 -k uvicorn.workers.UvicornWorker pi_server:app
    ```
    Remember to update the IP addresses in the `pi_server.py` and `script.js` files.

### Step 3: Conduct the Interactive Training Session
-   Open the `frontend/index.html` file in a web browser.
-   Have your expert investor make decisions for each scenario presented.
-   Their decisions, combined with facial analysis, will be logged to `investor_decisions_with_vision.csv` on the Raspberry Pi.

### Step 4: Train the Machine Learning Model
-   Once you have collected enough data, copy the `investor_decisions_with_vision.csv` file back to your development PC into the `interactive_training_app/backend/` folder.
-   In Visual Studio, right-click `train_model.py` and select **Start Without Debugging** (or run `python train_model.py` in a terminal).
-   This will process the data and create the `investor_model.joblib` file, which is your trained "brain."

### Step 5: Run the Live Trading Bot
-   With the `investor_model.joblib` file present, you can now run the live bot.
-   In Visual Studio, set `live_trader.py` as the Startup File.
-   Press **Start** to run the bot. It will connect to Alpaca, listen for market data, and make trading decisions based on your trained model's predictions.

---

## Technology Stack

-   **Backend & Orchestration:** Python 3.9+
-   **High-Performance Module:** C++17
-   **Web Framework:** FastAPI
-   **Python/C++ Bridge:** pybind11
-   **Machine Learning:** scikit-learn
-   **Data Manipulation:** pandas, NumPy
-   **Computer Vision:** deepface, OpenCV
-   **Frontend:** HTML5, CSS3, vanilla JavaScript
-   **IDE:** Visual Studio 2022