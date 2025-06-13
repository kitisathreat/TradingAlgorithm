#!/bin/bash

# Install system dependencies
apt-get update
apt-get install -y build-essential cmake python3-dev g++

# Build and install the C++ module
cd "1_High_Performance_Module_(C++)"
pip install -e .

# Build the decision_engine C++ module
python setup.py build_ext --inplace

# Copy the compiled module to the orchestrator directory
cp decision_engine*.so ../_2_Orchestrator_And_ML_Python/ 2>/dev/null || cp decision_engine*.pyd ../_2_Orchestrator_And_ML_Python/ 2>/dev/null

cd ..

# Create necessary directories
mkdir -p "_2_Orchestrator_And_ML_Python/models"
mkdir -p "_2_Orchestrator_And_ML_Python/training_data" 