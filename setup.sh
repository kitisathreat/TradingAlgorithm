#!/bin/bash

# Install system dependencies
apt-get update
apt-get install -y build-essential cmake python3-dev g++

# Build and install the C++ module
cd "1_High_Performance_Module_(C++)"
pip install -e .
cd ..

# Create necessary directories
mkdir -p "_2_Orchestrator_And_ML_Python/models"
mkdir -p "_2_Orchestrator_And_ML_Python/training_data" 