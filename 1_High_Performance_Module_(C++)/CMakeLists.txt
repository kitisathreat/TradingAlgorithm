cmake_minimum_required(VERSION 3.25)
project(TradingEngine)

# Find Python and pybind11
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Add the module
pybind11_add_module(decision_engine 
    bindings.cpp
    engine.cpp
)

# Include directories
target_include_directories(decision_engine PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${pybind11_INCLUDE_DIRS}
)

# Set C++ standard
set_target_properties(decision_engine PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Set output directory
set_target_properties(decision_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/2_Orchestrator_And_ML_(Python)"
) 