@echo off
echo Building and installing C++ trading engine module...

:: Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Please run setup_local_env.bat first.
    exit /b 1
)

:: Change to C++ module directory
cd 1_High_Performance_Module_(C++)

:: Install required packages
echo Installing required packages...
pip install -r cpp_requirements.txt

:: Build and install the module
echo Building C++ module...
python setup.py build_ext --inplace

:: Install the module
echo Installing module...
pip install -e .

:: Return to root directory
cd ..

echo C++ module build and installation complete!
echo You can now use the decision_engine module in Python. 