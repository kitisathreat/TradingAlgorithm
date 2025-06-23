# Financial Modeler

A comprehensive Python class for creating 3-part financial models from SEC filings and market data.

## Overview

The `FinancialModeler` class provides a complete solution for:
- Extracting financial data from SEC 10-K and 10-Q filings
- Creating standardized 3-part financial models (Income Statement, Balance Sheet, Cash Flows)
- Building sensitivity analysis and scenario modeling capabilities
- Exporting models to Excel with multiple sheets

## Features

### 🏢 Data Extraction
- **SEC Filings**: Access to 10-K and 10-Q filings via SEC API
- **Market Data**: Real-time financial data via Yahoo Finance
- **Company Information**: Basic company details and metrics

### 📊 Financial Modeling
- **Consolidated Statements**: Combined Income Statement, Balance Sheet, and Cash Flow
- **Standardized Format**: Consistent data structure across all statements
- **Historical Data**: Multi-year financial data analysis

### 🎯 Scenario Analysis
- **Assumptions Summary**: Key variables for sensitivity analysis
- **Case Master Control**: Base, Bull, and Bear case scenarios
- **Sensitivity Analysis**: Impact analysis of key variables

### 📁 Excel Export
- **Multiple Sheets**: Separate sheets for each model component
- **Formatted Output**: Professional Excel formatting
- **Easy Sharing**: Standard Excel format for stakeholders

## Installation

### Prerequisites
- Python 3.9+
- Required packages (see requirements.txt)

### Quick Setup
```bash
# Install required packages
pip install pandas numpy yfinance openpyxl sec-api

# Or install from requirements.txt
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from financial_modeler import FinancialModeler

# Initialize the modeler
modeler = FinancialModeler()

# Build a complete financial model
ticker = "AAPL"
model_components = modeler.build_financial_model(ticker, save_to_excel=True)
```

### Interactive Launcher

Run the interactive launcher for a user-friendly interface:

```bash
# Windows
run_financial_modeler.bat

# Python
python run_financial_modeler.py
```

### Advanced Usage

```python
# Get company information
company_info = modeler.get_company_info("AAPL")

# Extract financial data
financial_data = modeler.extract_financial_data("AAPL")

# Create individual components
consolidated = modeler.create_consolidated_model("AAPL")
assumptions = modeler.create_assumptions_summary("AAPL")
cases = modeler.create_case_master_control("AAPL")

# Run sensitivity analysis
sensitivity = modeler.run_sensitivity_analysis("AAPL", "Revenue_Growth_Rate", 0.10)

# Update assumptions
modeler.update_assumption("AAPL", "Revenue_Growth_Rate", "Base Case", 0.12)
```

## Model Structure

### 1. Consolidated Statements Sheet
Contains all three financial statements in a standardized format:
- **Income Statement**: Revenue, costs, operating income, net income
- **Balance Sheet**: Assets, liabilities, equity
- **Cash Flow**: Operating, investing, and financing cash flows

### 2. Assumptions Summary Sheet
Key variables for sensitivity analysis:
- **Revenue Growth Rate**: Annual revenue growth assumptions
- **Cost of Revenue %**: Cost structure assumptions
- **Operating Expenses %**: Operating efficiency assumptions
- **Tax Rate**: Effective tax rate assumptions
- **Working Capital %**: Working capital requirements
- **CapEx %**: Capital expenditure assumptions

### 3. Case Master Control Sheet
Scenario modeling with three cases:
- **Base Case**: Conservative assumptions
- **Bull Case**: Optimistic growth scenario
- **Bear Case**: Pessimistic downturn scenario

### 4. Company Information Sheet
Basic company details and metrics

## Key Methods

### Data Extraction
- `get_company_info(ticker)`: Get basic company information
- `get_sec_filings(ticker, filing_type, limit)`: Retrieve SEC filings
- `extract_financial_data(ticker)`: Extract and standardize financial data

### Model Creation
- `create_consolidated_model(ticker)`: Create combined financial statements
- `create_assumptions_summary(ticker)`: Generate assumptions framework
- `create_case_master_control(ticker)`: Create scenario modeling structure

### Analysis
- `run_sensitivity_analysis(ticker, variable, base_value, range_pct)`: Sensitivity analysis
- `update_assumption(ticker, variable, case, new_value)`: Update model assumptions
- `get_model_summary(ticker)`: Get model overview

### Export
- `build_financial_model(ticker, save_to_excel, filename)`: Complete model build
- `save_to_excel(ticker, filename)`: Export to Excel format

## Example Output

### Excel File Structure
```
AAPL_Financial_Model_20241201.xlsx
├── Consolidated_Statements
│   ├── Income Statement data
│   ├── Balance Sheet data
│   └── Cash Flow data
├── Assumptions_Summary
│   ├── Variable definitions
│   ├── Case-specific values
│   └── Descriptions
├── Case_Master_Control
│   ├── Base Case parameters
│   ├── Bull Case parameters
│   └── Bear Case parameters
└── Company_Info
    └── Basic company details
```

### Sample Sensitivity Analysis
```
Variable              Value  Revenue_Impact  Profit_Impact  Net_Income_Impact
Revenue_Growth_Rate   0.08   0.80           1.00           0.80
Revenue_Growth_Rate   0.09   0.90           1.00           0.90
Revenue_Growth_Rate   0.10   1.00           1.00           1.00
Revenue_Growth_Rate   0.11   1.10           1.00           1.10
Revenue_Growth_Rate   0.12   1.20           1.00           1.20
```

## Configuration

### SEC API Key (Optional)
For enhanced SEC filing access, set your SEC API key:

```python
# Environment variable
export SEC_API_KEY="your_api_key_here"

# Or pass directly
modeler = FinancialModeler(api_key="your_api_key_here")
```

### Custom Assumptions
Modify the default assumptions in the `create_assumptions_summary()` method:

```python
# Update specific assumption
modeler.update_assumption("AAPL", "Revenue_Growth_Rate", "Base Case", 0.15)

# Or modify the assumptions dataframe directly
assumptions = modeler.assumptions["AAPL"]
assumptions.loc[assumptions['Variable'] == 'Revenue_Growth_Rate', 'Value'] = 0.15
```

## Error Handling

The class includes comprehensive error handling:
- **API Failures**: Falls back to sample data if SEC API is unavailable
- **Data Issues**: Uses yfinance as backup for financial data
- **Missing Files**: Graceful handling of missing or corrupted data
- **User Input**: Validation of user inputs and parameters

## Testing

Run the test script to verify functionality:

```bash
python test_financial_modeler.py
```

The test script will:
- Build a sample financial model
- Run sensitivity analysis
- Update assumptions
- Generate Excel output

## Dependencies

### Required Packages
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `yfinance`: Yahoo Finance data access
- `openpyxl`: Excel file handling
- `sec-api`: SEC filing access (optional)

### Optional Dependencies
- `requests`: HTTP requests for API calls
- `matplotlib`: Data visualization (for future enhancements)

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install pandas numpy yfinance openpyxl
   ```

2. **SEC API Issues**
   - Check API key validity
   - Verify internet connection
   - Falls back to sample data if unavailable

3. **Excel Export Issues**
   - Ensure openpyxl is installed
   - Check file permissions
   - Verify sufficient disk space

4. **Data Quality Issues**
   - Some companies may have limited data
   - Check ticker symbol validity
   - Verify company is publicly traded

### Performance Tips

- Use specific ticker symbols (avoid common words)
- Limit SEC filing requests to avoid rate limits
- Cache results for repeated analysis
- Use sample data for testing and development

## Future Enhancements

### Planned Features
- **DCF Valuation**: Discounted cash flow modeling
- **Comparable Analysis**: Peer company benchmarking
- **Risk Analysis**: Monte Carlo simulations
- **Visualization**: Charts and graphs
- **Real-time Updates**: Live data feeds
- **API Integration**: Additional data sources

### Contributing
To contribute to the Financial Modeler:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is part of the TradingAlgorithm repository and follows the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test script for examples
3. Examine the source code for implementation details
4. Create an issue in the repository

---

**Note**: This financial modeler is designed for educational and analytical purposes. Always verify data accuracy and consult with financial professionals for investment decisions. 