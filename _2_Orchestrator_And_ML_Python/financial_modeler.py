import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional, Tuple, Any
import yfinance as yf
from sec_api import QueryApi
import os
from pathlib import Path

warnings.filterwarnings('ignore')

class FinancialModeler:
    """
    A comprehensive financial modeling class that handles SEC filings and creates
    a 3-part financial model (Income Statement, Balance Sheet, Cash Flows).
    
    Features:
    - SEC 10-K and 10-Q filing data extraction
    - Consolidated financial statements
    - Assumptions summary with sensitivity analysis
    - Case master control for scenario modeling
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the FinancialModeler.
        
        Args:
            api_key: SEC API key for accessing filing data
        """
        self.api_key = api_key or os.getenv('SEC_API_KEY')
        self.query_api = QueryApi(api_key=self.api_key) if self.api_key else None
        self.company_data = {}
        self.financial_statements = {}
        self.assumptions = {}
        self.cases = {}
        self.model_data = {}
        
    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get basic company information using yfinance.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing company information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            company_info = {
                'ticker': ticker,
                'name': info.get('longName', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'cik': info.get('cik', None)
            }
            
            self.company_data[ticker] = company_info
            return company_info
            
        except Exception as e:
            print(f"Error getting company info for {ticker}: {e}")
            return {}
    
    def get_sec_filings(self, ticker: str, filing_type: str = '10-K', limit: int = 5) -> List[Dict]:
        """
        Get SEC filings for a company.
        
        Args:
            ticker: Stock ticker symbol
            filing_type: Type of filing ('10-K' or '10-Q')
            limit: Number of filings to retrieve
            
        Returns:
            List of filing data
        """
        if not self.query_api:
            print("SEC API key not available. Using sample data.")
            return self._get_sample_filings(ticker, filing_type, limit)
        
        try:
            query = {
                "query": {
                    "query_string": {
                        "query": f"ticker:{ticker} AND formType:\"{filing_type}\""
                    }
                },
                "from": "0",
                "size": str(limit),
                "sort": [{"filedAt": {"order": "desc"}}]
            }
            
            response = self.query_api.get_filings(query)
            return response.get('filings', [])
            
        except Exception as e:
            print(f"Error getting SEC filings for {ticker}: {e}")
            return self._get_sample_filings(ticker, filing_type, limit)
    
    def _get_sample_filings(self, ticker: str, filing_type: str, limit: int) -> List[Dict]:
        """
        Generate sample filing data for testing purposes.
        
        Args:
            ticker: Stock ticker symbol
            filing_type: Type of filing
            limit: Number of filings to generate
            
        Returns:
            List of sample filing data
        """
        sample_filings = []
        base_date = datetime.now()
        
        for i in range(limit):
            filing_date = base_date - timedelta(days=365 * (i + 1))
            sample_filings.append({
                'ticker': ticker,
                'formType': filing_type,
                'filedAt': filing_date.strftime('%Y-%m-%d'),
                'filingDate': filing_date.strftime('%Y-%m-%d'),
                'periodOfReport': filing_date.strftime('%Y-%m-%d'),
                'description': f'Sample {filing_type} filing for {ticker}',
                'linkToFilingDetails': f'https://example.com/{ticker}_{filing_type}_{i}'
            })
        
        return sample_filings
    
    def extract_financial_data(self, ticker: str, filing_type: str = '10-K') -> Dict[str, pd.DataFrame]:
        """
        Extract financial data from SEC filings and create standardized dataframes.
        
        Args:
            ticker: Stock ticker symbol
            filing_type: Type of filing to analyze
            
        Returns:
            Dictionary containing Income Statement, Balance Sheet, and Cash Flow dataframes
        """
        # Get historical financial data using yfinance as fallback
        try:
            stock = yf.Ticker(ticker)
            
            # Get financial statements
            income_stmt = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            # Standardize column names and format
            financial_data = {
                'income_statement': self._standardize_financial_statement(income_stmt, 'Income Statement'),
                'balance_sheet': self._standardize_financial_statement(balance_sheet, 'Balance Sheet'),
                'cash_flow': self._standardize_financial_statement(cash_flow, 'Cash Flow')
            }
            
            self.financial_statements[ticker] = financial_data
            return financial_data
            
        except Exception as e:
            print(f"Error extracting financial data for {ticker}: {e}")
            return self._create_sample_financial_data(ticker)
    
    def _standardize_financial_statement(self, df: pd.DataFrame, statement_type: str) -> pd.DataFrame:
        """
        Standardize financial statement dataframe format.
        
        Args:
            df: Raw financial statement dataframe
            statement_type: Type of financial statement
            
        Returns:
            Standardized dataframe
        """
        if df.empty:
            return pd.DataFrame()
        
        # Transpose to have dates as columns
        df = df.T
        
        # Convert to millions for readability
        df = df / 1_000_000
        
        # Add statement type identifier
        df['Statement_Type'] = statement_type
        
        # Reset index to make dates a column
        df = df.reset_index()
        df = df.rename(columns={'index': 'Date'})
        
        return df
    
    def _create_sample_financial_data(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """
        Create sample financial data for testing purposes.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing sample financial statements
        """
        # Sample dates
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='Y')
        
        # Sample Income Statement
        income_stmt_data = {
            'Date': dates,
            'Statement_Type': 'Income Statement',
            'Revenue': [1000, 1100, 1200, 1300],
            'Cost_of_Revenue': [600, 660, 720, 780],
            'Gross_Profit': [400, 440, 480, 520],
            'Operating_Expenses': [200, 220, 240, 260],
            'Operating_Income': [200, 220, 240, 260],
            'Interest_Expense': [20, 22, 24, 26],
            'Net_Income': [180, 198, 216, 234]
        }
        
        # Sample Balance Sheet
        balance_sheet_data = {
            'Date': dates,
            'Statement_Type': 'Balance Sheet',
            'Cash_and_Equivalents': [200, 220, 240, 260],
            'Accounts_Receivable': [150, 165, 180, 195],
            'Inventory': [100, 110, 120, 130],
            'Total_Current_Assets': [450, 495, 540, 585],
            'Property_Plant_Equipment': [800, 880, 960, 1040],
            'Total_Assets': [1250, 1375, 1500, 1625],
            'Accounts_Payable': [120, 132, 144, 156],
            'Short_Term_Debt': [100, 110, 120, 130],
            'Total_Current_Liabilities': [220, 242, 264, 286],
            'Long_Term_Debt': [400, 440, 480, 520],
            'Total_Liabilities': [620, 682, 744, 806],
            'Total_Equity': [630, 693, 756, 819]
        }
        
        # Sample Cash Flow
        cash_flow_data = {
            'Date': dates,
            'Statement_Type': 'Cash Flow',
            'Net_Income': [180, 198, 216, 234],
            'Depreciation_Amortization': [80, 88, 96, 104],
            'Changes_in_Working_Capital': [-20, -22, -24, -26],
            'Operating_Cash_Flow': [240, 264, 288, 312],
            'Capital_Expenditures': [-100, -110, -120, -130],
            'Investing_Cash_Flow': [-100, -110, -120, -130],
            'Debt_Issuance': [50, 55, 60, 65],
            'Dividends_Paid': [-90, -99, -108, -117],
            'Financing_Cash_Flow': [-40, -44, -48, -52],
            'Net_Cash_Flow': [100, 110, 120, 130]
        }
        
        return {
            'income_statement': pd.DataFrame(income_stmt_data),
            'balance_sheet': pd.DataFrame(balance_sheet_data),
            'cash_flow': pd.DataFrame(cash_flow_data)
        }
    
    def create_consolidated_model(self, ticker: str) -> pd.DataFrame:
        """
        Create a consolidated financial model combining all three statements.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Consolidated dataframe with all financial statements
        """
        if ticker not in self.financial_statements:
            self.extract_financial_data(ticker)
        
        financial_data = self.financial_statements[ticker]
        
        # Combine all statements
        consolidated = pd.concat([
            financial_data['income_statement'],
            financial_data['balance_sheet'],
            financial_data['cash_flow']
        ], ignore_index=True)
        
        # Add model metadata
        consolidated['Model_Date'] = datetime.now().strftime('%Y-%m-%d')
        consolidated['Ticker'] = ticker
        
        return consolidated
    
    def create_assumptions_summary(self, ticker: str) -> pd.DataFrame:
        """
        Create an assumptions summary sheet for sensitivity analysis.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Assumptions dataframe with input fields for sensitivity analysis
        """
        assumptions_data = {
            'Category': [
                'Revenue Growth',
                'Revenue Growth',
                'Revenue Growth',
                'Cost of Revenue %',
                'Cost of Revenue %',
                'Cost of Revenue %',
                'Operating Expenses %',
                'Operating Expenses %',
                'Operating Expenses %',
                'Tax Rate %',
                'Tax Rate %',
                'Tax Rate %',
                'Working Capital %',
                'Working Capital %',
                'Working Capital %',
                'Capital Expenditures %',
                'Capital Expenditures %',
                'Capital Expenditures %'
            ],
            'Variable': [
                'Revenue_Growth_Rate',
                'Revenue_Growth_Rate',
                'Revenue_Growth_Rate',
                'Cost_of_Revenue_Percent',
                'Cost_of_Revenue_Percent',
                'Cost_of_Revenue_Percent',
                'Operating_Expenses_Percent',
                'Operating_Expenses_Percent',
                'Operating_Expenses_Percent',
                'Tax_Rate',
                'Tax_Rate',
                'Tax_Rate',
                'Working_Capital_Percent',
                'Working_Capital_Percent',
                'Working_Capital_Percent',
                'CapEx_Percent',
                'CapEx_Percent',
                'CapEx_Percent'
            ],
            'Case': [
                'Base Case',
                'Bull Case',
                'Bear Case',
                'Base Case',
                'Bull Case',
                'Bear Case',
                'Base Case',
                'Bull Case',
                'Bear Case',
                'Base Case',
                'Bull Case',
                'Bear Case',
                'Base Case',
                'Bull Case',
                'Bear Case',
                'Base Case',
                'Bull Case',
                'Bear Case'
            ],
            'Value': [
                0.10,  # 10% revenue growth base case
                0.15,  # 15% revenue growth bull case
                0.05,  # 5% revenue growth bear case
                0.60,  # 60% cost of revenue base case
                0.58,  # 58% cost of revenue bull case
                0.62,  # 62% cost of revenue bear case
                0.20,  # 20% operating expenses base case
                0.18,  # 18% operating expenses bull case
                0.22,  # 22% operating expenses bear case
                0.25,  # 25% tax rate base case
                0.23,  # 23% tax rate bull case
                0.27,  # 27% tax rate bear case
                0.15,  # 15% working capital base case
                0.12,  # 12% working capital bull case
                0.18,  # 18% working capital bear case
                0.08,  # 8% CapEx base case
                0.06,  # 6% CapEx bull case
                0.10   # 10% CapEx bear case
            ],
            'Description': [
                'Annual revenue growth rate',
                'Annual revenue growth rate (optimistic)',
                'Annual revenue growth rate (pessimistic)',
                'Cost of revenue as % of revenue',
                'Cost of revenue as % of revenue (optimistic)',
                'Cost of revenue as % of revenue (pessimistic)',
                'Operating expenses as % of revenue',
                'Operating expenses as % of revenue (optimistic)',
                'Operating expenses as % of revenue (pessimistic)',
                'Effective tax rate',
                'Effective tax rate (optimistic)',
                'Effective tax rate (pessimistic)',
                'Working capital as % of revenue',
                'Working capital as % of revenue (optimistic)',
                'Working capital as % of revenue (pessimistic)',
                'Capital expenditures as % of revenue',
                'Capital expenditures as % of revenue (optimistic)',
                'Capital expenditures as % of revenue (pessimistic)'
            ]
        }
        
        assumptions_df = pd.DataFrame(assumptions_data)
        assumptions_df['Ticker'] = ticker
        assumptions_df['Last_Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.assumptions[ticker] = assumptions_df
        return assumptions_df
    
    def create_case_master_control(self, ticker: str) -> pd.DataFrame:
        """
        Create a case master control sheet for scenario modeling.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Case master control dataframe
        """
        case_data = {
            'Case_Name': ['Base Case', 'Bull Case', 'Bear Case'],
            'Revenue_Growth_Rate': [0.10, 0.15, 0.05],
            'Cost_of_Revenue_Percent': [0.60, 0.58, 0.62],
            'Operating_Expenses_Percent': [0.20, 0.18, 0.22],
            'Tax_Rate': [0.25, 0.23, 0.27],
            'Working_Capital_Percent': [0.15, 0.12, 0.18],
            'CapEx_Percent': [0.08, 0.06, 0.10],
            'Discount_Rate': [0.10, 0.08, 0.12],
            'Terminal_Growth_Rate': [0.03, 0.04, 0.02],
            'Forecast_Periods': [5, 5, 5],
            'Description': [
                'Conservative base case scenario',
                'Optimistic growth scenario',
                'Pessimistic downturn scenario'
            ],
            'Probability': [0.60, 0.25, 0.15],
            'Created_Date': [
                datetime.now().strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d')
            ]
        }
        
        case_df = pd.DataFrame(case_data)
        case_df['Ticker'] = ticker
        case_df['Last_Modified'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.cases[ticker] = case_df
        return case_df
    
    def build_financial_model(self, ticker: str, save_to_excel: bool = True, 
                            filename: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Build a complete 3-part financial model for a company.
        
        Args:
            ticker: Stock ticker symbol
            save_to_excel: Whether to save the model to Excel
            filename: Custom filename for the Excel file
            
        Returns:
            Dictionary containing all model components
        """
        print(f"Building financial model for {ticker}...")
        
        # Get company information
        company_info = self.get_company_info(ticker)
        print(f"Company: {company_info.get('name', ticker)}")
        
        # Extract financial data
        financial_data = self.extract_financial_data(ticker)
        print("Financial data extracted successfully")
        
        # Create consolidated model
        consolidated_model = self.create_consolidated_model(ticker)
        print("Consolidated model created")
        
        # Create assumptions summary
        assumptions_summary = self.create_assumptions_summary(ticker)
        print("Assumptions summary created")
        
        # Create case master control
        case_master = self.create_case_master_control(ticker)
        print("Case master control created")
        
        # Compile all model components
        model_components = {
            'consolidated_statements': consolidated_model,
            'assumptions_summary': assumptions_summary,
            'case_master_control': case_master,
            'company_info': pd.DataFrame([company_info])
        }
        
        self.model_data[ticker] = model_components
        
        # Save to Excel if requested
        if save_to_excel:
            filename = filename or f"{ticker}_Financial_Model_{datetime.now().strftime('%Y%m%d')}.xlsx"
            self.save_to_excel(ticker, filename)
            print(f"Model saved to {filename}")
        
        return model_components
    
    def save_to_excel(self, ticker: str, filename: str) -> None:
        """
        Save the financial model to an Excel file with multiple sheets.
        
        Args:
            ticker: Stock ticker symbol
            filename: Output filename
        """
        if ticker not in self.model_data:
            print(f"No model data found for {ticker}. Please run build_financial_model first.")
            return
        
        model_components = self.model_data[ticker]
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Consolidated statements sheet
            model_components['consolidated_statements'].to_excel(
                writer, sheet_name='Consolidated_Statements', index=False
            )
            
            # Assumptions summary sheet
            model_components['assumptions_summary'].to_excel(
                writer, sheet_name='Assumptions_Summary', index=False
            )
            
            # Case master control sheet
            model_components['case_master_control'].to_excel(
                writer, sheet_name='Case_Master_Control', index=False
            )
            
            # Company info sheet
            model_components['company_info'].to_excel(
                writer, sheet_name='Company_Info', index=False
            )
        
        print(f"Financial model saved to {filename}")
    
    def update_assumption(self, ticker: str, variable: str, case: str, new_value: float) -> None:
        """
        Update a specific assumption value.
        
        Args:
            ticker: Stock ticker symbol
            variable: Variable name to update
            case: Case name (Base Case, Bull Case, Bear Case)
            new_value: New value for the assumption
        """
        if ticker not in self.assumptions:
            print(f"No assumptions found for {ticker}")
            return
        
        mask = (self.assumptions[ticker]['Variable'] == variable) & \
               (self.assumptions[ticker]['Case'] == case)
        
        if mask.any():
            self.assumptions[ticker].loc[mask, 'Value'] = new_value
            self.assumptions[ticker].loc[mask, 'Last_Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Updated {variable} for {case} to {new_value}")
        else:
            print(f"Variable {variable} not found for case {case}")
    
    def get_model_summary(self, ticker: str) -> Dict[str, Any]:
        """
        Get a summary of the financial model.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing model summary information
        """
        if ticker not in self.model_data:
            return {}
        
        model_components = self.model_data[ticker]
        
        summary = {
            'ticker': ticker,
            'company_name': model_components['company_info'].iloc[0].get('name', 'Unknown'),
            'model_date': datetime.now().strftime('%Y-%m-%d'),
            'consolidated_statements_rows': len(model_components['consolidated_statements']),
            'assumptions_count': len(model_components['assumptions_summary']),
            'cases_count': len(model_components['case_master_control']),
            'available_cases': model_components['case_master_control']['Case_Name'].tolist(),
            'key_variables': model_components['assumptions_summary']['Variable'].unique().tolist()
        }
        
        return summary
    
    def run_sensitivity_analysis(self, ticker: str, variable: str, 
                               base_value: float, range_pct: float = 0.20) -> pd.DataFrame:
        """
        Run sensitivity analysis for a specific variable.
        
        Args:
            ticker: Stock ticker symbol
            variable: Variable to analyze
            base_value: Base value for the variable
            range_pct: Percentage range for sensitivity analysis
            
        Returns:
            Sensitivity analysis results
        """
        # Generate sensitivity range
        min_value = base_value * (1 - range_pct)
        max_value = base_value * (1 + range_pct)
        step = (max_value - min_value) / 10
        
        sensitivity_values = np.arange(min_value, max_value + step, step)
        
        # Calculate impact on key metrics (simplified)
        results = []
        for value in sensitivity_values:
            # Simplified impact calculation
            revenue_impact = value / base_value if 'Revenue' in variable else 1.0
            profit_impact = 1.0 - (value - base_value) / base_value if 'Cost' in variable else 1.0
            
            results.append({
                'Variable': variable,
                'Value': value,
                'Revenue_Impact': revenue_impact,
                'Profit_Impact': profit_impact,
                'Net_Income_Impact': revenue_impact * profit_impact
            })
        
        return pd.DataFrame(results)


# Example usage and testing
if __name__ == "__main__":
    # Initialize the financial modeler
    modeler = FinancialModeler()
    
    # Example: Build a financial model for Apple
    ticker = "AAPL"
    
    try:
        # Build the complete model
        model_components = modeler.build_financial_model(ticker, save_to_excel=True)
        
        # Get model summary
        summary = modeler.get_model_summary(ticker)
        print("\nModel Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Run sensitivity analysis
        sensitivity_results = modeler.run_sensitivity_analysis(ticker, "Revenue_Growth_Rate", 0.10)
        print(f"\nSensitivity Analysis Results:")
        print(sensitivity_results.head())
        
    except Exception as e:
        print(f"Error building model: {e}") 