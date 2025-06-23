#!/usr/bin/env python3
"""
Test script for the FinancialModeler class.
Demonstrates how to use the 3-part financial modeling system.
"""

import sys
import os
from pathlib import Path

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from financial_modeler import FinancialModeler

def test_financial_modeler():
    """
    Test the FinancialModeler class with a sample company.
    """
    print("=" * 60)
    print("FINANCIAL MODELER TEST")
    print("=" * 60)
    
    # Initialize the financial modeler
    print("Initializing FinancialModeler...")
    modeler = FinancialModeler()
    
    # Test with Apple (AAPL) as an example
    ticker = "AAPL"
    print(f"\nBuilding financial model for {ticker}...")
    
    try:
        # Get company information
        print("1. Getting company information...")
        company_info = modeler.get_company_info(ticker)
        print(f"   Company: {company_info.get('name', 'Unknown')}")
        print(f"   Sector: {company_info.get('sector', 'Unknown')}")
        print(f"   Market Cap: ${company_info.get('market_cap', 0):,.0f}")
        
        # Get SEC filings
        print("\n2. Getting SEC filings...")
        filings = modeler.get_sec_filings(ticker, filing_type='10-K', limit=3)
        print(f"   Found {len(filings)} recent 10-K filings")
        
        # Extract financial data
        print("\n3. Extracting financial data...")
        financial_data = modeler.extract_financial_data(ticker)
        print(f"   Income Statement: {len(financial_data['income_statement'])} rows")
        print(f"   Balance Sheet: {len(financial_data['balance_sheet'])} rows")
        print(f"   Cash Flow: {len(financial_data['cash_flow'])} rows")
        
        # Create consolidated model
        print("\n4. Creating consolidated model...")
        consolidated = modeler.create_consolidated_model(ticker)
        print(f"   Consolidated model: {len(consolidated)} rows")
        
        # Create assumptions summary
        print("\n5. Creating assumptions summary...")
        assumptions = modeler.create_assumptions_summary(ticker)
        print(f"   Assumptions: {len(assumptions)} variables across 3 cases")
        
        # Create case master control
        print("\n6. Creating case master control...")
        cases = modeler.create_case_master_control(ticker)
        print(f"   Cases: {len(cases)} scenarios defined")
        
        # Build complete model
        print("\n7. Building complete financial model...")
        model_components = modeler.build_financial_model(
            ticker, 
            save_to_excel=True,
            filename=f"{ticker}_Financial_Model_Test.xlsx"
        )
        
        # Get model summary
        print("\n8. Getting model summary...")
        summary = modeler.get_model_summary(ticker)
        print("   Model Summary:")
        for key, value in summary.items():
            print(f"     {key}: {value}")
        
        # Run sensitivity analysis
        print("\n9. Running sensitivity analysis...")
        sensitivity = modeler.run_sensitivity_analysis(ticker, "Revenue_Growth_Rate", 0.10)
        print(f"   Sensitivity analysis: {len(sensitivity)} scenarios")
        print("   Sample results:")
        print(sensitivity.head(3).to_string(index=False))
        
        # Update an assumption
        print("\n10. Updating assumption...")
        modeler.update_assumption(ticker, "Revenue_Growth_Rate", "Base Case", 0.12)
        
        print("\n" + "=" * 60)
        print("FINANCIAL MODELER TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Excel file created: {ticker}_Financial_Model_Test.xlsx")
        print("\nThe model includes:")
        print("  - Consolidated financial statements")
        print("  - Assumptions summary with sensitivity variables")
        print("  - Case master control for scenario modeling")
        print("  - Company information")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Test failed. Please check the error message above.")
        return False

if __name__ == "__main__":
    # Run the main test
    success = test_financial_modeler()
    
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!") 