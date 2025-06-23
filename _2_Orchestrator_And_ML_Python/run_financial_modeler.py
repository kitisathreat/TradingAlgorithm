#!/usr/bin/env python3
"""
Interactive launcher for the FinancialModeler class.
Provides a user-friendly interface to build financial models.
"""

import sys
import os
from pathlib import Path

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from financial_modeler import FinancialModeler

def main():
    """
    Main interactive launcher for the FinancialModeler.
    """
    print("=" * 70)
    print("FINANCIAL MODELER - INTERACTIVE LAUNCHER")
    print("=" * 70)
    print("This tool creates comprehensive 3-part financial models")
    print("including Income Statement, Balance Sheet, and Cash Flows.")
    print()
    
    # Initialize the modeler
    modeler = FinancialModeler()
    
    while True:
        print("\nOptions:")
        print("1. Build financial model for a company")
        print("2. Run sensitivity analysis")
        print("3. Update model assumptions")
        print("4. View model summary")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            build_model(modeler)
        elif choice == "2":
            run_sensitivity(modeler)
        elif choice == "3":
            update_assumptions(modeler)
        elif choice == "4":
            view_summary(modeler)
        elif choice == "5":
            print("\nExiting Financial Modeler. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-5.")

def build_model(modeler):
    """
    Build a financial model for a company.
    """
    print("\n" + "-" * 50)
    print("BUILD FINANCIAL MODEL")
    print("-" * 50)
    
    ticker = input("Enter stock ticker (e.g., AAPL, MSFT, GOOGL): ").strip().upper()
    
    if not ticker:
        print("No ticker provided. Returning to main menu.")
        return
    
    try:
        print(f"\nBuilding financial model for {ticker}...")
        
        # Get company info first
        company_info = modeler.get_company_info(ticker)
        if company_info:
            print(f"Company: {company_info.get('name', 'Unknown')}")
            print(f"Sector: {company_info.get('sector', 'Unknown')}")
        
        # Build the complete model
        filename = f"{ticker}_Financial_Model_{modeler.get_current_date()}.xlsx"
        model_components = modeler.build_financial_model(
            ticker, 
            save_to_excel=True,
            filename=filename
        )
        
        print(f"\n✅ Financial model created successfully!")
        print(f"📁 Excel file: {filename}")
        print(f"📊 Model includes:")
        print(f"   - Consolidated financial statements")
        print(f"   - Assumptions summary")
        print(f"   - Case master control")
        print(f"   - Company information")
        
    except Exception as e:
        print(f"❌ Error building model: {e}")

def run_sensitivity(modeler):
    """
    Run sensitivity analysis.
    """
    print("\n" + "-" * 50)
    print("SENSITIVITY ANALYSIS")
    print("-" * 50)
    
    ticker = input("Enter stock ticker: ").strip().upper()
    if not ticker:
        print("No ticker provided. Returning to main menu.")
        return
    
    print("\nAvailable variables for sensitivity analysis:")
    print("1. Revenue_Growth_Rate")
    print("2. Cost_of_Revenue_Percent")
    print("3. Operating_Expenses_Percent")
    print("4. Tax_Rate")
    print("5. Working_Capital_Percent")
    print("6. CapEx_Percent")
    
    variable_choice = input("\nEnter variable number (1-6): ").strip()
    
    variable_map = {
        "1": "Revenue_Growth_Rate",
        "2": "Cost_of_Revenue_Percent", 
        "3": "Operating_Expenses_Percent",
        "4": "Tax_Rate",
        "5": "Working_Capital_Percent",
        "6": "CapEx_Percent"
    }
    
    if variable_choice not in variable_map:
        print("Invalid choice. Returning to main menu.")
        return
    
    variable = variable_map[variable_choice]
    
    try:
        base_value = float(input(f"Enter base value for {variable}: "))
        range_pct = float(input("Enter sensitivity range (e.g., 0.20 for ±20%): "))
        
        print(f"\nRunning sensitivity analysis for {variable}...")
        results = modeler.run_sensitivity_analysis(ticker, variable, base_value, range_pct)
        
        print(f"\n📈 Sensitivity Analysis Results:")
        print(results.to_string(index=False))
        
        # Save results
        filename = f"{ticker}_{variable}_Sensitivity_{modeler.get_current_date()}.csv"
        results.to_csv(filename, index=False)
        print(f"\n📁 Results saved to: {filename}")
        
    except ValueError:
        print("Invalid input. Please enter numeric values.")
    except Exception as e:
        print(f"❌ Error running sensitivity analysis: {e}")

def update_assumptions(modeler):
    """
    Update model assumptions.
    """
    print("\n" + "-" * 50)
    print("UPDATE ASSUMPTIONS")
    print("-" * 50)
    
    ticker = input("Enter stock ticker: ").strip().upper()
    if not ticker:
        print("No ticker provided. Returning to main menu.")
        return
    
    print("\nAvailable cases:")
    print("1. Base Case")
    print("2. Bull Case") 
    print("3. Bear Case")
    
    case_choice = input("\nEnter case number (1-3): ").strip()
    
    case_map = {
        "1": "Base Case",
        "2": "Bull Case",
        "3": "Bear Case"
    }
    
    if case_choice not in case_map:
        print("Invalid choice. Returning to main menu.")
        return
    
    case = case_map[case_choice]
    
    print(f"\nAvailable variables for {case}:")
    print("1. Revenue_Growth_Rate")
    print("2. Cost_of_Revenue_Percent")
    print("3. Operating_Expenses_Percent")
    print("4. Tax_Rate")
    print("5. Working_Capital_Percent")
    print("6. CapEx_Percent")
    
    variable_choice = input("\nEnter variable number (1-6): ").strip()
    
    variable_map = {
        "1": "Revenue_Growth_Rate",
        "2": "Cost_of_Revenue_Percent",
        "3": "Operating_Expenses_Percent", 
        "4": "Tax_Rate",
        "5": "Working_Capital_Percent",
        "6": "CapEx_Percent"
    }
    
    if variable_choice not in variable_map:
        print("Invalid choice. Returning to main menu.")
        return
    
    variable = variable_map[variable_choice]
    
    try:
        new_value = float(input(f"Enter new value for {variable}: "))
        
        modeler.update_assumption(ticker, variable, case, new_value)
        print(f"✅ Assumption updated successfully!")
        
    except ValueError:
        print("Invalid input. Please enter a numeric value.")
    except Exception as e:
        print(f"❌ Error updating assumption: {e}")

def view_summary(modeler):
    """
    View model summary.
    """
    print("\n" + "-" * 50)
    print("MODEL SUMMARY")
    print("-" * 50)
    
    ticker = input("Enter stock ticker: ").strip().upper()
    if not ticker:
        print("No ticker provided. Returning to main menu.")
        return
    
    try:
        summary = modeler.get_model_summary(ticker)
        
        if summary:
            print(f"\n📊 Model Summary for {ticker}:")
            print(f"   Company: {summary.get('company_name', 'Unknown')}")
            print(f"   Model Date: {summary.get('model_date', 'Unknown')}")
            print(f"   Consolidated Statements: {summary.get('consolidated_statements_rows', 0)} rows")
            print(f"   Assumptions: {summary.get('assumptions_count', 0)} variables")
            print(f"   Cases: {summary.get('cases_count', 0)} scenarios")
            print(f"   Available Cases: {', '.join(summary.get('available_cases', []))}")
            print(f"   Key Variables: {', '.join(summary.get('key_variables', []))}")
        else:
            print(f"No model found for {ticker}. Please build a model first.")
            
    except Exception as e:
        print(f"❌ Error getting model summary: {e}")

# Add helper method to FinancialModeler class
def get_current_date(self):
    """Get current date in YYYYMMDD format."""
    from datetime import datetime
    return datetime.now().strftime('%Y%m%d')

# Add the helper method to the FinancialModeler class
FinancialModeler.get_current_date = get_current_date

if __name__ == "__main__":
    main() 