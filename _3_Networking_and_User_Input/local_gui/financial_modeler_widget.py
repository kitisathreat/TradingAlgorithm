import sys
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTableWidget, 
    QTableWidgetItem, QPushButton, QLabel, QLineEdit, QComboBox,
    QMessageBox, QHeaderView, QSplitter, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QTextEdit, QScrollArea, QFrame,
    QSizePolicy, QApplication, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex
from PyQt5.QtGui import QFont, QColor, QPalette, QBrush
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

# Add the ML directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from _2_Orchestrator_And_ML_Python.financial_modeler import FinancialModeler

class EditableTableWidget(QTableWidget):
    """Custom table widget with Excel-like editing capabilities."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_table()
        self.data_changed = False
        
    def setup_table(self):
        """Setup table appearance and behavior."""
        # Enable editing
        self.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed)
        
        # Setup headers
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.verticalHeader().setVisible(True)
        
        # Setup selection
        self.setSelectionBehavior(QTableWidget.SelectItems)
        self.setSelectionMode(QTableWidget.SingleSelection)
        
        # Setup appearance
        self.setAlternatingRowColors(True)
        self.setStyleSheet("""
            QTableWidget {
                gridline-color: #d0d0d0;
                background-color: white;
                alternate-background-color: #f8f9fa;
                selection-background-color: #0078d4;
                selection-color: white;
            }
            QTableWidget::item {
                padding: 4px;
                border: 1px solid #e0e0e0;
            }
            QTableWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                border: 1px solid #d0d0d0;
                font-weight: bold;
            }
        """)
        
        # Connect signals
        self.itemChanged.connect(self.on_item_changed)
        
    def on_item_changed(self, item):
        """Handle item changes."""
        self.data_changed = True
        
    def load_dataframe(self, df):
        """Load a pandas DataFrame into the table."""
        if df is None or df.empty:
            self.setRowCount(0)
            self.setColumnCount(0)
            return
            
        # Set dimensions
        self.setRowCount(len(df))
        self.setColumnCount(len(df.columns))
        
        # Set headers
        self.setHorizontalHeaderLabels(df.columns.tolist())
        self.setVerticalHeaderLabels(df.index.astype(str).tolist())
        
        # Populate data
        for i, row in enumerate(df.itertuples(index=False)):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value) if pd.notna(value) else "")
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                self.setItem(i, j, item)
                
        # Resize columns to content
        self.resizeColumnsToContents()
        
    def get_dataframe(self):
        """Get the current table data as a pandas DataFrame."""
        if self.rowCount() == 0 or self.columnCount() == 0:
            return pd.DataFrame()
            
        # Get headers
        headers = []
        for i in range(self.columnCount()):
            header_item = self.horizontalHeaderItem(i)
            headers.append(header_item.text() if header_item else f"Column_{i}")
            
        # Get data
        data = []
        for i in range(self.rowCount()):
            row_data = []
            for j in range(self.columnCount()):
                item = self.item(i, j)
                value = item.text() if item else ""
                # Try to convert to numeric if possible
                try:
                    if value.replace('.', '').replace('-', '').isdigit():
                        value = float(value) if '.' in value else int(value)
                except:
                    pass
                row_data.append(value)
            data.append(row_data)
            
        return pd.DataFrame(data, columns=headers)

class FinancialModelerWidget(QWidget):
    """Main widget for the financial modeler with Excel-like interface."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.modeler = FinancialModeler()
        self.current_ticker = None
        self.model_data = {}
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header section
        self.create_header_section(layout)
        
        # Main content area
        self.create_main_content(layout)
        
        # Status bar
        self.create_status_bar(layout)
        
    def create_header_section(self, parent_layout):
        """Create the header section with controls."""
        header_group = QGroupBox("Financial Modeler Controls")
        header_layout = QHBoxLayout(header_group)
        
        # Ticker input
        ticker_layout = QFormLayout()
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Enter ticker (e.g., AAPL)")
        self.ticker_input.setMaximumWidth(150)
        ticker_layout.addRow("Ticker:", self.ticker_input)
        header_layout.addLayout(ticker_layout)
        
        # Buttons
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_company_data)
        header_layout.addWidget(self.load_btn)
        
        self.build_model_btn = QPushButton("Build Model")
        self.build_model_btn.clicked.connect(self.build_financial_model)
        self.build_model_btn.setEnabled(False)
        header_layout.addWidget(self.build_model_btn)
        
        self.save_btn = QPushButton("Save to Excel")
        self.save_btn.clicked.connect(self.save_to_excel)
        self.save_btn.setEnabled(False)
        header_layout.addWidget(self.save_btn)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_data)
        header_layout.addWidget(self.refresh_btn)
        
        header_layout.addStretch()
        parent_layout.addWidget(header_group)
        
    def create_main_content(self, parent_layout):
        """Create the main content area with tabs."""
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                padding: 8px 16px;
                margin-right: 2px;
                border: 1px solid #c0c0c0;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 1px solid white;
            }
            QTabBar::tab:hover {
                background-color: #e0e0e0;
            }
        """)
        
        # Create tabs
        self.create_consolidated_tab()
        self.create_assumptions_tab()
        self.create_cases_tab()
        self.create_company_info_tab()
        
        parent_layout.addWidget(self.tab_widget)
        
    def create_consolidated_tab(self):
        """Create the consolidated statements tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Header
        header_label = QLabel("Consolidated Financial Statements")
        header_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(header_label)
        
        # Table
        self.consolidated_table = EditableTableWidget()
        layout.addWidget(self.consolidated_table)
        
        self.tab_widget.addTab(tab, "📊 Consolidated Statements")
        
    def create_assumptions_tab(self):
        """Create the assumptions summary tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Header
        header_label = QLabel("Assumptions Summary - Sensitivity Variables")
        header_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(header_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.assumption_variable_combo = QComboBox()
        self.assumption_variable_combo.addItems([
            "Revenue_Growth_Rate", "Cost_of_Revenue_Percent", 
            "Operating_Expenses_Percent", "Tax_Rate",
            "Working_Capital_Percent", "CapEx_Percent"
        ])
        controls_layout.addWidget(QLabel("Variable:"))
        controls_layout.addWidget(self.assumption_variable_combo)
        
        self.assumption_case_combo = QComboBox()
        self.assumption_case_combo.addItems(["Base Case", "Bull Case", "Bear Case"])
        controls_layout.addWidget(QLabel("Case:"))
        controls_layout.addWidget(self.assumption_case_combo)
        
        self.assumption_value_input = QDoubleSpinBox()
        self.assumption_value_input.setRange(-100, 100)
        self.assumption_value_input.setDecimals(4)
        self.assumption_value_input.setSingleStep(0.01)
        controls_layout.addWidget(QLabel("Value:"))
        controls_layout.addWidget(self.assumption_value_input)
        
        self.update_assumption_btn = QPushButton("Update")
        self.update_assumption_btn.clicked.connect(self.update_assumption)
        controls_layout.addWidget(self.update_assumption_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Table
        self.assumptions_table = EditableTableWidget()
        layout.addWidget(self.assumptions_table)
        
        self.tab_widget.addTab(tab, "🎯 Assumptions")
        
    def create_cases_tab(self):
        """Create the case master control tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Header
        header_label = QLabel("Case Master Control - Scenario Modeling")
        header_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(header_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.case_combo = QComboBox()
        self.case_combo.addItems(["Base Case", "Bull Case", "Bear Case"])
        controls_layout.addWidget(QLabel("Case:"))
        controls_layout.addWidget(self.case_combo)
        
        self.run_sensitivity_btn = QPushButton("Run Sensitivity Analysis")
        self.run_sensitivity_btn.clicked.connect(self.run_sensitivity_analysis)
        controls_layout.addWidget(self.run_sensitivity_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Table
        self.cases_table = EditableTableWidget()
        layout.addWidget(self.cases_table)
        
        self.tab_widget.addTab(tab, "📈 Cases")
        
    def create_company_info_tab(self):
        """Create the company information tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Header
        header_label = QLabel("Company Information")
        header_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(header_label)
        
        # Company info display
        self.company_info_text = QTextEdit()
        self.company_info_text.setReadOnly(True)
        self.company_info_text.setMaximumHeight(200)
        layout.addWidget(self.company_info_text)
        
        # Model summary
        summary_label = QLabel("Model Summary")
        summary_label.setStyleSheet("font-size: 12px; font-weight: bold; padding: 5px;")
        layout.addWidget(summary_label)
        
        self.model_summary_text = QTextEdit()
        self.model_summary_text.setReadOnly(True)
        layout.addWidget(self.model_summary_text)
        
        self.tab_widget.addTab(tab, "ℹ️ Company Info")
        
    def create_status_bar(self, parent_layout):
        """Create the status bar."""
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.StyledPanel)
        status_frame.setMaximumHeight(30)
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(5, 2, 5, 2)
        
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        # Progress indicator
        self.progress_label = QLabel("")
        status_layout.addWidget(self.progress_label)
        
        parent_layout.addWidget(status_frame)
        
    def load_company_data(self):
        """Load company data for the specified ticker."""
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Input Error", "Please enter a ticker symbol.")
            return
            
        self.status_label.setText(f"Loading data for {ticker}...")
        self.progress_label.setText("Fetching company info...")
        QApplication.processEvents()
        
        try:
            # Get company info
            company_info = self.modeler.get_company_info(ticker)
            if not company_info:
                QMessageBox.warning(self, "Error", f"Could not load data for {ticker}")
                return
                
            self.current_ticker = ticker
            self.progress_label.setText("Extracting financial data...")
            QApplication.processEvents()
            
            # Extract financial data
            financial_data = self.modeler.extract_financial_data(ticker)
            
            # Load data into tables
            self.load_consolidated_data(financial_data)
            self.load_assumptions_data(ticker)
            self.load_cases_data(ticker)
            self.load_company_info_display(company_info)
            
            # Enable buttons
            self.build_model_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            
            self.status_label.setText(f"Data loaded for {ticker}")
            self.progress_label.setText("Ready")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading data: {str(e)}")
            self.status_label.setText("Error loading data")
            self.progress_label.setText("")
            
    def load_consolidated_data(self, financial_data):
        """Load consolidated financial data into the table."""
        if not financial_data:
            return
            
        # Combine all statements
        consolidated_data = []
        
        # Add income statement
        if 'income_statement' in financial_data and not financial_data['income_statement'].empty:
            income_df = financial_data['income_statement']
            for _, row in income_df.iterrows():
                consolidated_data.append({
                    'Statement': 'Income Statement',
                    'Date': str(row.get('Date', '')),
                    'Metric': 'Revenue',
                    'Value': row.get('Revenue', 0)
                })
                consolidated_data.append({
                    'Statement': 'Income Statement',
                    'Date': str(row.get('Date', '')),
                    'Metric': 'Net Income',
                    'Value': row.get('Net_Income', 0)
                })
        
        # Add balance sheet
        if 'balance_sheet' in financial_data and not financial_data['balance_sheet'].empty:
            balance_df = financial_data['balance_sheet']
            for _, row in balance_df.iterrows():
                consolidated_data.append({
                    'Statement': 'Balance Sheet',
                    'Date': str(row.get('Date', '')),
                    'Metric': 'Total Assets',
                    'Value': row.get('Total_Assets', 0)
                })
                consolidated_data.append({
                    'Statement': 'Balance Sheet',
                    'Date': str(row.get('Date', '')),
                    'Metric': 'Total Equity',
                    'Value': row.get('Total_Equity', 0)
                })
        
        # Add cash flow
        if 'cash_flow' in financial_data and not financial_data['cash_flow'].empty:
            cash_df = financial_data['cash_flow']
            for _, row in cash_df.iterrows():
                consolidated_data.append({
                    'Statement': 'Cash Flow',
                    'Date': str(row.get('Date', '')),
                    'Metric': 'Operating Cash Flow',
                    'Value': row.get('Operating_Cash_Flow', 0)
                })
                consolidated_data.append({
                    'Statement': 'Cash Flow',
                    'Date': str(row.get('Date', '')),
                    'Metric': 'Net Cash Flow',
                    'Value': row.get('Net_Cash_Flow', 0)
                })
        
        if consolidated_data:
            df = pd.DataFrame(consolidated_data)
            self.consolidated_table.load_dataframe(df)
            
    def load_assumptions_data(self, ticker):
        """Load assumptions data into the table."""
        try:
            assumptions_df = self.modeler.create_assumptions_summary(ticker)
            self.assumptions_table.load_dataframe(assumptions_df)
        except Exception as e:
            print(f"Error loading assumptions: {e}")
            
    def load_cases_data(self, ticker):
        """Load cases data into the table."""
        try:
            cases_df = self.modeler.create_case_master_control(ticker)
            self.cases_table.load_dataframe(cases_df)
        except Exception as e:
            print(f"Error loading cases: {e}")
            
    def load_company_info_display(self, company_info):
        """Load company information into the display."""
        info_text = f"""
Company: {company_info.get('name', 'Unknown')}
Ticker: {company_info.get('ticker', 'Unknown')}
Sector: {company_info.get('sector', 'Unknown')}
Industry: {company_info.get('industry', 'Unknown')}
Market Cap: ${company_info.get('market_cap', 0):,.0f}
Enterprise Value: ${company_info.get('enterprise_value', 0):,.0f}
Shares Outstanding: {company_info.get('shares_outstanding', 0):,.0f}
        """
        self.company_info_text.setText(info_text.strip())
        
    def build_financial_model(self):
        """Build the complete financial model."""
        if not self.current_ticker:
            QMessageBox.warning(self, "Error", "Please load company data first.")
            return
            
        self.status_label.setText("Building financial model...")
        self.progress_label.setText("Processing...")
        QApplication.processEvents()
        
        try:
            # Build the model
            model_components = self.modeler.build_financial_model(
                self.current_ticker, 
                save_to_excel=False  # We'll handle saving separately
            )
            
            # Update model summary
            summary = self.modeler.get_model_summary(self.current_ticker)
            summary_text = f"""
Model Summary for {self.current_ticker}:
- Company: {summary.get('company_name', 'Unknown')}
- Model Date: {summary.get('model_date', 'Unknown')}
- Consolidated Statements: {summary.get('consolidated_statements_rows', 0)} rows
- Assumptions: {summary.get('assumptions_count', 0)} variables
- Cases: {summary.get('cases_count', 0)} scenarios
- Available Cases: {', '.join(summary.get('available_cases', []))}
            """
            self.model_summary_text.setText(summary_text.strip())
            
            self.status_label.setText("Financial model built successfully")
            self.progress_label.setText("Ready")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error building model: {str(e)}")
            self.status_label.setText("Error building model")
            self.progress_label.setText("")
            
    def save_to_excel(self):
        """Save the current model to Excel."""
        if not self.current_ticker:
            QMessageBox.warning(self, "Error", "Please load company data first.")
            return
            
        try:
            filename = f"{self.current_ticker}_Financial_Model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            self.modeler.save_to_excel(self.current_ticker, filename)
            
            QMessageBox.information(self, "Success", f"Model saved to {filename}")
            self.status_label.setText(f"Model saved to {filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving model: {str(e)}")
            
    def refresh_data(self):
        """Refresh the current data."""
        if self.current_ticker:
            self.load_company_data()
        else:
            QMessageBox.information(self, "Info", "No data to refresh. Please load company data first.")
            
    def update_assumption(self):
        """Update a specific assumption."""
        if not self.current_ticker:
            QMessageBox.warning(self, "Error", "Please load company data first.")
            return
            
        variable = self.assumption_variable_combo.currentText()
        case = self.assumption_case_combo.currentText()
        value = self.assumption_value_input.value()
        
        try:
            self.modeler.update_assumption(self.current_ticker, variable, case, value)
            
            # Refresh assumptions table
            self.load_assumptions_data(self.current_ticker)
            
            self.status_label.setText(f"Updated {variable} for {case}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error updating assumption: {str(e)}")
            
    def run_sensitivity_analysis(self):
        """Run sensitivity analysis for the selected case."""
        if not self.current_ticker:
            QMessageBox.warning(self, "Error", "Please load company data first.")
            return
            
        case = self.case_combo.currentText()
        
        # For now, run a simple sensitivity analysis
        try:
            sensitivity_results = self.modeler.run_sensitivity_analysis(
                self.current_ticker, "Revenue_Growth_Rate", 0.10
            )
            
            # Show results in a dialog
            dialog = SensitivityResultsDialog(sensitivity_results, self)
            dialog.exec_()
            
            self.status_label.setText("Sensitivity analysis completed")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error running sensitivity analysis: {str(e)}")

class SensitivityResultsDialog(QDialog):
    """Dialog to display sensitivity analysis results."""
    
    def __init__(self, results_df, parent=None):
        super().__init__(parent)
        self.results_df = results_df
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the dialog UI."""
        self.setWindowTitle("Sensitivity Analysis Results")
        self.setModal(True)
        self.resize(600, 400)
        
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("Sensitivity Analysis Results")
        header_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(header_label)
        
        # Results table
        self.results_table = EditableTableWidget()
        self.results_table.load_dataframe(self.results_df)
        layout.addWidget(self.results_table)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Save)
        button_box.accepted.connect(self.accept)
        button_box.button(QDialogButtonBox.Save).clicked.connect(self.save_results)
        layout.addWidget(button_box)
        
    def save_results(self):
        """Save results to CSV."""
        try:
            filename = f"sensitivity_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.results_df.to_csv(filename, index=False)
            QMessageBox.information(self, "Success", f"Results saved to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving results: {str(e)}")

# Test function
def test_financial_modeler_widget():
    """Test the financial modeler widget."""
    app = QApplication(sys.argv)
    
    widget = FinancialModelerWidget()
    widget.resize(1200, 800)
    widget.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    test_financial_modeler_widget() 