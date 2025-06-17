import sys
import os
from datetime import datetime, timedelta, timezone
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QComboBox, QSpinBox,
    QTabWidget, QProgressBar, QMessageBox, QSplitter, QLineEdit,
    QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
import pyqtgraph as pg
import pandas as pd
import yfinance as yf
from qt_material import apply_stylesheet

# Add parent directory to path to import from orchestrator
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from _2_Orchestrator_And_ML_Python.model_trainer import ModelTrainer
from _2_Orchestrator_And_ML_Python.model_bridge import ModelBridge

class StockChartWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('w')  # White background
        self.showGrid(x=True, y=True)
        self.setLabel('left', 'Price')
        self.setLabel('bottom', 'Date')
        
        # Enable mouse interaction
        self.setMouseEnabled(x=True, y=True)
        self.showButtons()
        
    def plot_stock_data(self, df):
        self.clear()
        
        if df.empty:
            print("No data to plot")
            return
            
        print(f"Plotting {len(df)} data points")
        
        # Plot candlesticks
        for i in range(len(df)):
            # Calculate candlestick coordinates
            x = i
            open_price = df['Open'].iloc[i]
            close_price = df['Close'].iloc[i]
            high_price = df['High'].iloc[i]
            low_price = df['Low'].iloc[i]
            
            # Plot body
            if close_price >= open_price:
                color = 'g'  # Green for bullish
            else:
                color = 'r'  # Red for bearish
                
            # Plot candlestick
            self.plot([x, x], [low_price, high_price], pen=pg.mkPen(color=color, width=1))
            self.plot([x-0.2, x+0.2, x+0.2, x-0.2, x-0.2],
                     [open_price, open_price, close_price, close_price, open_price],
                     pen=pg.mkPen(color=color, width=1),
                     fillLevel=0,
                     brush=pg.mkBrush(color))
        
        # Add price range info
        min_price = df['Low'].min()
        max_price = df['High'].max()
        current_price = df['Close'].iloc[-1]
        
        # Set proper axis ranges
        price_padding = (max_price - min_price) * 0.05  # 5% padding
        self.setYRange(max(0, min_price - price_padding), max_price + price_padding)
        self.setXRange(0, len(df) - 1)
        
        # Set axis labels with more info
        self.setLabel('left', f'Price (${min_price:.2f} - ${max_price:.2f})')
        self.setLabel('bottom', f'Days ({len(df)} data points)')
        
        # Add current price line
        self.addLine(y=current_price, pen=pg.mkPen('b', width=2, style=pg.QtCore.Qt.DashLine))
        
        print(f"Chart updated - Price range: ${min_price:.2f} to ${max_price:.2f}, Current: ${current_price:.2f}")

class TrainingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, model_trainer, epochs):
        super().__init__()
        self.model_trainer = model_trainer
        self.epochs = epochs
        
    def run(self):
        try:
            self.model_trainer.train_model(epochs=self.epochs, 
                                         progress_callback=lambda x: self.progress.emit(x))
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network Trading System (PyQt5)")
        self.setMinimumSize(1200, 800)
        
        # Initialize model components
        self.model_trainer = ModelTrainer()
        self.model_bridge = ModelBridge()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Add padding around the window edges
        layout.setContentsMargins(10, 10, 10, 10)  # Left, Top, Right, Bottom padding
        layout.setSpacing(10)  # Add spacing between widgets
        
        # Create scrollable area for entire content
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create content widget that will be inside the scroll area
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.content_layout.setSpacing(15)
        
        # Create tab widget
        tabs = QTabWidget()
        self.content_layout.addWidget(tabs)
        
        # Create training tab
        training_tab = QWidget()
        tabs.addTab(training_tab, "Training")
        self.setup_training_tab(training_tab)
        
        # Create prediction tab
        prediction_tab = QWidget()
        tabs.addTab(prediction_tab, "Prediction")
        self.setup_prediction_tab(prediction_tab)
        
        # Set the content widget as the scroll area's widget
        self.scroll_area.setWidget(self.content_widget)
        layout.addWidget(self.scroll_area)
        
        # Add status bar
        self.statusBar().showMessage("Use scroll if content exceeds window bounds. F11: Fullscreen, F5: Refresh data.")
        
        # Apply material design theme
        apply_stylesheet(self, theme='light_blue.xml')
        
    def setup_training_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Add padding to the tab layout as well
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Top controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)  # Add spacing between controls
        
        # Stock selection
        self.stock_combo = QComboBox()
        
        # Import stock selection utilities
        try:
            import sys
            from pathlib import Path
            REPO_ROOT = Path(__file__).parent.parent.parent
            ORCHESTRATOR_PATH = REPO_ROOT / "_2_Orchestrator_And_ML_Python"
            sys.path.append(str(ORCHESTRATOR_PATH))
            
            from stock_selection_utils import StockSelectionManager
            self.stock_manager = StockSelectionManager()
            
            # Add special options first
            self.stock_combo.addItem("🎲 Random Pick", "random")
            self.stock_combo.addItem("🚀 Optimized Pick", "optimized")
            self.stock_combo.addItem("✏️ Custom Ticker", "custom")
            self.stock_combo.insertSeparator(3)
            
            # Add S&P100 stocks sorted by market cap
            stock_options = self.stock_manager.get_stock_options()
            for option in stock_options[3:]:  # Skip the first 3 special options
                display_name = f"{option['symbol']} - {option['description']}"
                self.stock_combo.addItem(display_name, option['symbol'])
                
        except Exception as e:
            print(f"Error loading stock selection utilities: {e}")
            # Fallback to basic stocks
            self.stock_combo.addItems(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'])
            self.stock_manager = None
        
        controls_layout.addWidget(QLabel("Stock:"))
        controls_layout.addWidget(self.stock_combo)
        
        # Custom ticker input (initially hidden)
        self.custom_ticker_input = QLineEdit()
        self.custom_ticker_input.setPlaceholderText("Enter ticker symbol...")
        self.custom_ticker_input.setVisible(False)
        controls_layout.addWidget(self.custom_ticker_input)
        
        # Connect stock selection change
        self.stock_combo.currentTextChanged.connect(self.on_stock_selection_changed)
        
        # Date range
        self.date_range = QSpinBox()
        self.date_range.setRange(1, 365)
        self.date_range.setValue(30)
        controls_layout.addWidget(QLabel("Days of History:"))
        controls_layout.addWidget(self.date_range)
        
        # Get data button
        self.get_data_btn = QPushButton("Get Stock Data")
        self.get_data_btn.clicked.connect(self.load_stock_data)
        controls_layout.addWidget(self.get_data_btn)
        
        layout.addLayout(controls_layout)
        
        # Splitter for chart and controls
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #cccccc;
                border: 1px solid #999999;
            }
            QSplitter::handle:hover {
                background-color: #aaaaaa;
            }
        """)
        layout.addWidget(splitter)
        
        # Chart widget
        self.chart = StockChartWidget()
        self.chart.setMinimumHeight(300)  # Minimum height for chart
        splitter.addWidget(self.chart)
        
        # Chart control buttons
        chart_controls = QHBoxLayout()
        chart_controls.setSpacing(10)
        
        self.maximize_chart_btn = QPushButton("📈 Max Chart")
        self.maximize_chart_btn.setToolTip("Maximize chart area (80% chart, 20% controls)")
        self.maximize_chart_btn.clicked.connect(lambda: self.adjust_splitter(splitter, 0.8))
        chart_controls.addWidget(self.maximize_chart_btn)
        
        self.minimize_chart_btn = QPushButton("📊 Min Chart")
        self.minimize_chart_btn.setToolTip("Minimize chart area (40% chart, 60% controls)")
        self.minimize_chart_btn.clicked.connect(lambda: self.adjust_splitter(splitter, 0.4))
        chart_controls.addWidget(self.minimize_chart_btn)
        
        self.reset_splitter_btn = QPushButton("🔄 Reset")
        self.reset_splitter_btn.setToolTip("Reset to default proportions (70% chart, 30% controls)")
        self.reset_splitter_btn.clicked.connect(lambda: self.adjust_splitter(splitter, 0.7))
        chart_controls.addWidget(self.reset_splitter_btn)
        
        chart_controls.addStretch()
        
        # Add chart controls to a widget
        chart_controls_widget = QWidget()
        chart_controls_widget.setLayout(chart_controls)
        splitter.addWidget(chart_controls_widget)
        chart_controls_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Trading decision controls
        decision_widget = QWidget()
        decision_layout = QVBoxLayout(decision_widget)
        decision_layout.setContentsMargins(10, 10, 10, 10)
        decision_layout.setSpacing(8)
        
        # Decision buttons
        decision_buttons = QHBoxLayout()
        decision_buttons.setSpacing(10)
        self.buy_btn = QPushButton("BUY")
        self.sell_btn = QPushButton("SELL")
        self.hold_btn = QPushButton("HOLD")
        
        for btn in [self.buy_btn, self.sell_btn, self.hold_btn]:
            btn.setMinimumHeight(40)
            decision_buttons.addWidget(btn)
            btn.clicked.connect(lambda checked, b=btn: self.make_decision(b.text()))
            
        decision_layout.addLayout(decision_buttons)
        
        # Reasoning input
        decision_layout.addWidget(QLabel("Trading Reasoning:"))
        self.reasoning_input = QTextEdit()
        self.reasoning_input.setPlaceholderText("Explain your trading decision...")
        self.reasoning_input.setMaximumHeight(100)
        decision_layout.addWidget(self.reasoning_input)
        
        # Submit button
        self.submit_btn = QPushButton("Submit Decision")
        self.submit_btn.clicked.connect(self.submit_decision)
        decision_layout.addWidget(self.submit_btn)
        
        splitter.addWidget(decision_widget)
        
        # Training controls
        training_controls = QHBoxLayout()
        training_controls.setSpacing(10)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(10)
        training_controls.addWidget(QLabel("Training Epochs:"))
        training_controls.addWidget(self.epochs_spin)
        
        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self.start_training)
        training_controls.addWidget(self.train_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        training_controls.addWidget(self.progress_bar)
        
        layout.addLayout(training_controls)
        
    def setup_prediction_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Add padding to the prediction tab layout as well
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Similar controls to training tab but for prediction
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        
        self.pred_stock_combo = QComboBox()
        
        # Import stock selection utilities for prediction tab
        try:
            import sys
            from pathlib import Path
            REPO_ROOT = Path(__file__).parent.parent.parent
            ORCHESTRATOR_PATH = REPO_ROOT / "_2_Orchestrator_And_ML_Python"
            sys.path.append(str(ORCHESTRATOR_PATH))
            
            from stock_selection_utils import StockSelectionManager
            self.pred_stock_manager = StockSelectionManager()
            
            # Add special options first
            self.pred_stock_combo.addItem("🎲 Random Pick", "random")
            self.pred_stock_combo.addItem("🚀 Optimized Pick", "optimized")
            self.pred_stock_combo.addItem("✏️ Custom Ticker", "custom")
            self.pred_stock_combo.insertSeparator(3)
            
            # Add S&P100 stocks sorted by market cap
            stock_options = self.pred_stock_manager.get_stock_options()
            for option in stock_options[3:]:  # Skip the first 3 special options
                display_name = f"{option['symbol']} - {option['description']}"
                self.pred_stock_combo.addItem(display_name, option['symbol'])
                
        except Exception as e:
            print(f"Error loading stock selection utilities for prediction: {e}")
            # Fallback to basic stocks
            self.pred_stock_combo.addItems(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'])
            self.pred_stock_manager = None
        
        controls_layout.addWidget(QLabel("Stock:"))
        controls_layout.addWidget(self.pred_stock_combo)
        
        # Custom ticker input for prediction (initially hidden)
        self.pred_custom_ticker_input = QLineEdit()
        self.pred_custom_ticker_input.setPlaceholderText("Enter ticker symbol...")
        self.pred_custom_ticker_input.setVisible(False)
        controls_layout.addWidget(self.pred_custom_ticker_input)
        
        # Connect stock selection change for prediction
        self.pred_stock_combo.currentTextChanged.connect(self.on_pred_stock_selection_changed)
        
        self.pred_btn = QPushButton("Get Prediction")
        self.pred_btn.clicked.connect(self.get_prediction)
        controls_layout.addWidget(self.pred_btn)
        
        layout.addLayout(controls_layout)
        
        # Prediction result
        self.prediction_label = QLabel("Prediction will appear here...")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet("font-size: 18px; padding: 20px;")
        layout.addWidget(self.prediction_label)
        
    def on_stock_selection_changed(self, text):
        """Handle stock selection changes"""
        try:
            # Get the current data (symbol) from the combo box
            current_data = self.stock_combo.currentData()
            
            if current_data == "random":
                # Get random stock
                if self.stock_manager:
                    symbol, name = self.stock_manager.get_random_stock()
                    # Update the combo box to show the selected stock
                    for i in range(self.stock_combo.count()):
                        if self.stock_combo.itemData(i) == symbol:
                            self.stock_combo.setCurrentIndex(i)
                            break
                    QMessageBox.information(self, "Random Pick", f"🎲 Random pick: {symbol} ({name})")
                self.custom_ticker_input.setVisible(False)
                
            elif current_data == "optimized":
                # Get optimized pick
                if self.stock_manager:
                    symbol, name = self.stock_manager.get_optimized_pick()
                    # Update the combo box to show the selected stock
                    for i in range(self.stock_combo.count()):
                        if self.stock_combo.itemData(i) == symbol:
                            self.stock_combo.setCurrentIndex(i)
                            break
                    QMessageBox.information(self, "Optimized Pick", f"🚀 Optimized pick: {symbol} ({name})")
                self.custom_ticker_input.setVisible(False)
                
            elif current_data == "custom":
                # Show custom ticker input
                self.custom_ticker_input.setVisible(True)
                self.custom_ticker_input.setFocus()
                
            else:
                # Regular stock selected
                self.custom_ticker_input.setVisible(False)
                
        except Exception as e:
            print(f"Error in stock selection change: {e}")
            self.custom_ticker_input.setVisible(False)

    def load_stock_data(self):
        try:
            # Get the selected stock symbol
            stock_data = self.stock_combo.currentData()
            
            # Validate stock_data
            if stock_data is None:
                QMessageBox.warning(self, "Warning", "Please select a valid stock")
                return
            
            # Handle special cases
            if stock_data == "custom":
                stock = self.custom_ticker_input.text().strip().upper()
                if not stock:
                    QMessageBox.warning(self, "Warning", "Please enter a ticker symbol")
                    return
                
                # Validate custom ticker
                if self.stock_manager:
                    is_valid, message, company_name = self.stock_manager.validate_custom_ticker(stock)
                    if not is_valid:
                        QMessageBox.warning(self, "Invalid Ticker", message)
                        return
            elif stock_data == "random":
                # Get a random stock
                if self.stock_manager:
                    symbol, name = self.stock_manager.get_random_stock()
                    stock = symbol
                    print(f"Random stock selected: {stock} ({name})")
                else:
                    QMessageBox.warning(self, "Warning", "Stock manager not available")
                    return
            elif stock_data == "optimized":
                # Get an optimized stock
                if self.stock_manager:
                    symbol, name = self.stock_manager.get_optimized_pick()
                    stock = symbol
                    print(f"Optimized stock selected: {stock} ({name})")
                else:
                    QMessageBox.warning(self, "Warning", "Stock manager not available")
                    return
            else:
                stock = stock_data
            
            # Additional validation for stock symbol
            if not stock or not isinstance(stock, str) or not stock.strip():
                QMessageBox.warning(self, "Warning", "Invalid stock symbol")
                return
            
            days = self.date_range.value()
            
            # Import the date range utilities
            import sys
            from pathlib import Path
            REPO_ROOT = Path(__file__).parent.parent.parent
            ORCHESTRATOR_PATH = REPO_ROOT / "_2_Orchestrator_And_ML_Python"
            sys.path.append(str(ORCHESTRATOR_PATH))
            
            from date_range_utils import find_available_data_range, validate_date_range
            
            # Get random date range within the last 25 years
            start_date, end_date = find_available_data_range(stock, days, max_years_back=25)
            
            # Validate the date range
            if not validate_date_range(start_date, end_date, stock):
                QMessageBox.critical(self, "Error", f"Invalid date range generated for {stock}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                return
            
            print(f"Fetching {days} days of data for {stock} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (random range within last 25 years)")
            
            ticker = yf.Ticker(stock)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                QMessageBox.warning(self, "Error", f"No data found for {stock}. Please try a different stock or fewer days.")
                return
            
            # Ensure df.index is timezone-aware (UTC)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
            
            # Check if we got the expected amount of data
            if len(df) < days * 0.8:  # Allow 20% tolerance for weekends/holidays
                print(f"Warning: Got {len(df)} days of data for {stock}, expected around {days} days")
            
            # Validate that data is not newer than our end_date
            if df.index.max() > end_date:
                QMessageBox.warning(self, "Warning", f"Data for {stock} contains dates newer than expected. This may indicate a system clock issue.")
                # Filter out dates newer than our end_date
                df = df[df.index <= end_date]
                if df.empty:
                    QMessageBox.warning(self, "Error", f"No valid data found for {stock} after filtering dates.")
                    return
                
            print(f"Successfully loaded {len(df)} days of data for {stock}")
            print(f"Data range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
                
            # Plot data
            self.chart.plot_stock_data(df)
            
            # Store data for training
            self.current_data = df
            
            QMessageBox.information(self, "Success", f"Loaded {len(df)} days of data for {stock}\nDate range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}\n\nTry:\n- Different stock symbol\n- Fewer days of history\n- Check internet connection\n- Check system clock")
    
    def make_decision(self, decision):
        # Highlight the selected button
        for btn in [self.buy_btn, self.sell_btn, self.hold_btn]:
            btn.setStyleSheet("")
        
        sender = self.sender()
        if sender:
            sender.setStyleSheet("background-color: #4CAF50; color: white;")
        
        self.current_decision = decision
    
    def submit_decision(self):
        if not hasattr(self, 'current_data') or not hasattr(self, 'current_decision'):
            QMessageBox.warning(self, "Error", "Please load data and make a decision first")
            return
            
        reasoning = self.reasoning_input.toPlainText()
        if not reasoning.strip():
            QMessageBox.warning(self, "Error", "Please provide reasoning for your decision")
            return
            
        try:
            # Here you would integrate with your model trainer
            # For now, just show a success message
            QMessageBox.information(self, "Success", 
                                  f"Decision submitted: {self.current_decision}\nReasoning: {reasoning}")
            
            # Clear the form
            self.reasoning_input.clear()
            for btn in [self.buy_btn, self.sell_btn, self.hold_btn]:
                btn.setStyleSheet("")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to submit decision: {str(e)}")
    
    def start_training(self):
        if not hasattr(self, 'current_data'):
            QMessageBox.warning(self, "Error", "Please load stock data first")
            return
            
        epochs = self.epochs_spin.value()
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.train_btn.setEnabled(False)
        
        # Start training in background thread
        self.training_thread = TrainingThread(self.model_trainer, epochs)
        self.training_thread.progress.connect(self.update_progress)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.error.connect(self.training_error)
        self.training_thread.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def training_finished(self):
        self.progress_bar.setVisible(False)
        self.train_btn.setEnabled(True)
        QMessageBox.information(self, "Success", "Training completed!")
    
    def training_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.train_btn.setEnabled(True)
        QMessageBox.critical(self, "Training Error", f"Training failed: {error_msg}")
    
    def on_pred_stock_selection_changed(self, text):
        """Handle prediction stock selection changes"""
        try:
            # Get the current data (symbol) from the combo box
            current_data = self.pred_stock_combo.currentData()
            
            if current_data == "random":
                # Get random stock
                if self.pred_stock_manager:
                    symbol, name = self.pred_stock_manager.get_random_stock()
                    # Update the combo box to show the selected stock
                    for i in range(self.pred_stock_combo.count()):
                        if self.pred_stock_combo.itemData(i) == symbol:
                            self.pred_stock_combo.setCurrentIndex(i)
                            break
                    QMessageBox.information(self, "Random Pick", f"🎲 Random pick: {symbol} ({name})")
                self.pred_custom_ticker_input.setVisible(False)
                
            elif current_data == "optimized":
                # Get optimized pick
                if self.pred_stock_manager:
                    symbol, name = self.pred_stock_manager.get_optimized_pick()
                    # Update the combo box to show the selected stock
                    for i in range(self.pred_stock_combo.count()):
                        if self.pred_stock_combo.itemData(i) == symbol:
                            self.pred_stock_combo.setCurrentIndex(i)
                            break
                    QMessageBox.information(self, "Optimized Pick", f"🚀 Optimized pick: {symbol} ({name})")
                self.pred_custom_ticker_input.setVisible(False)
                
            elif current_data == "custom":
                # Show custom ticker input
                self.pred_custom_ticker_input.setVisible(True)
                self.pred_custom_ticker_input.setFocus()
                
            else:
                # Regular stock selected
                self.pred_custom_ticker_input.setVisible(False)
                
        except Exception as e:
            print(f"Error in prediction stock selection change: {e}")
            self.pred_custom_ticker_input.setVisible(False)

    def get_prediction(self):
        try:
            # Get the selected stock symbol
            stock_data = self.pred_stock_combo.currentData()
            
            # Validate stock_data
            if stock_data is None:
                self.prediction_label.setText("Please select a valid stock")
                return
            
            # Handle special cases
            if stock_data == "custom":
                stock = self.pred_custom_ticker_input.text().strip().upper()
                if not stock:
                    self.prediction_label.setText("Please enter a ticker symbol")
                    return
                
                # Validate custom ticker
                if self.pred_stock_manager:
                    is_valid, message, company_name = self.pred_stock_manager.validate_custom_ticker(stock)
                    if not is_valid:
                        self.prediction_label.setText(f"Invalid ticker: {message}")
                        return
            elif stock_data == "random":
                # Get a random stock
                if self.pred_stock_manager:
                    symbol, name = self.pred_stock_manager.get_random_stock()
                    stock = symbol
                    print(f"Random stock selected for prediction: {stock} ({name})")
                else:
                    self.prediction_label.setText("Stock manager not available")
                    return
            elif stock_data == "optimized":
                # Get an optimized stock
                if self.pred_stock_manager:
                    symbol, name = self.pred_stock_manager.get_optimized_pick()
                    stock = symbol
                    print(f"Optimized stock selected for prediction: {stock} ({name})")
                else:
                    self.prediction_label.setText("Stock manager not available")
                    return
            else:
                stock = stock_data
            
            # Additional validation for stock symbol
            if not stock or not isinstance(stock, str) or not stock.strip():
                self.prediction_label.setText("Invalid stock symbol")
                return
            
            # Import the date range utilities
            import sys
            from pathlib import Path
            REPO_ROOT = Path(__file__).parent.parent.parent
            ORCHESTRATOR_PATH = REPO_ROOT / "_2_Orchestrator_And_ML_Python"
            sys.path.append(str(ORCHESTRATOR_PATH))
            
            from date_range_utils import find_available_data_range, validate_date_range
            
            # Get recent data for prediction (last 30 days)
            start_date, end_date = find_available_data_range(stock, 30, max_years_back=25)
            
            # Validate the date range
            if not validate_date_range(start_date, end_date, stock):
                self.prediction_label.setText(f"Invalid date range for {stock}")
                return
            
            print(f"Getting prediction for {stock} using data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            ticker = yf.Ticker(stock)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                self.prediction_label.setText(f"No data found for {stock}")
                return
            
            # Ensure df.index is timezone-aware (UTC)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
            
            # Check if we got the expected amount of data
            if len(df) < 24:  # Allow tolerance for weekends/holidays
                print(f"Warning: Got {len(df)} days of data for {stock}, expected around 30 days")
            
            # Validate that data is not newer than our end_date
            if df.index.max() > end_date:
                print(f"Warning: Data for {stock} contains dates newer than expected, filtering...")
                df = df[df.index <= end_date]
                if df.empty:
                    self.prediction_label.setText(f"No valid data found for {stock} after filtering dates")
                    return
                
            print(f"Prediction data range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
                
            # Calculate some basic metrics
            current_price = df['Close'].iloc[-1]
            price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) > 1 else 0
            price_change_pct = (price_change / df['Close'].iloc[-2]) * 100 if len(df) > 1 and df['Close'].iloc[-2] != 0 else 0
            
            # Simple prediction logic (you can enhance this later)
            if price_change > 0:
                signal = "BUY"
                confidence = min(0.8, 0.5 + abs(price_change_pct) / 100)
            elif price_change < 0:
                signal = "SELL" 
                confidence = min(0.8, 0.5 + abs(price_change_pct) / 100)
            else:
                signal = "HOLD"
                confidence = 0.6
                
            prediction_text = f"""
Prediction for {stock}:
Signal: {signal}
Confidence: {confidence:.1%}
Current Price: ${current_price:.2f}
Price Change: ${price_change:.2f} ({price_change_pct:+.2f}%)
Data Points: {len(df)} days
Data Range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}
            """.strip()
            
            self.prediction_label.setText(prediction_text)
            
        except Exception as e:
            print(f"Error getting prediction: {str(e)}")
            self.prediction_label.setText(f"Error getting prediction for {stock}: {str(e)}")

    def resizeEvent(self, event):
        """Handle window resize events to update status bar"""
        super().resizeEvent(event)
        # Update status bar with current window size and scroll info
        width = self.width()
        height = self.height()
        scroll_info = ""
        
        if hasattr(self, 'scroll_area'):
            scroll_area = self.scroll_area
            if scroll_area.verticalScrollBar().isVisible():
                scroll_info = " (Vertical scroll available)"
            elif scroll_area.horizontalScrollBar().isVisible():
                scroll_info = " (Horizontal scroll available)"
        
        self.statusBar().showMessage(
            f"Window: {width}x{height} | "
            f"Use scroll if content exceeds window bounds{scroll_info} | "
            f"F11: Fullscreen, F5: Refresh data"
        )

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for better user experience"""
        if event.key() == Qt.Key_F11:
            # Toggle fullscreen
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif event.key() == Qt.Key_F5:
            # Refresh data
            if hasattr(self, 'current_data'):
                self.load_stock_data()
        elif event.key() == Qt.Key_Escape:
            # Exit fullscreen
            if self.isFullScreen():
                self.showNormal()
        else:
            super().keyPressEvent(event)

    def adjust_splitter(self, splitter, ratio):
        """Adjust the splitter to the specified ratio"""
        if splitter.count() > 1:
            total_height = splitter.height()
            chart_height = int(total_height * ratio)
            controls_height = total_height - chart_height
            splitter.setSizes([chart_height, controls_height])

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Neural Network Trading System")
    app.setApplicationVersion("1.0")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 