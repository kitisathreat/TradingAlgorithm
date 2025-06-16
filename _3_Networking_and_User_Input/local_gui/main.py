import sys
import os
from datetime import datetime, timedelta, timezone
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QComboBox, QSpinBox,
    QTabWidget, QProgressBar, QMessageBox, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
import pyqtgraph as pg
import pandas as pd
import yfinance as yf
from qt_material import apply_stylesheet

# Add parent directory to path to import from orchestrator
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from _2_Orchestrator_And_ML_Python.interactive_training_app.backend.model_trainer import ModelTrainer

# Create a simple ModelBridge class for now
class ModelBridge:
    def __init__(self):
        pass
    
    def get_trading_decision(self, symbol, features, sentiment_data, vix, account_value, current_position):
        # Placeholder implementation
        return {
            'signal': 'HOLD',
            'confidence': 0.65,
            'position_size': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'reasoning': 'Model not yet trained'
        }

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
        self.setWindowTitle("Neural Network Trading System")
        self.setMinimumSize(1200, 800)
        
        # Initialize model components
        self.model_trainer = ModelTrainer()
        self.model_bridge = ModelBridge()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Create training tab
        training_tab = QWidget()
        tabs.addTab(training_tab, "Training")
        self.setup_training_tab(training_tab)
        
        # Create prediction tab
        prediction_tab = QWidget()
        tabs.addTab(prediction_tab, "Prediction")
        self.setup_prediction_tab(prediction_tab)
        
        # Apply material design theme
        apply_stylesheet(self, theme='light_blue.xml')
        
    def setup_training_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        # Stock selection
        self.stock_combo = QComboBox()
        self.stock_combo.addItems([
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 
            'AMD', 'INTC', 'CRM', 'ADBE', 'PYPL', 'UBER', 'SHOP', 'SQ',
            'SPY', 'QQQ', 'IWM', 'VTI'  # ETFs for more reliable data
        ])
        controls_layout.addWidget(QLabel("Stock:"))
        controls_layout.addWidget(self.stock_combo)
        
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
        layout.addWidget(splitter)
        
        # Chart widget
        self.chart = StockChartWidget()
        splitter.addWidget(self.chart)
        
        # Trading decision controls
        decision_widget = QWidget()
        decision_layout = QVBoxLayout(decision_widget)
        
        # Decision buttons
        decision_buttons = QHBoxLayout()
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
        decision_layout.addWidget(self.reasoning_input)
        
        # Submit button
        self.submit_btn = QPushButton("Submit Decision")
        self.submit_btn.clicked.connect(self.submit_decision)
        decision_layout.addWidget(self.submit_btn)
        
        splitter.addWidget(decision_widget)
        
        # Training controls
        training_controls = QHBoxLayout()
        
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
        
        # Similar controls to training tab but for prediction
        controls_layout = QHBoxLayout()
        
        self.pred_stock_combo = QComboBox()
        self.pred_stock_combo.addItems([
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 
            'AMD', 'INTC', 'CRM', 'ADBE', 'PYPL', 'UBER', 'SHOP', 'SQ',
            'SPY', 'QQQ', 'IWM', 'VTI'  # ETFs for more reliable data
        ])
        controls_layout.addWidget(QLabel("Stock:"))
        controls_layout.addWidget(self.pred_stock_combo)
        
        self.pred_btn = QPushButton("Get Prediction")
        self.pred_btn.clicked.connect(self.get_prediction)
        controls_layout.addWidget(self.pred_btn)
        
        layout.addLayout(controls_layout)
        
        # Prediction result
        self.prediction_label = QLabel("Prediction will appear here...")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet("font-size: 18px; padding: 20px;")
        layout.addWidget(self.prediction_label)
        
    def load_stock_data(self):
        try:
            stock = self.stock_combo.currentText()
            days = self.date_range.value()
            
            # Use a reference date that yfinance should definitely have data for
            # Since today is June 16, 2025, let's use a date from 2024 that we know exists
            reference_date = datetime(2024, 12, 20, tzinfo=timezone.utc)  # December 20, 2024 - should have data
            end_date = reference_date
            start_date = end_date - timedelta(days=days)
            
            # Validate dates to prevent future date issues
            if start_date >= end_date:
                QMessageBox.critical(self, "Error", f"Invalid date range: start_date {start_date.strftime('%Y-%m-%d')} >= end_date {end_date.strftime('%Y-%m-%d')}")
                return
            
            print(f"Fetching {days} days of data for {stock} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (using 2024 reference date)")
            
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
    
    def get_prediction(self):
        stock = self.pred_stock_combo.currentText()
        try:
            # Use a reference date that yfinance should definitely have data for
            # Since today is June 16, 2025, let's use a date from 2024 that we know exists
            reference_date = datetime(2024, 12, 20, tzinfo=timezone.utc)  # December 20, 2024 - should have data
            end_date = reference_date
            start_date = end_date - timedelta(days=30)  # Last 30 days from the reference date
            
            # Validate dates
            if start_date >= end_date:
                self.prediction_label.setText(f"Error: Invalid date range for {stock}")
                return
            
            print(f"Fetching prediction data for {stock} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (using 2024 reference date)")
            
            ticker = yf.Ticker(stock)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                self.prediction_label.setText(f"No data available for {stock}")
                return
            
            # Ensure df.index is timezone-aware (UTC)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
            
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