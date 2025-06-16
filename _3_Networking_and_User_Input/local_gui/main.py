import sys
import os
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QComboBox, QSpinBox,
    QTabWidget, QProgressBar, QMessageBox, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
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
        self.stock_combo.addItems(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'])  # Add more stocks
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
        splitter = QSplitter(Qt.Orientation.Vertical)
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
        self.pred_stock_combo.addItems(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'])
        controls_layout.addWidget(QLabel("Stock:"))
        controls_layout.addWidget(self.pred_stock_combo)
        
        self.predict_btn = QPushButton("Get Prediction")
        self.predict_btn.clicked.connect(self.get_prediction)
        controls_layout.addWidget(self.predict_btn)
        
        layout.addLayout(controls_layout)
        
        # Prediction display
        self.prediction_display = QTextEdit()
        self.prediction_display.setReadOnly(True)
        layout.addWidget(self.prediction_display)
        
    def load_stock_data(self):
        try:
            symbol = self.stock_combo.currentText()
            days = self.date_range.value()
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get stock data
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                QMessageBox.warning(self, "Error", "No data available for the selected period")
                return
                
            # Plot the data
            self.chart.plot_stock_data(df)
            
            # Store the data for later use
            self.current_data = df
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load stock data: {str(e)}")
            
    def make_decision(self, decision):
        self.current_decision = decision
        # Highlight the selected button
        for btn in [self.buy_btn, self.sell_btn, self.hold_btn]:
            btn.setStyleSheet("")
        sender = self.sender()
        sender.setStyleSheet("background-color: #4CAF50; color: white;")
        
    def submit_decision(self):
        if not hasattr(self, 'current_data') or not hasattr(self, 'current_decision'):
            QMessageBox.warning(self, "Error", "Please load stock data and make a decision first")
            return
            
        reasoning = self.reasoning_input.toPlainText()
        if not reasoning:
            QMessageBox.warning(self, "Error", "Please provide your trading reasoning")
            return
            
        try:
            # Add the decision to training data
            self.model_trainer.add_training_data(
                symbol=self.stock_combo.currentText(),
                decision=self.current_decision,
                reasoning=reasoning,
                data=self.current_data
            )
            
            QMessageBox.information(self, "Success", "Decision submitted successfully!")
            
            # Clear the form
            self.reasoning_input.clear()
            for btn in [self.buy_btn, self.sell_btn, self.hold_btn]:
                btn.setStyleSheet("")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to submit decision: {str(e)}")
            
    def start_training(self):
        if not self.model_trainer.has_training_data():
            QMessageBox.warning(self, "Error", "Please submit some trading decisions first")
            return
            
        self.train_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start training in a separate thread
        self.training_thread = TrainingThread(
            self.model_trainer,
            self.epochs_spin.value()
        )
        self.training_thread.progress.connect(self.update_progress)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.error.connect(self.training_error)
        self.training_thread.start()
        
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def training_finished(self):
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.information(self, "Success", "Model training completed!")
        
    def training_error(self, error_msg):
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", f"Training failed: {error_msg}")
        
    def get_prediction(self):
        try:
            symbol = self.pred_stock_combo.currentText()
            
            # Get recent data for prediction
            stock = yf.Ticker(symbol)
            df = stock.history(period="30d")
            
            if df.empty:
                QMessageBox.warning(self, "Error", "No data available for prediction")
                return
                
            # Get prediction from model
            decision = self.model_bridge.get_trading_decision(
                symbol=symbol,
                features=df,
                sentiment_data={},  # Add sentiment data if available
                vix=0.0,  # Add VIX data if available
                account_value=10000.0,  # Example account value
                current_position=0.0  # Example current position
            )
            
            # Display prediction
            prediction_text = f"""
            Symbol: {symbol}
            Decision: {decision['signal']}
            Confidence: {decision['confidence']:.2%}
            Position Size: {decision['position_size']:.2f}
            Stop Loss: {decision['stop_loss']:.2f}
            Take Profit: {decision['take_profit']:.2f}
            
            Reasoning:
            {decision['reasoning']}
            """
            
            self.prediction_display.setText(prediction_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to get prediction: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 