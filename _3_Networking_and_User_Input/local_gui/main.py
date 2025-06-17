import sys
import os
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"
os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QComboBox, QSpinBox,
    QTabWidget, QProgressBar, QMessageBox, QSplitter,
    QSizePolicy, QFrame, QScrollArea, QGridLayout, QGroupBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QPainter, QColor
import pyqtgraph as pg
import pandas as pd
import yfinance as yf
import numpy as np
from qt_material import apply_stylesheet
from pyqtgraph import DateAxisItem

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

class CustomDateAxisItem(DateAxisItem):
    def tickStrings(self, values, scale, spacing):
        # Format as DD/MM/YY
        return [pd.to_datetime(val, unit='s').strftime('%d/%m/%y') for val in values]

class MetricsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumWidth(300)
        self.setMinimumWidth(250)
        self.is_expanded = True
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Header with toggle button
        header_layout = QHBoxLayout()
        self.toggle_btn = QPushButton("▼")
        self.toggle_btn.setMaximumWidth(20)
        self.toggle_btn.clicked.connect(self.toggle_expansion)
        header_layout.addWidget(self.toggle_btn)
        
        title_label = QLabel("Financial Metrics")
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Scrollable content area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(5)
        
        # Create metric groups
        self.create_price_metrics()
        self.create_volume_metrics()
        self.create_volatility_metrics()
        self.create_technical_metrics()
        
        self.scroll_area.setWidget(self.content_widget)
        layout.addWidget(self.scroll_area)
        
        # Style the widget
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ced4da;
                border-radius: 3px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                font-size: 10px;
            }
        """)
        
    def create_price_metrics(self):
        group = QGroupBox("Price Metrics")
        layout = QGridLayout(group)
        
        self.current_price_label = QLabel("Current Price: $0.00")
        self.price_change_label = QLabel("Price Change: $0.00 (0.00%)")
        self.high_low_label = QLabel("High/Low: $0.00 / $0.00")
        self.avg_price_label = QLabel("Avg Price: $0.00")
        
        layout.addWidget(self.current_price_label, 0, 0)
        layout.addWidget(self.price_change_label, 1, 0)
        layout.addWidget(self.high_low_label, 2, 0)
        layout.addWidget(self.avg_price_label, 3, 0)
        
        self.content_layout.addWidget(group)
        
    def create_volume_metrics(self):
        group = QGroupBox("Volume Metrics")
        layout = QGridLayout(group)
        
        self.volume_label = QLabel("Volume: 0")
        self.avg_volume_label = QLabel("Avg Volume: 0")
        self.volume_ratio_label = QLabel("Volume Ratio: 0.00")
        
        layout.addWidget(self.volume_label, 0, 0)
        layout.addWidget(self.avg_volume_label, 1, 0)
        layout.addWidget(self.volume_ratio_label, 2, 0)
        
        self.content_layout.addWidget(group)
        
    def create_volatility_metrics(self):
        group = QGroupBox("Volatility Metrics")
        layout = QGridLayout(group)
        
        self.daily_vol_label = QLabel("Daily Volatility: 0.00%")
        self.weekly_vol_label = QLabel("Weekly Volatility: 0.00%")
        self.atr_label = QLabel("ATR: $0.00")
        self.bollinger_label = QLabel("Bollinger Width: 0.00%")
        
        layout.addWidget(self.daily_vol_label, 0, 0)
        layout.addWidget(self.weekly_vol_label, 1, 0)
        layout.addWidget(self.atr_label, 2, 0)
        layout.addWidget(self.bollinger_label, 3, 0)
        
        self.content_layout.addWidget(group)
        
    def create_technical_metrics(self):
        group = QGroupBox("Technical Indicators")
        layout = QGridLayout(group)
        
        self.rsi_label = QLabel("RSI: 0.00")
        self.macd_label = QLabel("MACD: 0.00")
        self.sma_20_label = QLabel("SMA(20): $0.00")
        self.ema_12_label = QLabel("EMA(12): $0.00")
        
        layout.addWidget(self.rsi_label, 0, 0)
        layout.addWidget(self.macd_label, 1, 0)
        layout.addWidget(self.sma_20_label, 2, 0)
        layout.addWidget(self.ema_12_label, 3, 0)
        
        self.content_layout.addWidget(group)
        
    def toggle_expansion(self):
        self.is_expanded = not self.is_expanded
        if self.is_expanded:
            self.toggle_btn.setText("▼")
            self.scroll_area.show()
        else:
            self.toggle_btn.setText("▶")
            self.scroll_area.hide()
            
    def update_metrics(self, df):
        if df is None or df.empty:
            return
            
        try:
            # Price metrics
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
            
            high_price = df['High'].max()
            low_price = df['Low'].min()
            avg_price = df['Close'].mean()
            
            self.current_price_label.setText(f"Current Price: ${current_price:.2f}")
            self.price_change_label.setText(f"Price Change: ${price_change:.2f} ({price_change_pct:+.2f}%)")
            self.high_low_label.setText(f"High/Low: ${high_price:.2f} / ${low_price:.2f}")
            self.avg_price_label.setText(f"Avg Price: ${avg_price:.2f}")
            
            # Color coding for price change
            if price_change > 0:
                self.price_change_label.setStyleSheet("color: green; font-weight: bold;")
            elif price_change < 0:
                self.price_change_label.setStyleSheet("color: red; font-weight: bold;")
            else:
                self.price_change_label.setStyleSheet("color: black;")
            
            # Volume metrics
            current_volume = df['Volume'].iloc[-1]
            avg_volume = df['Volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            self.volume_label.setText(f"Volume: {current_volume:,}")
            self.avg_volume_label.setText(f"Avg Volume: {avg_volume:,.0f}")
            self.volume_ratio_label.setText(f"Volume Ratio: {volume_ratio:.2f}")
            
            # Volatility metrics
            daily_returns = df['Close'].pct_change().dropna()
            daily_vol = daily_returns.std() * 100 if len(daily_returns) > 0 else 0
            
            # Weekly volatility (5-day rolling)
            weekly_returns = df['Close'].pct_change(5).dropna()
            weekly_vol = weekly_returns.std() * 100 if len(weekly_returns) > 0 else 0
            
            # Average True Range (ATR)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1] if len(true_range) >= 14 else true_range.mean()
            
            # Bollinger Bands width
            sma_20 = df['Close'].rolling(20).mean()
            std_20 = df['Close'].rolling(20).std()
            bollinger_width = (2 * std_20 / sma_20 * 100).iloc[-1] if len(sma_20) >= 20 else 0
            
            self.daily_vol_label.setText(f"Daily Volatility: {daily_vol:.2f}%")
            self.weekly_vol_label.setText(f"Weekly Volatility: {weekly_vol:.2f}%")
            self.atr_label.setText(f"ATR: ${atr:.2f}")
            self.bollinger_label.setText(f"Bollinger Width: {bollinger_width:.2f}%")
            
            # Technical indicators
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_value = rsi.iloc[-1] if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else 50
            
            # MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_value = macd.iloc[-1] if len(macd) > 0 else 0
            
            # Moving averages
            sma_20_current = df['Close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['Close'].mean()
            ema_12_current = df['Close'].ewm(span=12).mean().iloc[-1] if len(df) >= 12 else df['Close'].mean()
            
            self.rsi_label.setText(f"RSI: {rsi_value:.2f}")
            self.macd_label.setText(f"MACD: {macd_value:.2f}")
            self.sma_20_label.setText(f"SMA(20): ${sma_20_current:.2f}")
            self.ema_12_label.setText(f"EMA(12): ${ema_12_current:.2f}")
            
        except Exception as e:
            print(f"Error updating metrics: {str(e)}")
            # Set default values on error
            self.current_price_label.setText("Current Price: Error")
            self.price_change_label.setText("Price Change: Error")
            self.high_low_label.setText("High/Low: Error")
            self.avg_price_label.setText("Avg Price: Error")

class StockChartWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        # Use CustomDateAxisItem for the x-axis
        date_axis = CustomDateAxisItem(orientation='bottom')
        super().__init__(parent, axisItems={'bottom': date_axis})
        self.setBackground('w')  # White background
        self.showGrid(x=True, y=True)
        self.setLabel('left', 'Price (USD$)')
        self.setLabel('bottom', 'Date')
        
        # Enable mouse interaction
        self.setMouseEnabled(x=True, y=True)
        self.showButtons()
        
        # Enable panning and zooming
        self.setInteractive(True)
        self.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        
        # Store data for tooltips
        self.df = None
        self.candlestick_items = []
        
        # Create tooltip
        self.tooltip = pg.TextItem(text='', anchor=(0, 1))
        self.tooltip.setVisible(False)
        self.addItem(self.tooltip)
        
        # Connect mouse events
        self.scene().sigMouseMoved.connect(self.mouse_moved)
        
        # Ensure the widget expands to fill available space
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
    def mouse_moved(self, pos):
        if self.df is None or self.df.empty:
            return
            
        # Convert mouse position to data coordinates
        view = self.getViewBox()
        if view is None:
            return
            
        mouse_point = view.mapSceneToView(pos)
        x_pos = mouse_point.x()
        
        # Find the closest candlestick
        dates = pd.to_datetime(self.df.index)
        x_vals = dates.astype('int64') // 10**9
        
        # Find the closest data point
        closest_idx = np.argmin(np.abs(x_vals - x_pos))
        
        if 0 <= closest_idx < len(self.df):
            row = self.df.iloc[closest_idx]
            date_str = dates[closest_idx].strftime('%Y-%m-%d')
            
            tooltip_text = f"""
Date: {date_str}
Open: ${row['Open']:.2f}
High: ${row['High']:.2f}
Low: ${row['Low']:.2f}
Close: ${row['Close']:.2f}
Volume: {row['Volume']:,}
            """.strip()
            
            self.tooltip.setText(tooltip_text)
            self.tooltip.setPos(x_vals[closest_idx], row['High'])
            self.tooltip.setVisible(True)
            
            # Style the tooltip
            self.tooltip.setDefaultTextColor(pg.mkColor('black'))
            self.tooltip.setHtml(f'<div style="background-color: white; border: 1px solid black; padding: 5px; border-radius: 3px;">{tooltip_text.replace(chr(10), "<br>")}</div>')
        else:
            self.tooltip.setVisible(False)
        
    def plot_stock_data(self, df):
        self.clear()
        if df.empty:
            print("No data to plot")
            return
            
        self.df = df  # Store for tooltips
        print(f"Plotting {len(df)} data points")
        dates = pd.to_datetime(df.index)
        x_vals = dates.astype('int64') // 10**9  # Convert to seconds since epoch (fix for pandas error)
        
        self.candlestick_items = []  # Clear previous items
        
        for i in range(len(df)):
            x = x_vals[i]
            open_price = df['Open'].iloc[i]
            close_price = df['Close'].iloc[i]
            high_price = df['High'].iloc[i]
            low_price = df['Low'].iloc[i]
            color = 'g' if close_price >= open_price else 'r'
            
            # Create wick (high-low line)
            wick = self.plot([x, x], [low_price, high_price], 
                           pen=pg.mkPen(color=color, width=1))
            
            # Create body (open-close rectangle)
            body_width = 0.5e4  # Adjust for better visibility
            body = self.plot([x-body_width, x+body_width, x+body_width, x-body_width, x-body_width],
                           [open_price, open_price, close_price, close_price, open_price],
                           pen=pg.mkPen(color=color, width=1),
                           fillLevel=0,
                           brush=pg.mkBrush(color))
            
            self.candlestick_items.extend([wick, body])
            
        min_price = df['Low'].min()
        max_price = df['High'].max()
        current_price = df['Close'].iloc[-1]
        price_padding = (max_price - min_price) * 0.05  # 5% padding
        
        # Always auto-range after plotting to match axes to data
        self.getViewBox().enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        
        # Then set default zoom to last 30 days (or all data if less)
        num_points = len(x_vals)
        if num_points > 30:
            x_min = x_vals[-30]
        else:
            x_min = x_vals[0]
        x_max = x_vals[-1]
        self.getViewBox().setXRange(x_min, x_max, padding=0)
        self.getViewBox().setYRange(max(0, min_price - price_padding), max_price + price_padding, padding=0)
        self.setLabel('left', f'Price (USD$)')
        self.setLabel('bottom', 'Date')
        
        # Add current price line
        self.addLine(y=current_price, pen=pg.mkPen('b', width=2, style=pg.QtCore.Qt.DashLine))
        
        # Re-add tooltip
        self.tooltip = pg.TextItem(text='', anchor=(0, 1))
        self.tooltip.setVisible(False)
        self.addItem(self.tooltip)
        
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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(0)
        
        # Stock selection
        self.stock_combo = QComboBox()
        self.stock_combo.addItems([
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 
            'AMD', 'INTC', 'CRM', 'ADBE', 'PYPL', 'UBER', 'SHOP', 'SQ',
            'SPY', 'QQQ', 'IWM', 'VTI'  # ETFs for more reliable data
        ])
        controls_layout.addWidget(QLabel("Stock:"))
        controls_layout.addWidget(self.stock_combo)
        
        # Date range selection
        controls_layout.addWidget(QLabel("Days of History:"))
        self.date_range_combo = QComboBox()
        self.date_range_combo.addItems(["30", "60", "90", "180", "365", "Custom"])
        self.date_range_combo.setCurrentIndex(0)
        controls_layout.addWidget(self.date_range_combo)
        self.date_range_spin = QSpinBox()
        self.date_range_spin.setRange(1, 365)
        self.date_range_spin.setValue(30)
        self.date_range_spin.setVisible(False)
        controls_layout.addWidget(self.date_range_spin)
        self.date_range_combo.currentTextChanged.connect(self.on_date_range_changed)
        
        # Get data button
        self.get_data_btn = QPushButton("Get Stock Data")
        self.get_data_btn.clicked.connect(self.load_stock_data)
        controls_layout.addWidget(self.get_data_btn)
        
        layout.addLayout(controls_layout)
        
        # Main horizontal splitter for chart and metrics
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        layout.addWidget(main_splitter)
        
        # Chart widget
        self.chart = StockChartWidget()
        main_splitter.addWidget(self.chart)
        
        # Metrics widget
        self.metrics_widget = MetricsWidget()
        main_splitter.addWidget(self.metrics_widget)
        
        # Set splitter proportions (chart gets more space)
        main_splitter.setSizes([800, 300])
        
        # Vertical splitter for chart and controls
        chart_splitter = QSplitter(Qt.Vertical)
        chart_splitter.setChildrenCollapsible(False)
        
        # Trading decision controls
        decision_widget = QWidget()
        decision_layout = QVBoxLayout(decision_widget)
        decision_layout.setContentsMargins(0, 0, 0, 0)
        decision_layout.setSpacing(0)
        
        # Decision buttons
        decision_buttons = QHBoxLayout()
        decision_buttons.setContentsMargins(0, 0, 0, 0)
        decision_buttons.setSpacing(0)
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
        
        chart_splitter.addWidget(decision_widget)
        
        # Training controls
        training_controls = QHBoxLayout()
        training_controls.setContentsMargins(0, 0, 0, 0)
        training_controls.setSpacing(0)
        
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
        
        # Create a widget to hold training controls
        training_widget = QWidget()
        training_widget.setLayout(training_controls)
        chart_splitter.addWidget(training_widget)
        
        # Add the chart splitter to the main splitter
        main_splitter.insertWidget(0, chart_splitter)
        
        # Set the chart splitter to take up the remaining space
        main_splitter.setSizes([1100, 300])
        
    def setup_prediction_tab(self, tab):
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(0)
        
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
        
        # Main horizontal splitter for chart and metrics
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        layout.addWidget(main_splitter)
        
        # Chart widget for prediction
        self.pred_chart = StockChartWidget()
        main_splitter.addWidget(self.pred_chart)
        
        # Metrics widget for prediction
        self.pred_metrics_widget = MetricsWidget()
        main_splitter.addWidget(self.pred_metrics_widget)
        
        # Set splitter proportions (chart gets more space)
        main_splitter.setSizes([800, 300])
        
        # Prediction result
        self.prediction_label = QLabel("Prediction will appear here...")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet("font-size: 18px; padding: 20px;")
        layout.addWidget(self.prediction_label)
        
    def on_date_range_changed(self, value):
        if value == "Custom":
            self.date_range_spin.setVisible(True)
        else:
            self.date_range_spin.setVisible(False)
    
    def load_stock_data(self):
        try:
            stock = self.stock_combo.currentText()
            # Get days from combo or spinbox
            if hasattr(self, 'date_range_combo') and self.date_range_combo.currentText() == "Custom":
                days = self.date_range_spin.value()
            elif hasattr(self, 'date_range_combo'):
                days = int(self.date_range_combo.currentText())
            else:
                days = 30
            
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
            
            # Update metrics
            self.metrics_widget.update_metrics(df)
            
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
            # Import the date range utilities
            import sys
            from pathlib import Path
            REPO_ROOT = Path(__file__).parent.parent.parent
            ORCHESTRATOR_PATH = REPO_ROOT / "_2_Orchestrator_And_ML_Python"
            sys.path.append(str(ORCHESTRATOR_PATH))
            
            from date_range_utils import find_available_data_range, validate_date_range
            
            # Get random date range within the last 25 years (30 days for prediction)
            start_date, end_date = find_available_data_range(stock, 30, max_years_back=25)
            
            # Validate the date range
            if not validate_date_range(start_date, end_date, stock):
                self.prediction_label.setText(f"Error: Invalid date range generated for {stock}")
                return
            
            print(f"Fetching prediction data for {stock} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (random range within last 25 years)")
            
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
            
            # Plot data
            self.pred_chart.plot_stock_data(df)
            
            # Update metrics
            self.pred_metrics_widget.update_metrics(df)
                
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
                confidence = 0.5
                
            # Format prediction result
            prediction_text = f"""
Prediction for {stock}:
Signal: {signal}
Confidence: {confidence:.1%}
Current Price: ${current_price:.2f}
Price Change: ${price_change:.2f} ({price_change_pct:+.2f}%)
            """.strip()
            
            self.prediction_label.setText(prediction_text)
            
        except Exception as e:
            print(f"Error getting prediction: {str(e)}")
            self.prediction_label.setText(f"Error: {str(e)}")

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
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