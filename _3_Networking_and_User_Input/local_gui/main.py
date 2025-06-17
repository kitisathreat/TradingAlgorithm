import sys
import os
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"
os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QComboBox, QSpinBox,
    QTabWidget, QProgressBar, QMessageBox, QSplitter,
    QSizePolicy, QFrame, QScrollArea, QGridLayout, QGroupBox, QLineEdit, QDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex, QPoint
from PyQt5.QtGui import QFont, QIcon, QPainter, QColor
import pyqtgraph as pg
import pandas as pd
import yfinance as yf
import numpy as np
from qt_material import apply_stylesheet
from pyqtgraph import DateAxisItem
import time
from functools import lru_cache
from custom_model_dialog import CustomModelDialog

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
        self.display_level = 'medium'  # 'min', 'medium', 'full'
        self.is_dragging = False
        self.drag_start_global = None
        self.widget_start_pos = None
        self.setWindowFlags(Qt.Widget | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(self._get_style())
        self.setMinimumWidth(250)
        self.setMaximumWidth(350)
        self.setMinimumHeight(40)
        self.setMaximumHeight(600)
        self.setup_ui()
        self.update_display_level('medium')

    def _get_style(self):
        return """
            QWidget {
                background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 8px;
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
        """

    def setup_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(5)

        # Header with drag and display level buttons
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(5)
        self.header = QWidget(self)
        self.header.setLayout(header_layout)
        self.header.setFixedHeight(32)
        self.header.setStyleSheet("background: #e9ecef; border-radius: 6px;")
        self.header.mousePressEvent = self.mousePressEvent
        self.header.mouseMoveEvent = self.mouseMoveEvent
        self.header.mouseReleaseEvent = self.mouseReleaseEvent

        self.title_label = QLabel("Financial Metrics", self)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        self.min_btn = QPushButton("_", self)
        self.min_btn.setFixedSize(24, 24)
        self.min_btn.clicked.connect(lambda: self.update_display_level('min'))
        header_layout.addWidget(self.min_btn)

        self.med_btn = QPushButton("□", self)
        self.med_btn.setFixedSize(24, 24)
        self.med_btn.clicked.connect(lambda: self.update_display_level('medium'))
        header_layout.addWidget(self.med_btn)

        self.full_btn = QPushButton("⬜", self)
        self.full_btn.setFixedSize(24, 24)
        self.full_btn.clicked.connect(lambda: self.update_display_level('full'))
        header_layout.addWidget(self.full_btn)

        self.layout.addWidget(self.header)

        # Scrollable content area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(5)
        self.create_price_metrics()
        self.create_volume_metrics()
        self.create_volatility_metrics()
        self.create_technical_metrics()
        self.scroll_area.setWidget(self.content_widget)
        self.layout.addWidget(self.scroll_area)

    def update_display_level(self, level):
        self.display_level = level
        if level == 'min':
            self.scroll_area.setVisible(False)
            self.setFixedHeight(self.header.height())
            self.setMaximumWidth(200)
        elif level == 'medium':
            self.scroll_area.setVisible(True)
            self.setMaximumWidth(350)
            self.setMinimumWidth(250)
            self.setMinimumHeight(200)
            self.setMaximumHeight(400)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        elif level == 'full':
            self.scroll_area.setVisible(True)
            parent = self.parentWidget()
            if parent:
                self.setMinimumHeight(parent.height() - 20)
                self.setMaximumHeight(parent.height() - 20)
            else:
                self.setMinimumHeight(400)
                self.setMaximumHeight(800)
            self.setMaximumWidth(400)
            self.setMinimumWidth(300)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    # --- DPI-safe Draggable overlay logic ---
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Only start drag if click is in header area
            if event.pos().y() <= self.header.height():
                self.is_dragging = True
                self.drag_start_global = event.globalPos()
                self.widget_start_pos = self.pos()
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            # Calculate delta in global coordinates
            delta = event.globalPos() - self.drag_start_global
            new_pos = self.widget_start_pos + delta
            parent = self.parentWidget()
            if parent:
                # Clamp to parent (chart) area
                x = max(0, min(new_pos.x(), parent.width() - self.width()))
                y = max(0, min(new_pos.y(), parent.height() - self.height()))
                self.move(x, y)
            else:
                self.move(new_pos)
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event):
        self.is_dragging = False
        event.accept()

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
            
            # Volume metrics
            current_volume = df['Volume'].iloc[-1]
            avg_volume = df['Volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            self.volume_label.setText(f"Volume: {current_volume:,}")
            self.avg_volume_label.setText(f"Avg Volume: {avg_volume:,.0f}")
            self.volume_ratio_label.setText(f"Volume Ratio: {volume_ratio:.2f}")
            
            # Volatility metrics
            daily_returns = df['Close'].pct_change().dropna()
            daily_vol = daily_returns.std() * 100
            weekly_vol = daily_returns.rolling(5).std().iloc[-1] * 100 if len(daily_returns) >= 5 else daily_vol
            
            # ATR calculation
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(14).mean().iloc[-1] if len(df) >= 14 else true_range.mean()
            
            # Bollinger Bands width
            sma_20 = df['Close'].rolling(20).mean()
            std_20 = df['Close'].rolling(20).std()
            bollinger_width = (std_20.iloc[-1] / sma_20.iloc[-1]) * 100 if not pd.isna(sma_20.iloc[-1]) else 0
            
            self.daily_vol_label.setText(f"Daily Volatility: {daily_vol:.2f}%")
            self.weekly_vol_label.setText(f"Weekly Volatility: {weekly_vol:.2f}%")
            self.atr_label.setText(f"ATR: ${atr:.2f}")
            self.bollinger_label.setText(f"Bollinger Width: {bollinger_width:.2f}%")
            
            # Technical indicators
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
            # MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            current_macd = macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0
            
            # Moving averages
            sma_20_current = sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else current_price
            ema_12_current = ema_12.iloc[-1] if not pd.isna(ema_12.iloc[-1]) else current_price
            
            self.rsi_label.setText(f"RSI: {current_rsi:.2f}")
            self.macd_label.setText(f"MACD: {current_macd:.2f}")
            self.sma_20_label.setText(f"SMA(20): ${sma_20_current:.2f}")
            self.ema_12_label.setText(f"EMA(12): ${ema_12_current:.2f}")
            
        except Exception as e:
            print(f"Error updating metrics: {str(e)}")

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
        self.plot_items = []
        
        # Create tooltip
        self.tooltip = pg.TextItem(text='', anchor=(0, 1))
        self.tooltip.setVisible(False)
        self.addItem(self.tooltip)
        
        # Throttle mouse events to improve performance
        self.last_mouse_update = 0
        self.mouse_update_threshold = 0.1  # 100ms between updates
        
        # Connect mouse events
        self.scene().sigMouseMoved.connect(self.mouse_moved)
        
        # Ensure the widget expands to fill available space
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
    def mouse_moved(self, pos):
        # Throttle mouse updates for better performance
        current_time = time.time()
        if current_time - self.last_mouse_update < self.mouse_update_threshold:
            return
        self.last_mouse_update = current_time
        
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
            
            # Style the tooltip (HTML only)
            self.tooltip.setHtml(f'<div style="background-color: white; border: 1px solid black; padding: 5px; border-radius: 3px; color: black;">{tooltip_text.replace(chr(10), "<br>")}</div>')
        else:
            self.tooltip.setVisible(False)
        
    def plot_stock_data(self, df):
        self.clear()
        if df.empty:
            print("No data to plot")
            return
            
        self.df = df  # Store for tooltips
        print(f"Plotting {len(df)} data points")
        
        # Convert dates to timestamps
        dates = pd.to_datetime(df.index)
        x_vals = dates.astype('int64') // 10**9
        
        # Clear previous plot items
        self.plot_items = []
        
        # Draw candlesticks using BarGraphItem for filled bodies and lines for wicks
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        
        body_width = 1.2e4  # Width for BarGraphItem
        bar_items = []
        # Draw wicks (vertical lines)
        for i in range(len(df)):
            color = 'g' if closes[i] >= opens[i] else 'r'
            wick = pg.PlotDataItem([x_vals[i], x_vals[i]], [lows[i], highs[i]], pen=pg.mkPen(color=color, width=1))
            self.addItem(wick)
            self.plot_items.append(wick)
        # Draw filled bodies
        for i in range(len(df)):
            color = 'g' if closes[i] >= opens[i] else 'r'
            x = x_vals[i]
            y = min(opens[i], closes[i])
            height = abs(opens[i] - closes[i])
            bar = pg.BarGraphItem(x=[x], height=[height], width=body_width*2, y=[y], brush=color, pen=pg.mkPen(color=color, width=1))
            self.addItem(bar)
            self.plot_items.append(bar)
        
        min_price = df['Low'].min()
        max_price = df['High'].max()
        current_price = df['Close'].iloc[-1]
        price_padding = (max_price - min_price) * 0.05  # 5% padding
        
        # Auto-range ONCE, then lock
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
        # Lock auto-range so it doesn't rescale on hover
        self.getViewBox().enableAutoRange(axis=pg.ViewBox.XYAxes, enable=False)
        
        # Add current price line
        self.addLine(y=current_price, pen=pg.mkPen('b', width=2, style=pg.QtCore.Qt.DashLine))
        
        # Re-add tooltip
        self.tooltip = pg.TextItem(text='', anchor=(0, 1))
        self.tooltip.setVisible(False)
        self.addItem(self.tooltip)
        
        print(f"Chart updated - Price range: ${min_price:.2f} to ${max_price:.2f}, Current: ${current_price:.2f}")

class DataLoadingThread(QThread):
    data_loaded = pyqtSignal(object, str, str)  # df, stock, date_range
    error = pyqtSignal(str)
    
    def __init__(self, stock, days, start_date, end_date):
        super().__init__()
        self.stock = stock
        self.days = days
        self.start_date = start_date
        self.end_date = end_date
        
    def run(self):
        try:
            print(f"Fetching {self.days} days of data for {self.stock} from {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
            
            ticker = yf.Ticker(self.stock)
            df = ticker.history(start=self.start_date, end=self.end_date)
            
            if df.empty:
                self.error.emit(f"No data found for {self.stock}")
                return
            
            # Ensure df.index is timezone-aware (UTC)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
            
            # Validate that data is not newer than our end_date
            if df.index.max() > self.end_date:
                df = df[df.index <= self.end_date]
                if df.empty:
                    self.error.emit(f"No valid data found for {self.stock} after filtering dates")
                    return
            
            date_range_str = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
            self.data_loaded.emit(df, self.stock, date_range_str)
            
        except Exception as e:
            self.error.emit(f"Failed to load data: {str(e)}")

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
            print(f"Starting training with {self.epochs} epochs...")
            print(f"Training data: {len(self.model_trainer.training_examples)} examples")
            print(f"Model type: {self.model_trainer.model_type}")
            
            # Train the neural network
            success = self.model_trainer.train_neural_network(epochs=self.epochs)
            
            if success:
                print("Training completed successfully!")
                self.finished.emit()
            else:
                self.error.emit("Training failed - check console for details")
                
        except Exception as e:
            print(f"Error in training thread: {e}")
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network Trading System")
        self.setMinimumSize(800, 600)
        
        # Initialize model components
        self.model_trainer = ModelTrainer()
        self.model_bridge = ModelBridge()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create scrollable area for entire content
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create content widget that will be inside the scroll area
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.content_layout.setSpacing(10)
        
        # Controls (stock, date, get data)
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)
        
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
            self.stock_combo.addItems([
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 
                'AMD', 'INTC', 'CRM', 'ADBE', 'PYPL', 'UBER', 'SHOP', 'SQ',
                'SPY', 'QQQ', 'IWM', 'VTI'  # ETFs for more reliable data
            ])
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
        
        # Loading indicator
        self.loading_label = QLabel("")
        self.loading_label.setVisible(False)
        controls_layout.addWidget(self.loading_label)
        
        self.content_layout.addWidget(controls_widget)
        controls_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Create a splitter for the chart and controls area
        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(8)
        self.main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #cccccc;
                border: 1px solid #999999;
            }
            QSplitter::handle:hover {
                background-color: #aaaaaa;
            }
        """)
        
        # Chart widget (top part of splitter)
        self.chart = StockChartWidget()
        self.chart.setMinimumHeight(300)  # Minimum height for chart
        self.main_splitter.addWidget(self.chart)
        
        # Auto Resize buttons below the chart
        auto_resize_widget = QWidget()
        auto_resize_layout = QHBoxLayout(auto_resize_widget)
        auto_resize_layout.setContentsMargins(0, 0, 0, 0)
        auto_resize_layout.setSpacing(10)
        self.auto_x_btn = QPushButton("Auto Resize X Axis")
        self.auto_x_btn.clicked.connect(self.auto_resize_x)
        auto_resize_layout.addWidget(self.auto_x_btn)
        self.auto_y_btn = QPushButton("Auto Resize Y Axis")
        self.auto_y_btn.clicked.connect(self.auto_resize_y)
        auto_resize_layout.addWidget(self.auto_y_btn)
        auto_resize_layout.addStretch()
        
        # Add splitter control buttons
        self.maximize_chart_btn = QPushButton("📈 Max Chart")
        self.maximize_chart_btn.setToolTip("Maximize chart area (80% chart, 20% controls)")
        self.maximize_chart_btn.clicked.connect(self.maximize_chart_area)
        auto_resize_layout.addWidget(self.maximize_chart_btn)
        
        self.minimize_chart_btn = QPushButton("📊 Min Chart")
        self.minimize_chart_btn.setToolTip("Minimize chart area (40% chart, 60% controls)")
        self.minimize_chart_btn.clicked.connect(self.minimize_chart_area)
        auto_resize_layout.addWidget(self.minimize_chart_btn)
        
        self.reset_splitter_btn = QPushButton("🔄 Reset")
        self.reset_splitter_btn.setToolTip("Reset to default proportions (70% chart, 30% controls)")
        self.reset_splitter_btn.clicked.connect(self.reset_splitter_proportions)
        auto_resize_layout.addWidget(self.reset_splitter_btn)
        
        self.main_splitter.addWidget(auto_resize_widget)
        auto_resize_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Overlay metrics widget on top of chart
        self.metrics_widget = MetricsWidget(self.chart)
        self.metrics_widget.move(30, 30)
        self.metrics_widget.show()
        self.metrics_widget.raise_()
        
        # Add splitter to content layout
        self.content_layout.addWidget(self.main_splitter, stretch=1)
        
        # Create a container for all the controls below the chart
        controls_container = QWidget()
        controls_container_layout = QVBoxLayout(controls_container)
        controls_container_layout.setContentsMargins(0, 0, 0, 0)
        controls_container_layout.setSpacing(10)
        
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
        self.reasoning_input.setMaximumHeight(100)  # Limit height for better layout
        decision_layout.addWidget(self.reasoning_input)
        
        # Submit button
        self.submit_btn = QPushButton("Submit Decision")
        self.submit_btn.clicked.connect(self.submit_decision)
        decision_layout.addWidget(self.submit_btn)
        
        controls_container_layout.addWidget(decision_widget)
        decision_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Training controls
        training_widget = QWidget()
        training_layout = QVBoxLayout(training_widget)
        training_layout.setContentsMargins(0, 0, 0, 0)
        training_layout.setSpacing(10)
        
        # Additive training info
        additive_info = QLabel(f"📚 Additive Training Enabled: {len(self.model_trainer.training_examples)} examples from previous sessions")
        additive_info.setStyleSheet("color: #2196F3; font-weight: bold; padding: 5px;")
        training_layout.addWidget(additive_info)
        
        # Model selection section
        model_selection_widget = QWidget()
        model_layout = QHBoxLayout(model_selection_widget)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(10)
        
        model_layout.addWidget(QLabel("Neural Network Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Simple Model (32-16-3 layers)",
            "Standard Model (128-64-32-3 layers)", 
            "Deep Model (256-128-64-32-16-3 layers)",
            "LSTM Model (Sequential patterns)",
            "Ensemble Model (Multiple architectures)",
            "➕ Create Your Own Model..."
        ])
        self.model_combo.setCurrentIndex(1)  # Default to standard
        model_layout.addWidget(self.model_combo)
        self.model_combo.setContextMenuPolicy(Qt.CustomContextMenu)
        self.model_combo.customContextMenuRequested.connect(self.show_model_context_menu)
        
        self.change_model_btn = QPushButton("Change Model")
        self.change_model_btn.clicked.connect(self.change_model)
        model_layout.addWidget(self.change_model_btn)
        
        training_layout.addWidget(model_selection_widget)
        
        # Training parameters section
        training_controls = QHBoxLayout()
        training_controls.setContentsMargins(0, 0, 0, 0)
        training_controls.setSpacing(10)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(50)
        training_controls.addWidget(QLabel("Training Epochs:"))
        training_controls.addWidget(self.epochs_spin)
        
        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self.start_training)
        training_controls.addWidget(self.train_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        training_controls.addWidget(self.progress_bar)
        
        training_layout.addLayout(training_controls)
        
        controls_container_layout.addWidget(training_widget)
        training_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Add controls container to main splitter
        self.main_splitter.addWidget(controls_container)
        
        # Set initial splitter sizes (70% chart, 30% controls)
        self.main_splitter.setSizes([700, 300])
        
        # Set the content widget as the scroll area's widget
        self.scroll_area.setWidget(self.content_widget)
        main_layout.addWidget(self.scroll_area)
        
        # Add a status bar to show scroll information
        self.statusBar().showMessage("Use the splitter handle to resize the chart area. Scroll if content exceeds window bounds. F11: Fullscreen, F5: Refresh data.")
        
    def on_date_range_changed(self, value):
        if value == "Custom":
            self.date_range_spin.setVisible(True)
        else:
            self.date_range_spin.setVisible(False)
    
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
            else:
                stock = stock_data
            
            # Additional validation for stock symbol
            if not stock or not isinstance(stock, str) or not stock.strip():
                QMessageBox.warning(self, "Warning", "Invalid stock symbol")
                return
            
            # Get days from combo or spinbox
            if hasattr(self, 'date_range_combo') and self.date_range_combo.currentText() == "Custom":
                days = self.date_range_spin.value()
            elif hasattr(self, 'date_range_combo'):
                days = int(self.date_range_combo.currentText())
            else:
                days = 30  # Default fallback
            
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
        """Start neural network training with selected model"""
        try:
            if not hasattr(self, 'model_trainer') or not self.model_trainer.training_examples:
                QMessageBox.warning(self, "Warning", "No training examples available. Please add some training examples first.")
                return
            
            epochs = self.epochs_spin.value()
            
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, epochs)
            self.progress_bar.setValue(0)
            self.train_btn.setEnabled(False)
            
            # Start training in background thread
            self.training_thread = TrainingThread(self.model_trainer, epochs)
            self.training_thread.progress.connect(self.update_progress)
            self.training_thread.finished.connect(self.training_finished)
            self.training_thread.error.connect(self.training_error)
            self.training_thread.start()
            
        except Exception as e:
            print(f"Error starting training: {e}")
            QMessageBox.critical(self, "Error", f"Error starting training: {e}")
            self.progress_bar.setVisible(False)
            self.train_btn.setEnabled(True)
    
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

    def auto_resize_x(self):
        vb = self.chart.getViewBox()
        vb.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
        vb.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)

    def auto_resize_y(self):
        vb = self.chart.getViewBox()
        vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)

    def change_model(self):
        """Change the neural network model type or open custom model dialog"""
        try:
            selected_index = self.model_combo.currentIndex()
            if selected_index == self.model_combo.count() - 1:  # 'Create Your Own Model...'
                dialog = CustomModelDialog(self)
                if dialog.exec_() == dialog.Accepted:
                    config = dialog.get_model_config()
                    custom_label = f"{config['model_name']} (Custom)"
                    # Insert before the last item (so 'Create Your Own Model...' stays last)
                    self.model_combo.insertItem(self.model_combo.count() - 1, custom_label)
                    self.model_combo.setCurrentIndex(self.model_combo.count() - 2)
                    # Store config for later use (could be a dict or list)
                    if not hasattr(self, 'custom_models'):
                        self.custom_models = {}
                    self.custom_models[custom_label] = config
                    QMessageBox.information(self, "Success", f"Custom model '{config['model_name']}' added and selected.")
                return
            model_mapping = {
                0: "simple",
                1: "standard", 
                2: "deep",
                3: "lstm",
                4: "ensemble"
            }
            new_model_type = model_mapping.get(selected_index, "standard")
            if self.model_trainer.change_model_type(new_model_type):
                QMessageBox.information(self, "Success", f"Model changed to {self.model_combo.currentText()}")
            else:
                QMessageBox.warning(self, "Error", "Failed to change model")
        except Exception as e:
            print(f"Error changing model: {e}")
            QMessageBox.critical(self, "Error", f"Error changing model: {e}")

    def resizeEvent(self, event):
        """Handle window resize events to maintain proper splitter proportions"""
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
            f"Chart area: {self.chart.width()}x{self.chart.height()} | "
            f"Use splitter handle to resize chart area{scroll_info}"
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

    def maximize_chart_area(self):
        """Maximize the chart area by minimizing controls"""
        if hasattr(self, 'main_splitter'):
            total_height = self.main_splitter.height()
            # Give 80% to chart, 20% to controls
            chart_height = int(total_height * 0.8)
            controls_height = total_height - chart_height
            self.main_splitter.setSizes([chart_height, controls_height])

    def minimize_chart_area(self):
        """Minimize the chart area to show more controls"""
        if hasattr(self, 'main_splitter'):
            total_height = self.main_splitter.height()
            # Give 40% to chart, 60% to controls
            chart_height = int(total_height * 0.4)
            controls_height = total_height - chart_height
            self.main_splitter.setSizes([chart_height, controls_height])

    def reset_splitter_proportions(self):
        """Reset splitter to default proportions (70% chart, 30% controls)"""
        if hasattr(self, 'main_splitter'):
            total_height = self.main_splitter.height()
            chart_height = int(total_height * 0.7)
            controls_height = total_height - chart_height
            self.main_splitter.setSizes([chart_height, controls_height])

    def show_model_context_menu(self, pos):
        index = self.model_combo.view().indexAt(pos)
        if not index.isValid():
            return
        item_text = self.model_combo.itemText(index.row())
        if not item_text.endswith("(Custom)"):
            return
        menu = QMenu(self)
        edit_action = menu.addAction("Edit Custom Model")
        delete_action = menu.addAction("Delete Custom Model")
        action = menu.exec_(self.model_combo.mapToGlobal(pos))
        if action == edit_action:
            self.edit_custom_model(index.row())
        elif action == delete_action:
            self.delete_custom_model(index.row())

    def edit_custom_model(self, row):
        label = self.model_combo.itemText(row)
        config = self.custom_models.get(label)
        if not config:
            QMessageBox.warning(self, "Error", "Custom model config not found.")
            return
        dialog = CustomModelDialog(self, config)
        if dialog.exec_() == dialog.Accepted:
            new_config = dialog.get_model_config()
            new_label = f"{new_config['model_name']} (Custom)"
            self.model_combo.setItemText(row, new_label)
            self.custom_models.pop(label)
            self.custom_models[new_label] = new_config
            self.model_combo.setCurrentIndex(row)
            QMessageBox.information(self, "Success", f"Custom model updated.")

    def delete_custom_model(self, row):
        label = self.model_combo.itemText(row)
        if label in self.custom_models:
            self.custom_models.pop(label)
        self.model_combo.removeItem(row)
        # Select previous or first model
        if self.model_combo.count() > 1:
            self.model_combo.setCurrentIndex(max(0, row - 1))

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