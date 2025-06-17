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
        
        # Draw candlesticks using line segments for wicks and rectangles for bodies
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        
        # Draw wicks (vertical lines)
        for i in range(len(df)):
            color = 'g' if closes[i] >= opens[i] else 'r'
            wick = pg.PlotDataItem([x_vals[i], x_vals[i]], [lows[i], highs[i]], pen=pg.mkPen(color=color, width=1))
            self.addItem(wick)
            self.plot_items.append(wick)
        
        # Draw bodies (rectangles)
        body_width = 1.2e4  # Increased width
        for i in range(len(df)):
            color = 'g' if closes[i] >= opens[i] else 'r'
            left = x_vals[i] - body_width
            right = x_vals[i] + body_width
            top = max(opens[i], closes[i])
            bottom = min(opens[i], closes[i])
            # Draw as a filled polygon (rectangle)
            body = pg.PlotDataItem(
                [left, right, right, left, left],
                [top, top, bottom, bottom, top],
                pen=pg.mkPen(color=color, width=1),
                brush=pg.mkBrush(color)
            )
            self.addItem(body)
            self.plot_items.append(body)
        
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
            self.model_trainer.train_model(epochs=self.epochs, 
                                         progress_callback=lambda x: self.progress.emit(x))
            self.finished.emit()
        except Exception as e:
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
        main_layout.setSpacing(10)
        
        # Controls (stock, date, get data)
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)
        
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
        
        # Loading indicator
        self.loading_label = QLabel("")
        self.loading_label.setVisible(False)
        controls_layout.addWidget(self.loading_label)
        
        main_layout.addWidget(controls_widget)
        controls_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Chart widget
        self.chart = StockChartWidget()
        main_layout.addWidget(self.chart, stretch=1)
        self.chart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
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
        main_layout.addWidget(auto_resize_widget)
        auto_resize_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Overlay metrics widget on top of chart
        self.metrics_widget = MetricsWidget(self.chart)
        self.metrics_widget.move(30, 30)
        self.metrics_widget.show()
        self.metrics_widget.raise_()
        # Optionally, set a higher z-order if needed
        # self.metrics_widget.setWindowFlags(self.metrics_widget.windowFlags() | Qt.WindowStaysOnTopHint)
        
        # Trading decision controls (below chart)
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
        
        main_layout.addWidget(decision_widget)
        decision_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Training controls (below decision controls)
        training_widget = QWidget()
        training_controls = QHBoxLayout(training_widget)
        training_controls.setContentsMargins(0, 0, 0, 0)
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
        
        main_layout.addWidget(training_widget)
        training_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
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
            
            # Show loading indicator
            self.get_data_btn.setEnabled(False)
            self.loading_label.setText("Loading data...")
            self.loading_label.setVisible(True)
            
            # Start data loading in background thread
            self.data_thread = DataLoadingThread(stock, days, start_date, end_date)
            self.data_thread.data_loaded.connect(self.on_data_loaded)
            self.data_thread.error.connect(self.on_data_error)
            self.data_thread.start()
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}\n\nTry:\n- Different stock symbol\n- Fewer days of history\n- Check internet connection\n- Check system clock")
    
    def on_data_loaded(self, df, stock, date_range_str):
        try:
            # Hide loading indicator
            self.get_data_btn.setEnabled(True)
            self.loading_label.setVisible(False)
            
            print(f"Successfully loaded {len(df)} days of data for {stock}")
            print(f"Data range: {date_range_str}")
            
            # Plot data
            self.chart.plot_stock_data(df)
            
            # Update metrics
            self.metrics_widget.update_metrics(df)
            
            # Store data for training
            self.current_data = df
            
            QMessageBox.information(self, "Success", f"Loaded {len(df)} days of data for {stock}\nDate range: {date_range_str}")
            
        except Exception as e:
            print(f"Error processing loaded data: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to process data: {str(e)}")
    
    def on_data_error(self, error_msg):
        # Hide loading indicator
        self.get_data_btn.setEnabled(True)
        self.loading_label.setVisible(False)
        
        QMessageBox.critical(self, "Error", error_msg)
    
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

    def auto_resize_x(self):
        vb = self.chart.getViewBox()
        vb.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
        vb.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)

    def auto_resize_y(self):
        vb = self.chart.getViewBox()
        vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)

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