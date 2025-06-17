#!/usr/bin/env python3
"""
Test script for the enhanced GUI with hover tooltips and metrics widget.
This script tests the basic functionality without requiring the full application.
"""

import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtCore import Qt
import pyqtgraph as pg

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(__file__))

from main import StockChartWidget, MetricsWidget

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced GUI Test")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QHBoxLayout(central_widget)
        
        # Create chart widget
        self.chart = StockChartWidget()
        layout.addWidget(self.chart)
        
        # Create metrics widget
        self.metrics = MetricsWidget()
        layout.addWidget(self.metrics)
        
        # Generate sample data
        self.generate_sample_data()
        
    def generate_sample_data(self):
        """Generate sample stock data for testing"""
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        
        # Generate realistic price data
        np.random.seed(42)  # For reproducible results
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, 30)  # Daily returns
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLC data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC from close price
            volatility = 0.02
            high = price * (1 + abs(np.random.normal(0, volatility)))
            low = price * (1 - abs(np.random.normal(0, volatility)))
            open_price = price * (1 + np.random.normal(0, volatility * 0.5))
            
            # Ensure OHLC relationships are correct
            high = max(high, open_price, price)
            low = min(low, open_price, price)
            
            volume = int(np.random.uniform(1000000, 10000000))
            
            data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': price,
                'Volume': volume
            })
        
        # Create DataFrame
        df = pd.DataFrame(data, index=dates)
        
        # Plot data
        self.chart.plot_stock_data(df)
        
        # Update metrics
        self.metrics.update_metrics(df)
        
        print("Sample data generated and displayed!")
        print(f"Data range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"Current price: ${df['Close'].iloc[-1]:.2f}")

def main():
    app = QApplication(sys.argv)
    
    # Apply a simple style
    app.setStyle('Fusion')
    
    # Create and show test window
    window = TestWindow()
    window.show()
    
    print("Enhanced GUI Test Window opened!")
    print("Features to test:")
    print("1. Hover over candlesticks to see tooltips")
    print("2. Click the ▼ button in the metrics panel to collapse/expand")
    print("3. Scroll through the metrics to see all financial indicators")
    print("4. Zoom and pan the chart")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 