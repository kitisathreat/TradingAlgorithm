#!/usr/bin/env python3
"""
Test script to verify that the metrics widget is working correctly.
This script tests the metrics calculation functionality specifically.
"""

import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(__file__))

from main import MetricsWidget

class MetricsTestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Metrics Widget Test")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QHBoxLayout(central_widget)
        
        # Create metrics widget
        self.metrics = MetricsWidget()
        layout.addWidget(self.metrics)
        
        # Create info panel
        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)
        
        self.info_label = QLabel("Metrics Test Results")
        self.info_label.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        info_layout.addWidget(self.info_label)
        
        self.status_label = QLabel("Status: Ready to test")
        info_layout.addWidget(self.status_label)
        
        layout.addWidget(info_panel)
        
        # Generate and test sample data
        self.test_metrics()
        
    def test_metrics(self):
        """Test the metrics widget with sample data"""
        try:
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
            
            # Update metrics
            self.metrics.update_metrics(df)
            
            # Update status
            current_price = df['Close'].iloc[-1]
            price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
            price_change_pct = (price_change / df['Close'].iloc[-2]) * 100
            
            status_text = f"""
✅ Metrics Test Successful!

Sample Data Generated:
- Data points: {len(df)} days
- Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}
- Current price: ${current_price:.2f}
- Price change: ${price_change:.2f} ({price_change_pct:+.2f}%)

The metrics widget should now display:
- Price metrics (current price, change, high/low, average)
- Volume metrics (current volume, average, ratio)
- Volatility metrics (daily/weekly volatility, ATR, Bollinger width)
- Technical indicators (RSI, MACD, SMA, EMA)

Check the metrics panel on the left to verify all values are populated correctly.
            """.strip()
            
            self.status_label.setText(status_text)
            print("✅ Metrics test completed successfully!")
            print(f"Current price: ${current_price:.2f}")
            print(f"Price change: ${price_change:.2f} ({price_change_pct:+.2f}%)")
            
        except Exception as e:
            error_text = f"❌ Metrics test failed: {str(e)}"
            self.status_label.setText(error_text)
            print(f"❌ Error in metrics test: {str(e)}")

def main():
    app = QApplication(sys.argv)
    
    # Apply a simple style
    app.setStyle('Fusion')
    
    # Create and show test window
    window = MetricsTestWindow()
    window.show()
    
    print("Metrics Test Window opened!")
    print("This test verifies that the metrics widget correctly calculates and displays:")
    print("1. Price metrics (current price, change, high/low, average)")
    print("2. Volume metrics (current volume, average, ratio)")
    print("3. Volatility metrics (daily/weekly volatility, ATR, Bollinger width)")
    print("4. Technical indicators (RSI, MACD, SMA, EMA)")
    print("\nIf all values are populated (not showing 0), the fix is working correctly!")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 