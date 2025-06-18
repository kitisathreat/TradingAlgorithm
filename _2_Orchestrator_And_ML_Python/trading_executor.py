"""
Trading Execution Module for Alpaca Paper Trading
Handles order execution, position management, and account monitoring
"""

import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Alpaca imports (will be available when alpaca-trade-api is installed)
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logging.warning("Alpaca trading API not available. Install with: pip install alpaca-trade-api==3.2.0 --no-deps")

logger = logging.getLogger(__name__)

class TradingExecutor:
    """
    Handles trading execution through Alpaca paper trading
    Manages orders, positions, and account monitoring
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True):
        """
        Initialize the trading executor
        
        Args:
            api_key: Alpaca API key (optional, can use environment variables)
            secret_key: Alpaca secret key (optional, can use environment variables)
            paper: Whether to use paper trading (default: True)
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.paper = paper
        self.api = None
        self.account = None
        self.positions = {}
        self.orders = []
        
        if not ALPACA_AVAILABLE:
            logger.error("Alpaca trading API not available. Cannot initialize trading executor.")
            return
            
        if not self.api_key or not self.secret_key:
            logger.error("Alpaca API credentials not provided. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
            return
            
        self._initialize_api()
    
    def _initialize_api(self):
        """Initialize the Alpaca API connection"""
        try:
            base_url = 'https://paper-api.alpaca.markets' if self.paper else 'https://api.alpaca.markets'
            self.api = tradeapi.REST(
                self.api_key,
                self.secret_key,
                base_url,
                api_version='v2'
            )
            
            # Test connection
            self.account = self.api.get_account()
            logger.info(f"Connected to Alpaca {'paper' if self.paper else 'live'} trading")
            logger.info(f"Account status: {self.account.status}")
            logger.info(f"Buying power: ${float(self.account.buying_power):,.2f}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca API: {e}")
            self.api = None
    
    def get_account_info(self) -> Dict:
        """Get current account information"""
        if not self.api:
            return {"error": "API not initialized"}
        
        try:
            account = self.api.get_account()
            return {
                "status": account.status,
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "daytrade_count": account.daytrade_count
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {"error": str(e)}
    
    def get_positions(self) -> Dict:
        """Get current positions"""
        if not self.api:
            return {"error": "API not initialized"}
        
        try:
            positions = self.api.list_positions()
            self.positions = {}
            
            for position in positions:
                self.positions[position.symbol] = {
                    "symbol": position.symbol,
                    "quantity": int(position.qty),
                    "side": position.side,
                    "market_value": float(position.market_value),
                    "unrealized_pl": float(position.unrealized_pl),
                    "current_price": float(position.current_price)
                }
            
            return self.positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {"error": str(e)}
    
    def place_order(self, symbol: str, qty: int, side: str, 
                   order_type: str = 'market', time_in_force: str = 'day',
                   limit_price: float = None, stop_price: float = None) -> Dict:
        """
        Place a trading order
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            qty: Number of shares
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            
        Returns:
            Dict with order information or error
        """
        if not self.api:
            return {"error": "API not initialized"}
        
        try:
            order_params = {
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'type': order_type,
                'time_in_force': time_in_force
            }
            
            if limit_price:
                order_params['limit_price'] = limit_price
            if stop_price:
                order_params['stop_price'] = stop_price
            
            order = self.api.submit_order(**order_params)
            
            order_info = {
                "id": order.id,
                "symbol": order.symbol,
                "qty": order.qty,
                "side": order.side,
                "type": order.type,
                "status": order.status,
                "submitted_at": order.submitted_at,
                "filled_at": order.filled_at,
                "filled_avg_price": order.filled_avg_price
            }
            
            self.orders.append(order_info)
            logger.info(f"Order placed: {order.symbol} {order.side} {order.qty} shares")
            
            return order_info
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {"error": str(e)}
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get status of a specific order"""
        if not self.api:
            return {"error": "API not initialized"}
        
        try:
            order = self.api.get_order(order_id)
            return {
                "id": order.id,
                "symbol": order.symbol,
                "qty": order.qty,
                "side": order.side,
                "type": order.type,
                "status": order.status,
                "submitted_at": order.submitted_at,
                "filled_at": order.filled_at,
                "filled_avg_price": order.filled_avg_price
            }
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return {"error": str(e)}
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel a pending order"""
        if not self.api:
            return {"error": "API not initialized"}
        
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order {order_id} cancelled")
            return {"success": True, "message": f"Order {order_id} cancelled"}
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return {"error": str(e)}
    
    def get_portfolio_history(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get portfolio performance history"""
        if not self.api:
            return pd.DataFrame()
        
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            portfolio_history = self.api.get_portfolio_history(
                start=start_date,
                end=end_date,
                timeframe='1D'
            )
            
            return portfolio_history.df
            
        except Exception as e:
            logger.error(f"Failed to get portfolio history: {e}")
            return pd.DataFrame()
    
    def execute_trading_signal(self, signal: Dict) -> Dict:
        """
        Execute a trading signal from the ML model
        
        Args:
            signal: Dict with 'symbol', 'action', 'confidence', 'quantity' keys
            
        Returns:
            Dict with execution results
        """
        if not self.api:
            return {"error": "API not initialized"}
        
        try:
            symbol = signal.get('symbol')
            action = signal.get('action')  # 'buy', 'sell', 'hold'
            confidence = signal.get('confidence', 0.0)
            quantity = signal.get('quantity', 1)
            
            if action == 'hold':
                return {"status": "hold", "message": "No action taken"}
            
            if confidence < 0.7:  # Only execute high-confidence signals
                return {"status": "low_confidence", "message": f"Confidence {confidence} below threshold"}
            
            # Check if we have enough buying power
            account_info = self.get_account_info()
            if 'error' in account_info:
                return account_info
            
            if action == 'buy':
                estimated_cost = quantity * 100  # Rough estimate, should get current price
                if account_info['buying_power'] < estimated_cost:
                    return {"error": "Insufficient buying power"}
            
            # Place the order
            order_result = self.place_order(
                symbol=symbol,
                qty=quantity,
                side=action,
                order_type='market'
            )
            
            return {
                "status": "executed",
                "signal": signal,
                "order": order_result
            }
            
        except Exception as e:
            logger.error(f"Failed to execute trading signal: {e}")
            return {"error": str(e)}
    
    def close_all_positions(self) -> Dict:
        """Close all open positions"""
        if not self.api:
            return {"error": "API not initialized"}
        
        try:
            positions = self.get_positions()
            if 'error' in positions:
                return positions
            
            results = []
            for symbol, position in positions.items():
                if position['quantity'] > 0:
                    side = 'sell' if position['side'] == 'long' else 'buy'
                    order_result = self.place_order(
                        symbol=symbol,
                        qty=abs(position['quantity']),
                        side=side,
                        order_type='market'
                    )
                    results.append(order_result)
            
            return {
                "status": "closing_positions",
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Failed to close positions: {e}")
            return {"error": str(e)}
    
    def is_market_open(self) -> bool:
        """Check if the market is currently open"""
        if not self.api:
            return False
        
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Failed to check market status: {e}")
            return False
    
    def get_market_hours(self) -> Dict:
        """Get current market hours information"""
        if not self.api:
            return {"error": "API not initialized"}
        
        try:
            clock = self.api.get_clock()
            return {
                "is_open": clock.is_open,
                "next_open": clock.next_open,
                "next_close": clock.next_close,
                "timestamp": clock.timestamp
            }
        except Exception as e:
            logger.error(f"Failed to get market hours: {e}")
            return {"error": str(e)}


# Example usage and testing
def test_trading_executor():
    """Test function for the trading executor"""
    executor = TradingExecutor()
    
    if not executor.api:
        print("Trading executor not available - check API credentials")
        return
    
    # Get account info
    account_info = executor.get_account_info()
    print(f"Account info: {account_info}")
    
    # Get positions
    positions = executor.get_positions()
    print(f"Positions: {positions}")
    
    # Check market status
    market_open = executor.is_market_open()
    print(f"Market open: {market_open}")
    
    # Get market hours
    market_hours = executor.get_market_hours()
    print(f"Market hours: {market_hours}")


if __name__ == "__main__":
    test_trading_executor() 