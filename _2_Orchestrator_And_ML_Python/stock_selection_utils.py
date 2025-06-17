"""
Stock Selection Utilities for Trading Algorithm
Provides functions for enhanced stock selection including random picks, optimized picks, and custom ticker validation.
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yfinance as yf
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class StockSelectionManager:
    """Manages stock selection with enhanced features"""
    
    def __init__(self):
        self.sp100_data = self._load_sp100_data()
        self.sp100_symbols = [stock['symbol'] for stock in self.sp100_data]
        self._broader_stock_list = None  # Cache for broader stock list
        
    def _load_sp100_data(self) -> List[Dict]:
        """Load S&P100 data with market cap information"""
        try:
            sp100_file = Path(__file__).parent / "sp100_symbols_with_market_cap.json"
            with open(sp100_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading S&P100 data: {e}")
            # Fallback to basic symbols
            return [{"symbol": symbol, "name": symbol, "market_cap": 0} 
                   for symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX"]]
    
    def _get_broader_stock_list(self) -> List[str]:
        """Get a broader list of stocks from yfinance for random selection"""
        if self._broader_stock_list is not None:
            return self._broader_stock_list
            
        try:
            # Start with popular stock lists that yfinance can access
            broader_stocks = []
            
            # Add major indices and ETFs that contain many stocks
            index_symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO']
            
            for index_symbol in index_symbols:
                try:
                    ticker = yf.Ticker(index_symbol)
                    # Get holdings if available (this works for some ETFs)
                    if hasattr(ticker, 'holdings') and ticker.holdings is not None:
                        holdings = ticker.holdings
                        if isinstance(holdings, dict):
                            broader_stocks.extend(list(holdings.keys()))
                except Exception as e:
                    logger.debug(f"Could not get holdings for {index_symbol}: {e}")
                    continue
            
            # Add popular stocks from different sectors
            popular_stocks = [
                # Tech
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'NFLX', 'CRM', 'ADBE', 'PYPL', 'UBER', 'SHOP', 'SQ',
                # Finance
                'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA', 'BLK',
                # Healthcare
                'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
                # Consumer
                'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'DIS', 'CMCSA',
                # Energy
                'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'KMI', 'PSX', 'VLO',
                # Industrial
                'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'FDX', 'RTX', 'LMT',
                # Materials
                'LIN', 'APD', 'FCX', 'NEM', 'DOW', 'DD', 'NUE', 'BLL',
                # Utilities
                'NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'XEL', 'WEC',
                # Real Estate
                'AMT', 'PLD', 'CCI', 'EQIX', 'DLR', 'PSA', 'O', 'SPG',
                # Communication
                'T', 'VZ', 'TMUS', 'CHTR', 'CMCSA', 'FOX', 'NWSA',
                # International
                'ASML', 'TSM', 'NVO', 'SAP', 'NVS', 'RHHBY', 'TCEHY', 'BABA', 'JD', 'PDD'
            ]
            
            broader_stocks.extend(popular_stocks)
            
            # Remove duplicates and invalid symbols
            broader_stocks = list(set(broader_stocks))
            broader_stocks = [s for s in broader_stocks if s and len(s) <= 5 and s.isalpha()]
            
            # Add our S&P100 stocks to ensure we have them
            broader_stocks.extend(self.sp100_symbols)
            broader_stocks = list(set(broader_stocks))  # Remove duplicates again
            
            logger.info(f"Generated broader stock list with {len(broader_stocks)} symbols")
            self._broader_stock_list = broader_stocks
            return broader_stocks
            
        except Exception as e:
            logger.error(f"Error generating broader stock list: {e}")
            # Fallback to S&P100 + some popular stocks
            fallback_stocks = self.sp100_symbols + ['TSLA', 'NVDA', 'AMD', 'NFLX', 'CRM', 'ADBE', 'PYPL', 'UBER', 'SHOP', 'SQ']
            self._broader_stock_list = fallback_stocks
            return fallback_stocks
    
    def get_random_stock(self) -> Tuple[str, str]:
        """Get a random stock from a broader list including yfinance accessible stocks"""
        try:
            # Get broader stock list
            broader_stocks = self._get_broader_stock_list()
            
            # Try to get a random stock and validate it
            max_attempts = 10
            for attempt in range(max_attempts):
                symbol = random.choice(broader_stocks)
                
                # First check if it's in our S&P100 data
                for stock in self.sp100_data:
                    if stock['symbol'] == symbol:
                        return symbol, stock['name']
                
                # If not in S&P100, try to get info from yfinance
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    if info and 'longName' in info and info['longName']:
                        company_name = info['longName']
                        logger.info(f"Random pick from broader list: {symbol} ({company_name})")
                        return symbol, company_name
                        
                except Exception as e:
                    logger.debug(f"Could not validate {symbol}: {e}")
                    continue
            
            # If all attempts failed, fallback to S&P100
            logger.warning("Random pick from broader list failed, falling back to S&P100")
            stock = random.choice(self.sp100_data)
            return stock['symbol'], stock['name']
            
        except Exception as e:
            logger.error(f"Error in get_random_stock: {e}")
            # Final fallback
            stock = random.choice(self.sp100_data)
            return stock['symbol'], stock['name']
    
    def get_optimized_pick(self, trainer=None) -> Tuple[str, str]:
        """
        Get an optimized stock pick for neural network training.
        Prioritizes stocks with:
        1. High volatility (more training signal)
        2. Good liquidity (reliable data)
        3. Recent price movements (current market relevance)
        """
        try:
            # Define stocks that are typically good for training due to volatility and liquidity
            training_optimized_stocks = [
                "TSLA", "NVDA", "AMD", "NFLX", "META", "AAPL", "MSFT", "GOOGL", "AMZN",
                "CRM", "ADBE", "PYPL", "UBER", "SHOP", "SQ", "SPY", "QQQ", "PLTR", "COIN",
                "RIVN", "LCID", "NIO", "XPEV", "LI", "BYND", "ZM", "PTON", "HOOD", "SNAP"
            ]
            
            # Get broader stock list
            broader_stocks = self._get_broader_stock_list()
            
            # Filter to only include stocks that are in our broader list
            available_optimized = [s for s in training_optimized_stocks if s in broader_stocks]
            
            if not available_optimized:
                # Fallback to top market cap stocks from S&P100
                available_optimized = [stock['symbol'] for stock in self.sp100_data[:10]]
            
            # Select random from optimized list
            selected_symbol = random.choice(available_optimized)
            
            # Try to get company name
            selected_stock = next((stock for stock in self.sp100_data if stock['symbol'] == selected_symbol), None)
            
            if selected_stock:
                return selected_symbol, selected_stock['name']
            else:
                # Try to get name from yfinance
                try:
                    ticker = yf.Ticker(selected_symbol)
                    info = ticker.info
                    if info and 'longName' in info and info['longName']:
                        return selected_symbol, info['longName']
                except:
                    pass
                
                return selected_symbol, selected_symbol
                
        except Exception as e:
            logger.error(f"Error getting optimized pick: {e}")
            return self.get_random_stock()
    
    def validate_custom_ticker(self, ticker: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validate a custom ticker using yfinance
        
        Returns:
            Tuple of (is_valid, message, company_name)
        """
        try:
            # Clean the ticker symbol
            ticker = ticker.strip().upper()
            
            if not ticker:
                return False, "Please enter a ticker symbol", None
            
            # Check if it's already in our S&P100 list
            for stock in self.sp100_data:
                if stock['symbol'] == ticker:
                    return True, f"✅ {ticker} ({stock['name']}) is in S&P100", stock['name']
            
            # Try to get info from yfinance
            try:
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info
                
                if info and 'longName' in info and info['longName']:
                    company_name = info['longName']
                    return True, f"✅ {ticker} ({company_name}) found", company_name
                else:
                    return False, f"❌ {ticker} not found or invalid", None
                    
            except Exception as e:
                logger.warning(f"Error validating ticker {ticker}: {e}")
                return False, f"❌ Error validating {ticker}. Please check the symbol.", None
                
        except Exception as e:
            logger.error(f"Error in validate_custom_ticker: {e}")
            return False, "❌ Error validating ticker", None
    
    def get_stock_options(self) -> List[Dict]:
        """Get all stock options for dropdown including special options"""
        options = [
            {
                "symbol": "random",
                "name": "🎲 Random Pick",
                "description": "Select a random stock from S&P100"
            },
            {
                "symbol": "optimized",
                "name": "🚀 Optimized Pick",
                "description": "Select stock optimized for neural network training"
            },
            {
                "symbol": "custom",
                "name": "✏️ Custom Ticker",
                "description": "Enter your own stock ticker symbol"
            }
        ]
        
        # Add S&P100 stocks sorted by market cap
        for stock in self.sp100_data:
            market_cap_billions = stock['market_cap'] / 1_000_000_000
            options.append({
                "symbol": stock['symbol'],
                "name": f"{stock['symbol']} - {stock['name']}",
                "description": f"Market Cap: ${market_cap_billions:.1f}B"
            })
        
        return options
    
    def get_stock_display_name(self, symbol: str) -> str:
        """Get display name for a stock symbol"""
        if symbol == "random":
            return "🎲 Random Pick"
        elif symbol == "optimized":
            return "🚀 Optimized Pick"
        elif symbol == "custom":
            return "✏️ Custom Ticker"
        else:
            # Find in S&P100 data
            for stock in self.sp100_data:
                if stock['symbol'] == symbol:
                    return f"{symbol} - {stock['name']}"
            return symbol
    
    def get_stock_description(self, symbol: str) -> str:
        """Get description for a stock symbol"""
        if symbol == "random":
            return "Select a random stock from S&P100"
        elif symbol == "optimized":
            return "Select stock optimized for neural network training"
        elif symbol == "custom":
            return "Enter your own stock ticker symbol"
        else:
            # Find in S&P100 data
            for stock in self.sp100_data:
                if stock['symbol'] == symbol:
                    market_cap_billions = stock['market_cap'] / 1_000_000_000
                    return f"Market Cap: ${market_cap_billions:.1f}B"
            return "Custom ticker"

def format_market_cap(market_cap: int) -> str:
    """Format market cap in human readable format"""
    if market_cap >= 1_000_000_000_000:
        return f"${market_cap / 1_000_000_000_000:.1f}T"
    elif market_cap >= 1_000_000_000:
        return f"${market_cap / 1_000_000_000:.1f}B"
    elif market_cap >= 1_000_000:
        return f"${market_cap / 1_000_000:.1f}M"
    else:
        return f"${market_cap:,}" 