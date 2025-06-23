#!/usr/bin/env python3
"""
Rate Limiting and Caching Module for yfinance
Handles rate limiting, caching, and fallback mechanisms for EC2 deployment
"""

import time
import random
import logging
import threading
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from collections import deque, defaultdict
import pickle
import os
from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimiter:
    """Advanced rate limiter with exponential backoff and caching"""
    
    def __init__(self, 
                 max_requests_per_minute: int = 30,
                 max_requests_per_hour: int = 1000,
                 cache_duration_minutes: int = 15,
                 cache_dir: str = "cache",
                 enable_proxy_rotation: bool = False):
        
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.enable_proxy_rotation = enable_proxy_rotation
        
        # Request tracking
        self.request_times = deque()
        self.hourly_requests = deque()
        self.lock = threading.RLock()
        
        # Cache setup
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Rate limiting state
        self.backoff_multiplier = 1.0
        self.last_error_time = None
        self.consecutive_errors = 0
        self.max_backoff_seconds = 300  # 5 minutes max backoff
        
        # Request queue for throttling
        self.request_queue = deque()
        self.queue_processor = threading.Thread(target=self._process_queue, daemon=True)
        self.queue_processor.start()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'cached_requests': 0,
            'rate_limited_requests': 0,
            'failed_requests': 0,
            'backoff_events': 0
        }
        
        logger.info(f"RateLimiter initialized: {max_requests_per_minute} req/min, {max_requests_per_hour} req/hour")
    
    def _get_cache_key(self, symbol: str, start_date: str, end_date: str, period: str = None) -> str:
        """Generate cache key for data request"""
        key_data = f"{symbol}_{start_date}_{end_date}_{period}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _load_from_cache(self, cache_key: str) -> Optional[Tuple[Dict, pd.DataFrame]]:
        """Load data from cache if available and not expired"""
        try:
            cache_path = self._get_cache_path(cache_key)
            if not cache_path.exists():
                return None
            
            # Check if cache is expired
            if time.time() - cache_path.stat().st_mtime > self.cache_duration.total_seconds():
                cache_path.unlink()  # Remove expired cache
                return None
            
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            self.stats['cached_requests'] += 1
            logger.debug(f"Cache hit for key: {cache_key}")
            return cached_data
            
        except Exception as e:
            logger.warning(f"Cache load error: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, stock_info: Dict, data: pd.DataFrame):
        """Save data to cache"""
        try:
            cache_path = self._get_cache_path(cache_key)
            cached_data = (stock_info, data)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            
            logger.debug(f"Data cached for key: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Cache save error: {e}")
    
    def _wait_for_rate_limit(self):
        """Wait if rate limit is exceeded"""
        with self.lock:
            now = time.time()
            
            # Clean old requests
            while self.request_times and now - self.request_times[0] > 60:
                self.request_times.popleft()
            
            while self.hourly_requests and now - self.hourly_requests[0] > 3600:
                self.hourly_requests.popleft()
            
            # Check rate limits
            if len(self.request_times) >= self.max_requests_per_minute:
                wait_time = 60 - (now - self.request_times[0])
                if wait_time > 0:
                    logger.warning(f"Rate limit exceeded, waiting {wait_time:.2f} seconds")
                    self.stats['rate_limited_requests'] += 1
                    time.sleep(wait_time)
            
            if len(self.hourly_requests) >= self.max_requests_per_hour:
                wait_time = 3600 - (now - self.hourly_requests[0])
                if wait_time > 0:
                    logger.warning(f"Hourly rate limit exceeded, waiting {wait_time:.2f} seconds")
                    self.stats['rate_limited_requests'] += 1
                    time.sleep(wait_time)
            
            # Add current request
            self.request_times.append(now)
            self.hourly_requests.append(now)
    
    def _exponential_backoff(self):
        """Implement exponential backoff on errors"""
        if self.consecutive_errors > 0:
            backoff_time = min(
                self.backoff_multiplier * (2 ** (self.consecutive_errors - 1)),
                self.max_backoff_seconds
            )
            
            # Add jitter to prevent thundering herd
            jitter = random.uniform(0.1, 0.3) * backoff_time
            total_wait = backoff_time + jitter
            
            logger.warning(f"Exponential backoff: waiting {total_wait:.2f} seconds (error #{self.consecutive_errors})")
            self.stats['backoff_events'] += 1
            time.sleep(total_wait)
    
    def _process_queue(self):
        """Process queued requests with rate limiting"""
        while True:
            try:
                if self.request_queue:
                    request_func, args, kwargs, future = self.request_queue.popleft()
                    
                    # Wait for rate limit
                    self._wait_for_rate_limit()
                    
                    # Apply exponential backoff if needed
                    self._exponential_backoff()
                    
                    # Execute request
                    try:
                        result = request_func(*args, **kwargs)
                        future.set_result(result)
                        self.consecutive_errors = 0  # Reset on success
                        self.backoff_multiplier = 1.0
                    except Exception as e:
                        self.consecutive_errors += 1
                        self.last_error_time = time.time()
                        future.set_exception(e)
                        logger.error(f"Request failed: {e}")
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Queue processor error: {e}")
                time.sleep(1)
    
    def _make_yfinance_request(self, symbol: str, start_date: str, end_date: str, period: str = None) -> Tuple[Dict, pd.DataFrame]:
        """Make actual yfinance request with error handling"""
        try:
            ticker = yf.Ticker(symbol)
            
            if period:
                data = ticker.history(period=period)
            else:
                data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # Calculate basic metrics
            current_price = data['Close'].iloc[-1]
            volume = data['Volume'].iloc[-1]
            
            # Get market cap (approximate)
            try:
                info = ticker.info
                market_cap = info.get('marketCap', 0)
            except:
                market_cap = 0
            
            stock_info = {
                'current_price': float(current_price),
                'volume': int(volume),
                'market_cap': int(market_cap) if market_cap else 0,
                'symbol': symbol,
                'data_points': len(data),
                'date_range': f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
            }
            
            return stock_info, data
            
        except Exception as e:
            logger.error(f"yfinance request failed for {symbol}: {e}")
            raise
    
    def get_stock_data(self, symbol: str, days: int = 30, use_cache: bool = True) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
        """Get stock data with rate limiting and caching"""
        self.stats['total_requests'] += 1
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate cache key
        cache_key = self._get_cache_key(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # Try cache first
        if use_cache:
            cached_result = self._load_from_cache(cache_key)
            if cached_result:
                return cached_result
        
        # Queue the request for rate limiting
        from concurrent.futures import Future
        future = Future()
        
        self.request_queue.append((
            self._make_yfinance_request,
            (symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')),
            {},
            future
        ))
        
        try:
            stock_info, data = future.result(timeout=60)  # 60 second timeout
            
            # Cache the result
            if use_cache:
                self._save_to_cache(cache_key, stock_info, data)
            
            return stock_info, data
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"Failed to get stock data for {symbol}: {e}")
            return None, None
    
    def get_stock_data_immediate(self, symbol: str, days: int = 30, use_cache: bool = True) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
        """Get stock data immediately (bypassing queue) - use sparingly"""
        self.stats['total_requests'] += 1
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate cache key
        cache_key = self._get_cache_key(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # Try cache first
        if use_cache:
            cached_result = self._load_from_cache(cache_key)
            if cached_result:
                return cached_result
        
        # Wait for rate limit
        self._wait_for_rate_limit()
        
        # Apply exponential backoff if needed
        self._exponential_backoff()
        
        try:
            stock_info, data = self._make_yfinance_request(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            # Cache the result
            if use_cache:
                self._save_to_cache(cache_key, stock_info, data)
            
            return stock_info, data
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"Failed to get stock data for {symbol}: {e}")
            return None, None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        with self.lock:
            return {
                **self.stats,
                'queue_size': len(self.request_queue),
                'consecutive_errors': self.consecutive_errors,
                'backoff_multiplier': self.backoff_multiplier,
                'cache_size': len(list(self.cache_dir.glob("*.pkl")))
            }
    
    def clear_cache(self):
        """Clear all cached data"""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def reset_rate_limits(self):
        """Reset rate limiting state"""
        with self.lock:
            self.request_times.clear()
            self.hourly_requests.clear()
            self.consecutive_errors = 0
            self.backoff_multiplier = 1.0
            self.last_error_time = None
            logger.info("Rate limits reset")


# Global rate limiter instance
_rate_limiter = None

def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter

def rate_limited(func):
    """Decorator to apply rate limiting to functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        rate_limiter = get_rate_limiter()
        return rate_limiter.get_stock_data_immediate(*args, **kwargs)
    return wrapper


# Fallback data generator for when yfinance fails
class FallbackDataGenerator:
    """Generate synthetic stock data when yfinance is unavailable"""
    
    @staticmethod
    def generate_synthetic_data(symbol: str, days: int = 30) -> Tuple[Dict, pd.DataFrame]:
        """Generate realistic synthetic stock data"""
        try:
            # Generate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Filter to business days only
            business_days = date_range[date_range.weekday < 5]
            
            # Generate realistic price data
            base_price = random.uniform(50, 500)
            volatility = random.uniform(0.01, 0.05)  # 1-5% daily volatility
            
            prices = []
            volumes = []
            
            for i, date in enumerate(business_days):
                if i == 0:
                    price = base_price
                else:
                    # Random walk with mean reversion
                    change = random.gauss(0, volatility)
                    price = prices[-1] * (1 + change)
                    price = max(price, base_price * 0.5)  # Prevent going too low
                
                prices.append(price)
                
                # Generate realistic volume
                base_volume = random.randint(1000000, 10000000)
                volume = int(base_volume * random.uniform(0.5, 2.0))
                volumes.append(volume)
            
            # Create OHLC data
            data = pd.DataFrame({
                'Open': prices,
                'High': [p * random.uniform(1.0, 1.02) for p in prices],
                'Low': [p * random.uniform(0.98, 1.0) for p in prices],
                'Close': prices,
                'Volume': volumes
            }, index=business_days)
            
            # Ensure High >= Low and High >= Close >= Low
            data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
            data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
            
            stock_info = {
                'current_price': float(data['Close'].iloc[-1]),
                'volume': int(data['Volume'].iloc[-1]),
                'market_cap': random.randint(1000000000, 100000000000),  # 1B to 100B
                'symbol': symbol,
                'data_points': len(data),
                'date_range': f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}",
                'synthetic': True
            }
            
            logger.warning(f"Using synthetic data for {symbol} (yfinance unavailable)")
            return stock_info, data
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            return None, None


# Enhanced stock data fetcher with multiple fallbacks
class EnhancedStockDataFetcher:
    """Enhanced stock data fetcher with rate limiting and fallbacks"""
    
    def __init__(self):
        self.rate_limiter = get_rate_limiter()
        self.fallback_generator = FallbackDataGenerator()
    
    def get_stock_data(self, symbol: str, days: int = 30, use_fallback: bool = True) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
        """Get stock data with multiple fallback strategies"""
        
        # Try rate-limited yfinance first
        try:
            stock_info, data = self.rate_limiter.get_stock_data(symbol, days)
            if stock_info and data is not None and not data.empty:
                return stock_info, data
        except Exception as e:
            logger.warning(f"Rate-limited yfinance failed for {symbol}: {e}")
        
        # Try immediate yfinance request (bypassing queue)
        try:
            stock_info, data = self.rate_limiter.get_stock_data_immediate(symbol, days)
            if stock_info and data is not None and not data.empty:
                return stock_info, data
        except Exception as e:
            logger.warning(f"Immediate yfinance failed for {symbol}: {e}")
        
        # Use fallback synthetic data
        if use_fallback:
            logger.warning(f"All yfinance attempts failed for {symbol}, using synthetic data")
            return self.fallback_generator.generate_synthetic_data(symbol, days)
        
        return None, None
    
    def get_multiple_stocks(self, symbols: List[str], days: int = 30) -> Dict[str, Tuple[Optional[Dict], Optional[pd.DataFrame]]]:
        """Get data for multiple stocks with intelligent spacing"""
        results = {}
        
        for i, symbol in enumerate(symbols):
            try:
                # Add delay between requests to avoid overwhelming the API
                if i > 0:
                    time.sleep(random.uniform(1, 3))
                
                stock_info, data = self.get_stock_data(symbol, days)
                results[symbol] = (stock_info, data)
                
                logger.info(f"Processed {symbol} ({i+1}/{len(symbols)})")
                
            except Exception as e:
                logger.error(f"Failed to get data for {symbol}: {e}")
                results[symbol] = (None, None)
        
        return results


# Convenience functions for easy integration
def get_stock_data_with_rate_limiting(symbol: str, days: int = 30) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    """Convenience function to get stock data with rate limiting"""
    fetcher = EnhancedStockDataFetcher()
    return fetcher.get_stock_data(symbol, days)

def get_rate_limiter_stats() -> Dict[str, Any]:
    """Get current rate limiter statistics"""
    return get_rate_limiter().get_stats()

def clear_yfinance_cache():
    """Clear yfinance cache"""
    get_rate_limiter().clear_cache()

def reset_yfinance_rate_limits():
    """Reset yfinance rate limits"""
    get_rate_limiter().reset_rate_limits() 