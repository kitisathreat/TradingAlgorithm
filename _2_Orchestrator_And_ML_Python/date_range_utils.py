"""
Date Range Utilities for Trading Algorithm
Provides functions to randomly select date ranges with intelligent fallback to find the closest available data.
No longer limited to 25 years - can find data from anywhere in history.
"""

import random
from datetime import datetime, timedelta, timezone
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def get_random_date_range(days: int, max_years_back: int = None) -> Tuple[datetime, datetime]:
    """
    Generate a random date range of the specified number of days.
    If max_years_back is None, will look as far back as needed to find data.
    
    Args:
        days: Number of days of data requested
        max_years_back: Maximum years to look back (None = no limit)
    
    Returns:
        Tuple of (start_date, end_date) in UTC timezone
    """
    # Get current date in UTC
    current_date = datetime.now(timezone.utc)
    
    # Calculate the earliest possible start date
    if max_years_back is not None:
        earliest_start = current_date - timedelta(days=max_years_back * 365)
    else:
        # No limit - go back to 1900 for stocks (reasonable historical limit)
        earliest_start = datetime(1900, 1, 1, tzinfo=timezone.utc)
    
    # Calculate the latest possible end date (current date minus buffer for data availability)
    # Use a 5-day buffer to ensure we get real data
    latest_end = current_date - timedelta(days=5)
    
    # Calculate the latest possible start date (latest_end - days)
    latest_start = latest_end - timedelta(days=days)
    
    # Ensure latest_start is not before earliest_start
    if latest_start < earliest_start:
        latest_start = earliest_start
        # Adjust end date if necessary
        latest_end = latest_start + timedelta(days=days)
        if latest_end > current_date - timedelta(days=5):
            latest_end = current_date - timedelta(days=5)
    
    # Generate random start date within the valid range
    days_range = (latest_start - earliest_start).days
    if days_range <= 0:
        # If range is too small, use the earliest possible range
        start_date = earliest_start
        end_date = start_date + timedelta(days=days)
        if end_date > latest_end:
            end_date = latest_end
            start_date = end_date - timedelta(days=days)
    else:
        random_days = random.randint(0, days_range)
        start_date = earliest_start + timedelta(days=random_days)
        end_date = start_date + timedelta(days=days)
    
    # Ensure dates are in UTC
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)
    
    logger.info(f"Generated random date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({days} days)")
    
    return start_date, end_date

def find_available_data_range(symbol: str, requested_days: int, max_years_back: int = None) -> Tuple[datetime, datetime]:
    """
    Find an available data range for a given symbol, with fallback logic.
    No longer limited to 25 years - can find data from anywhere in history.
    
    Args:
        symbol: Stock symbol
        requested_days: Number of days of data requested
        max_years_back: Maximum years to look back (None = no limit)
    
    Returns:
        Tuple of (start_date, end_date) that should have available data
    """
    # Validate symbol parameter
    if symbol is None or not isinstance(symbol, str) or not symbol.strip():
        logger.error(f"Invalid symbol provided: {symbol}")
        return get_random_date_range(requested_days, max_years_back)
    
    symbol = symbol.strip().upper()
    
    try:
        import yfinance as yf
        
        # First, try to get the actual data range available for this symbol
        ticker = yf.Ticker(symbol)
        
        # Try different periods to find available data, starting with max to get full history
        for period in ["max", "10y", "5y", "2y", "1y"]:
            try:
                data = ticker.history(period=period)
                if not data.empty and len(data) >= requested_days:
                    # Get the actual date range
                    actual_start = data.index.min()
                    actual_end = data.index.max()
                    
                    # Ensure timezone awareness
                    if actual_start.tzinfo is None:
                        actual_start = actual_start.tz_localize('UTC')
                    if actual_end.tzinfo is None:
                        actual_end = actual_end.tz_localize('UTC')
                    
                    # Calculate a random start date within the available range
                    # We want to find a window of requested_days within the available data
                    available_days = (actual_end - actual_start).days
                    if available_days >= requested_days:
                        # Find a random start point that allows for requested_days of data
                        max_start_offset = available_days - requested_days
                        random_offset = random.randint(0, max_start_offset)
                        start_date = actual_start + timedelta(days=random_offset)
                        end_date = start_date + timedelta(days=requested_days)
                        
                        logger.info(f"Found available data for {symbol}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({requested_days} days)")
                        return start_date, end_date
                    
            except Exception as e:
                logger.warning(f"Failed to get data for {symbol} with period {period}: {e}")
                continue
        
        # If we can't determine the actual range, fall back to random selection
        logger.warning(f"Could not determine actual data range for {symbol}, using random selection")
        return get_random_date_range(requested_days, max_years_back)
        
    except ImportError:
        logger.warning("yfinance not available, using random date range")
        return get_random_date_range(requested_days, max_years_back)
    except Exception as e:
        logger.error(f"Error finding available data range for {symbol}: {e}")
        return get_random_date_range(requested_days, max_years_back)

def validate_date_range(start_date: datetime, end_date: datetime, symbol: str = None) -> bool:
    """
    Validate that a date range is reasonable and should have data.
    Updated to allow much older data (no longer limited to 25 years).
    
    Args:
        start_date: Start date
        end_date: End date
        symbol: Optional symbol for logging
    
    Returns:
        True if the date range is valid
    """
    # Handle None symbol gracefully
    symbol_display = symbol if symbol is not None else 'unknown'
    
    current_date = datetime.now(timezone.utc)
    
    # Check if dates are in the future
    if start_date > current_date or end_date > current_date:
        logger.warning(f"Date range contains future dates for {symbol_display}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        return False
    
    # Check if start date is too far in the past (more than 150 years - reasonable for stock data)
    if start_date < current_date - timedelta(days=150 * 365):
        logger.warning(f"Start date too far in the past for {symbol_display}: {start_date.strftime('%Y-%m-%d')}")
        return False
    
    # Check if date range is reasonable (not negative or too long)
    if start_date >= end_date:
        logger.warning(f"Invalid date range for {symbol_display}: start >= end")
        return False
    
    # Allow longer date ranges since we're looking for specific lengths from anywhere in history
    if (end_date - start_date).days > 365 * 50:  # More than 50 years
        logger.warning(f"Date range too long for {symbol_display}: {(end_date - start_date).days} days")
        return False
    
    return True 