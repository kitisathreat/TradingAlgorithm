#!/usr/bin/env python3
"""
Rate Limiting Management Script
Command-line tool to monitor and control yfinance rate limiting on EC2
"""

import sys
import json
import requests
import time
from datetime import datetime
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from rate_limiter import get_rate_limiter_stats, clear_yfinance_cache, reset_yfinance_rate_limits, EnhancedStockDataFetcher
    RATE_LIMITER_AVAILABLE = True
except ImportError as e:
    print(f"Rate limiter not available: {e}")
    RATE_LIMITER_AVAILABLE = False

def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("🤖 YFinance Rate Limiting Management Tool")
    print("=" * 60)
    print("Manage rate limiting, caching, and fallback mechanisms")
    print("for your EC2 trading algorithm deployment")
    print("=" * 60)

def print_stats():
    """Print current rate limiter statistics"""
    if not RATE_LIMITER_AVAILABLE:
        print("❌ Rate limiter not available")
        return
    
    try:
        stats = get_rate_limiter_stats()
        
        print("\n📊 Rate Limiter Statistics:")
        print("-" * 40)
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Cached Requests: {stats['cached_requests']}")
        print(f"Rate Limited Requests: {stats['rate_limited_requests']}")
        print(f"Failed Requests: {stats['failed_requests']}")
        print(f"Backoff Events: {stats['backoff_events']}")
        print(f"Queue Size: {stats['queue_size']}")
        print(f"Consecutive Errors: {stats['consecutive_errors']}")
        print(f"Cache Size: {stats['cache_size']} files")
        print(f"Backoff Multiplier: {stats['backoff_multiplier']:.2f}")
        
        # Calculate success rate
        if stats['total_requests'] > 0:
            success_rate = ((stats['total_requests'] - stats['failed_requests']) / stats['total_requests']) * 100
            cache_hit_rate = (stats['cached_requests'] / stats['total_requests']) * 100 if stats['total_requests'] > 0 else 0
            print(f"Success Rate: {success_rate:.1f}%")
            print(f"Cache Hit Rate: {cache_hit_rate:.1f}%")
        
    except Exception as e:
        print(f"❌ Error getting stats: {e}")

def clear_cache():
    """Clear the yfinance cache"""
    if not RATE_LIMITER_AVAILABLE:
        print("❌ Rate limiter not available")
        return
    
    try:
        print("\n🗑️  Clearing yfinance cache...")
        clear_yfinance_cache()
        print("✅ Cache cleared successfully")
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")

def reset_limits():
    """Reset rate limits"""
    if not RATE_LIMITER_AVAILABLE:
        print("❌ Rate limiter not available")
        return
    
    try:
        print("\n🔄 Resetting rate limits...")
        reset_yfinance_rate_limits()
        print("✅ Rate limits reset successfully")
    except Exception as e:
        print(f"❌ Error resetting rate limits: {e}")

def test_data_fetching(symbols=None, days=7):
    """Test data fetching with multiple symbols"""
    if not RATE_LIMITER_AVAILABLE:
        print("❌ Rate limiter not available")
        return
    
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    print(f"\n🧪 Testing data fetching for {len(symbols)} symbols...")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Days: {days}")
    print("-" * 50)
    
    fetcher = EnhancedStockDataFetcher()
    results = {}
    
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Testing {symbol}...", end=" ")
        
        try:
            start_time = time.time()
            stock_info, data = fetcher.get_stock_data(symbol, days)
            end_time = time.time()
            
            if stock_info and data is not None and not data.empty:
                synthetic = stock_info.get('synthetic', False)
                data_points = len(data)
                duration = end_time - start_time
                
                status = "✅" if not synthetic else "🔄"
                print(f"{status} {data_points} data points in {duration:.2f}s")
                if synthetic:
                    print(f"    ⚠️  Using synthetic data (yfinance unavailable)")
                
                results[symbol] = {
                    'success': True,
                    'data_points': data_points,
                    'synthetic': synthetic,
                    'duration': duration
                }
            else:
                print("❌ No data returned")
                results[symbol] = {'success': False}
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results[symbol] = {'success': False, 'error': str(e)}
    
    # Summary
    print("\n📋 Test Summary:")
    print("-" * 30)
    successful = sum(1 for r in results.values() if r.get('success', False))
    synthetic_count = sum(1 for r in results.values() if r.get('synthetic', False))
    
    print(f"Successful: {successful}/{len(symbols)}")
    print(f"Synthetic data used: {synthetic_count}")
    print(f"Success rate: {(successful/len(symbols)*100):.1f}%")

def monitor_realtime():
    """Monitor rate limiting in real-time"""
    if not RATE_LIMITER_AVAILABLE:
        print("❌ Rate limiter not available")
        return
    
    print("\n📡 Real-time monitoring (Ctrl+C to stop)...")
    print("-" * 50)
    
    try:
        while True:
            stats = get_rate_limiter_stats()
            
            # Clear screen (works on most terminals)
            print("\033[2J\033[H", end="")
            
            print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 50)
            print(f"Total Requests: {stats['total_requests']}")
            print(f"Cached: {stats['cached_requests']} | Rate Limited: {stats['rate_limited_requests']}")
            print(f"Failed: {stats['failed_requests']} | Backoff Events: {stats['backoff_events']}")
            print(f"Queue Size: {stats['queue_size']} | Cache Files: {stats['cache_size']}")
            print(f"Consecutive Errors: {stats['consecutive_errors']}")
            print(f"Backoff Multiplier: {stats['backoff_multiplier']:.2f}")
            
            if stats['total_requests'] > 0:
                success_rate = ((stats['total_requests'] - stats['failed_requests']) / stats['total_requests']) * 100
                cache_hit_rate = (stats['cached_requests'] / stats['total_requests']) * 100
                print(f"Success Rate: {success_rate:.1f}% | Cache Hit Rate: {cache_hit_rate:.1f}%")
            
            print("-" * 50)
            print("Press Ctrl+C to stop monitoring")
            
            time.sleep(2)  # Update every 2 seconds
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped")

def show_help():
    """Show help information"""
    print("\n📖 Available Commands:")
    print("-" * 30)
    print("stats          - Show rate limiter statistics")
    print("clear-cache    - Clear yfinance cache")
    print("reset-limits   - Reset rate limits")
    print("test           - Test data fetching")
    print("monitor        - Real-time monitoring")
    print("help           - Show this help")
    print("exit           - Exit the program")
    print("\n💡 Tips:")
    print("- Use 'test' to verify rate limiting is working")
    print("- Use 'monitor' to watch real-time performance")
    print("- Use 'clear-cache' if you suspect stale data")
    print("- Use 'reset-limits' if you're getting blocked")

def main():
    """Main function"""
    print_banner()
    
    if not RATE_LIMITER_AVAILABLE:
        print("❌ Rate limiter module not available")
        print("Make sure rate_limiter.py is in the same directory")
        return
    
    print("✅ Rate limiter available")
    print_stats()
    
    while True:
        try:
            print("\n" + "=" * 40)
            command = input("Enter command (or 'help'): ").strip().lower()
            
            if command == 'exit' or command == 'quit':
                print("👋 Goodbye!")
                break
            elif command == 'help':
                show_help()
            elif command == 'stats':
                print_stats()
            elif command == 'clear-cache':
                clear_cache()
            elif command == 'reset-limits':
                reset_limits()
            elif command == 'test':
                symbols_input = input("Enter symbols (comma-separated, or press Enter for default): ").strip()
                if symbols_input:
                    symbols = [s.strip().upper() for s in symbols_input.split(',')]
                else:
                    symbols = None
                
                days_input = input("Enter number of days (or press Enter for 7): ").strip()
                days = int(days_input) if days_input.isdigit() else 7
                
                test_data_fetching(symbols, days)
            elif command == 'monitor':
                monitor_realtime()
            else:
                print("❌ Unknown command. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 