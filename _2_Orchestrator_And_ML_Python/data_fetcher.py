# Standard library imports
import time
import threading
import glob
import os
from datetime import datetime, timedelta
from typing import List, Dict
import logging
import requests

# Third-party imports
import yfinance as yf
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# In-memory cache to limit API calls
cache = {}
CACHE_TTL_SECONDS = 3600  # Increased to 1 hour
cache_lock = threading.Lock()

# Rate limiting and parallel processing settings
RATE_LIMIT_DELAY = 0.5  # Increased from 0.1 to 0.5 seconds
MAX_WORKERS = 5  # Reduced from 10 to 5 to avoid rate limits
BATCH_SIZE = 3  # Reduced from 5 to 3 to avoid rate limits

class RateLimiter:
    def __init__(self, calls_per_second):
        self.calls_per_second = calls_per_second
        self.last_call = 0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call
            if time_since_last_call < (1.0 / self.calls_per_second):
                time.sleep((1.0 / self.calls_per_second) - time_since_last_call)
            self.last_call = time.time()

# Create rate limiters for different API endpoints
yf_rate_limiter = RateLimiter(10)  # 10 calls per second for yfinance
fmp_rate_limiter = RateLimiter(5)  # 5 calls per second for FMP

def get_cached_data(key: str) -> dict:
    with cache_lock:
        if key in cache and (time.time() - cache[key]['timestamp']) < CACHE_TTL_SECONDS:
            return cache[key]['data']
    return None

def set_cached_data(key: str, data: dict):
    with cache_lock:
        cache[key] = {'timestamp': time.time(), 'data': data}

def fetch_historical_data(symbol: str, years_back: int = 2) -> pd.DataFrame:
    """
    Fetch historical data for a given symbol with rate limiting
    """
    try:
        yf_rate_limiter.wait()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)
        
        # Check cache first
        cache_key = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        cached_data = get_cached_data(cache_key)
        if cached_data is not None:
            logging.info(f"Using cached data for {symbol}")
            return pd.DataFrame(cached_data)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')
        
        if not df.empty:
            # Add symbol column
            df['Symbol'] = symbol
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Rename columns for clarity
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open_price',
                'High': 'high_price',
                'Low': 'low_price',
                'Close': 'close_price',
                'Volume': 'volume',
                'Dividends': 'dividends',
                'Stock Splits': 'stock_splits'
            })
            
            # Cache the successful result
            with cache_lock:
                cache[cache_key] = {
                    'data': df.to_dict(),
                    'timestamp': time.time()
                }
            
            return df
    
    except Exception as e:
        logging.error(f"Error fetching historical data for {symbol}: {str(e)}")
    return pd.DataFrame()

def process_companies_parallel(companies: List[Dict], years_back: int) -> List[pd.DataFrame]:
    """
    Process multiple companies in parallel using ThreadPoolExecutor
    """
    all_data = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a future for each company
        future_to_company = {
            executor.submit(fetch_historical_data, company['symbol'], years_back): company 
            for company in companies
        }
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_company), 
                         total=len(future_to_company),
                         desc="Downloading historical data",
                         unit="company"):
            company = future_to_company[future]
            try:
                df = future.result()
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logging.error(f"Error processing {company['symbol']}: {str(e)}")
    
    return all_data

def get_sp500_companies() -> List[Dict]:
    """Fetch current S&P 500 companies from Wikipedia and return them sorted by market cap"""
    try:
        # Get S&P 500 table from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        
        # Get market cap for each company in parallel
        companies = []
        print("\nFetching market cap data for S&P 500 companies...")
        
        def fetch_market_cap(symbol):
            try:
                yf_rate_limiter.wait()
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if 'marketCap' in info:
                    return {
                        'symbol': symbol,
                        'name': df[df['Symbol'] == symbol]['Security'].iloc[0],
                        'market_cap': info['marketCap']
                    }
            except Exception as e:
                logging.warning(f"Could not fetch market cap for {symbol}: {str(e)}")
            return None

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_symbol = {
                executor.submit(fetch_market_cap, symbol): symbol 
                for symbol in df['Symbol']
            }
            
            for future in tqdm(as_completed(future_to_symbol), 
                             total=len(future_to_symbol),
                             desc="Processing companies",
                             unit="company"):
                result = future.result()
                if result:
                    companies.append(result)
        
        # Sort by market cap in descending order
        companies.sort(key=lambda x: x['market_cap'], reverse=True)
        return companies
    
    except Exception as e:
        logging.error(f"Error fetching S&P 500 companies: {str(e)}")
        raise

def manage_csv_files():
    """
    Manages CSV files by keeping only the three most recent ones.
    Deletes older files in descending order of age.
    """
    try:
        # Get all CSV files matching the pattern
        csv_files = glob.glob('sp500_historical_data_*.csv')
        
        if len(csv_files) > 3:
            # Sort files by modification time (newest first)
            csv_files.sort(key=os.path.getmtime, reverse=True)
            
            # Delete all but the three most recent files
            for old_file in csv_files[3:]:
                try:
                    os.remove(old_file)
                    logging.info(f"Deleted old data file: {old_file}")
                except Exception as e:
                    logging.error(f"Error deleting file {old_file}: {str(e)}")
    except Exception as e:
        logging.error(f"Error managing CSV files: {str(e)}")

class FMPFetcher:
    def __init__(self):
        self.api_key = os.getenv("FMP_API_KEY")
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def get_analyst_ratings(self, symbol):
        if not self.api_key:
            logging.warning("FMP API key not set.")
            return {"buy_ratio": 0.5}
        url = f"{self.base_url}/historical-rating/{symbol}?apikey={self.api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if not data:
                return {"buy_ratio": 0.5}
            latest = data[0]
            buy = latest.get("ratingBuy", 0) + latest.get("ratingOverweight", 0)
            total = buy + latest.get("ratingHold", 0) + latest.get("ratingUnderweight", 0) + latest.get("ratingSell", 0) + latest.get("ratingStrongSell", 0)
            buy_ratio = buy / total if total else 0.5
            return {"buy_ratio": buy_ratio}
        except Exception as e:
            logging.error(f"Error fetching analyst ratings: {e}")
            return {"buy_ratio": 0.5}

class NewsFetcher:
    def __init__(self, api=None):
        self.api = api  # Optionally pass an API client

    def get_latest_headline(self, symbol):
        # Placeholder: implement actual news fetching logic
        return f"No news available for {symbol}."

def main():
    try:
        # Get user input
        years_back = int(input("How many years of historical data do you want? "))
        num_companies = int(input("How many top S&P 500 companies do you want to analyze? "))
        
        if num_companies > 500:
            print("Maximum number of companies is 500. Setting to 500.")
            num_companies = 500
        
        # Get S&P 500 companies
        logging.info("Fetching S&P 500 companies...")
        companies = get_sp500_companies()
        
        # Take top N companies
        selected_companies = companies[:num_companies]
        
        # Fetch historical data for all companies in parallel
        print(f"\nFetching {years_back} years of historical data for {num_companies} companies...")
        all_data = process_companies_parallel(selected_companies, years_back)
        
        if not all_data:
            logging.error("No data was fetched for any companies")
            return
        
        # Combine all data
        print("\nProcessing and combining data...")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by date and symbol
        combined_df = combined_df.sort_values(['date', 'Symbol'])
        
        # Manage existing CSV files before saving new one
        manage_csv_files()
        
        # Save to CSV
        output_filename = f"sp500_historical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        print(f"\nSaving data to {output_filename}...")
        combined_df.to_csv(output_filename, index=False)
        logging.info(f"Data saved to {output_filename}")
        
        # Print summary
        print("\nSummary:")
        print(f"Total companies processed: {len(selected_companies)}")
        print(f"Date range: {combined_df['date'].min().date()} to {combined_df['date'].max().date()}")
        print(f"Total records: {len(combined_df)}")
        print(f"Output file: {output_filename}")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
