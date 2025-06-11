# Standard library imports
import glob
import os
import logging
from datetime import datetime, timedelta

# Third-party imports
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarketAnalyzer:
    def __init__(self):
        # Find all data files and get the newest one
        data_files = glob.glob('sp500_historical_data_*.csv')
        if not data_files:
            raise FileNotFoundError("No historical data files found. Please run data_fetcher.py first.")
        
        # Sort files by modification time (newest first) and get the most recent
        data_files.sort(key=os.path.getmtime, reverse=True)
        latest_file = data_files[0]
        
        # Log the file being used and its modification time
        file_time = datetime.fromtimestamp(os.path.getmtime(latest_file))
        logging.info(f"Using data file: {latest_file}")
        logging.info(f"File last modified: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load the data
        self.df = pd.read_csv(latest_file)
        # Fix the datetime parsing warning by using utc=True
        self.df['date'] = pd.to_datetime(self.df['date'], utc=True)
        
        # Calculate daily returns for each stock
        self.df['daily_return'] = self.df.groupby('Symbol')['close_price'].pct_change()
        
        # Get S&P 500 data for market returns
        self.sp500 = self._get_sp500_data()
        
        # Set the style for all plots
        plt.style.use('seaborn-v0_8')  # Using a specific seaborn style version
        
        # Log data summary
        logging.info(f"Data range: {self.df['date'].min().date()} to {self.df['date'].max().date()}")
        logging.info(f"Number of companies: {len(self.df['Symbol'].unique())}")
        
    def _get_sp500_data(self):
        """Get S&P 500 data for market returns calculation"""
        try:
            sp500 = yf.download('^GSPC', 
                              start=self.df['date'].min(),
                              end=self.df['date'].max(),
                              auto_adjust=True)  # Explicitly set auto_adjust
            sp500['market_return'] = sp500['Close'].pct_change()
            return sp500
        except Exception as e:
            print(f"Warning: Could not fetch S&P 500 data: {e}")
            return None

    def calculate_beta(self, symbol, window_years=1):
        """Calculate beta for a given symbol over a specified window"""
        stock_data = self.df[self.df['Symbol'] == symbol].copy()
        if stock_data.empty:
            return None
            
        # Merge with market data
        stock_data = stock_data.merge(
            self.sp500['market_return'].reset_index(),
            left_on='date',
            right_on='Date',
            how='left'
        )
        
        # Calculate rolling beta
        window_days = int(window_years * 252)  # Approximate trading days in a year
        if len(stock_data) < window_days:
            return None
            
        # Calculate covariance and variance
        covariance = stock_data['daily_return'].rolling(window=window_days).cov(stock_data['market_return'])
        market_variance = stock_data['market_return'].rolling(window=window_days).var()
        
        # Calculate beta
        beta = covariance / market_variance
        return beta.iloc[-1]  # Return the most recent beta

    def analyze_market_trends(self):
        """Create visualizations for market trends"""
        # Set the style for this specific plot
        plt.style.use('seaborn-v0_8')
        
        # 1. Overall Market Trend
        plt.figure(figsize=(15, 10))
        
        # Plot average price trend
        avg_prices = self.df.groupby('date')['close_price'].mean()
        plt.plot(avg_prices.index, avg_prices.values, label='Average Stock Price', linewidth=2)
        
        # Plot S&P 500 if available
        if self.sp500 is not None:
            plt.plot(self.sp500.index, self.sp500['Close'], 
                    label='S&P 500', linewidth=2, alpha=0.7)
        
        plt.title('Market Trend Analysis', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.savefig('market_trend.png')
        plt.close()

    def analyze_beta_distribution(self):
        """Analyze and visualize beta distribution across different time horizons"""
        # Set the style for this specific plot
        plt.style.use('seaborn-v0_8')
        
        time_horizons = [1, 3, 5, 10]  # years
        symbols = self.df['Symbol'].unique()
        
        beta_data = []
        for symbol in symbols:
            for years in time_horizons:
                beta = self.calculate_beta(symbol, years)
                if beta is not None:
                    beta_data.append({
                        'Symbol': symbol,
                        'Years': years,
                        'Beta': beta
                    })
        
        beta_df = pd.DataFrame(beta_data)
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        sns.boxplot(data=beta_df, x='Years', y='Beta')
        plt.title('Beta Distribution Across Time Horizons', fontsize=14)
        plt.xlabel('Time Horizon (Years)', fontsize=12)
        plt.ylabel('Beta', fontsize=12)
        plt.savefig('beta_distribution.png')
        plt.close()
        
        # Find companies with highest beta for each horizon
        high_beta_companies = {}
        for years in time_horizons:
            horizon_data = beta_df[beta_df['Years'] == years]
            top_5 = horizon_data.nlargest(5, 'Beta')
            high_beta_companies[years] = top_5[['Symbol', 'Beta']].values.tolist()
        
        return high_beta_companies

    def analyze_volatility(self):
        """Analyze and visualize price volatility"""
        # Set the style for this specific plot
        plt.style.use('seaborn-v0_8')
        
        # Calculate rolling standard deviation for each stock
        volatility = self.df.groupby('Symbol')['close_price'].agg([
            ('volatility_1y', lambda x: x.pct_change().rolling(252).std().iloc[-1]),
            ('volatility_3y', lambda x: x.pct_change().rolling(756).std().iloc[-1]),
            ('volatility_5y', lambda x: x.pct_change().rolling(1260).std().iloc[-1])
        ]).reset_index()
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        volatility_melted = pd.melt(volatility, 
                                  id_vars=['Symbol'],
                                  value_vars=['volatility_1y', 'volatility_3y', 'volatility_5y'],
                                  var_name='Horizon',
                                  value_name='Volatility')
        
        sns.boxplot(data=volatility_melted, x='Horizon', y='Volatility')
        plt.title('Price Volatility Distribution Across Time Horizons', fontsize=14)
        plt.xlabel('Time Horizon', fontsize=12)
        plt.ylabel('Volatility (Standard Deviation)', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('volatility_distribution.png')
        plt.close()
        
        # Find most volatile companies
        most_volatile = {}
        for horizon in ['volatility_1y', 'volatility_3y', 'volatility_5y']:
            top_5 = volatility.nlargest(5, horizon)
            most_volatile[horizon] = top_5[['Symbol', horizon]].values.tolist()
        
        return most_volatile

    def generate_report(self):
        """Generate a comprehensive market analysis report"""
        print("\n=== Market Analysis Report ===")
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data Range: {self.df['date'].min().date()} to {self.df['date'].max().date()}")
        print(f"Number of Companies Analyzed: {len(self.df['Symbol'].unique())}")
        
        # Generate visualizations
        print("\nGenerating market trend visualization...")
        self.analyze_market_trends()
        
        print("\nAnalyzing beta distribution...")
        high_beta = self.analyze_beta_distribution()
        print("\nCompanies with Highest Beta:")
        for years, companies in high_beta.items():
            print(f"\n{years}-Year Horizon:")
            for symbol, beta in companies:
                print(f"  {symbol}: {beta:.2f}")
        
        print("\nAnalyzing volatility...")
        most_volatile = self.analyze_volatility()
        print("\nMost Volatile Companies:")
        for horizon, companies in most_volatile.items():
            print(f"\n{horizon.replace('volatility_', '')}-Year Horizon:")
            for symbol, vol in companies:
                print(f"  {symbol}: {vol:.4f}")
        
        print("\nVisualizations have been saved as:")
        print("- market_trend.png")
        print("- beta_distribution.png")
        print("- volatility_distribution.png")

if __name__ == "__main__":
    try:
        analyzer = MarketAnalyzer()
        analyzer.generate_report()
    except Exception as e:
        print(f"Error during analysis: {e}")
