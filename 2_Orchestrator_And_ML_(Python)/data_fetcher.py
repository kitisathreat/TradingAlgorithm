import requests
import config
import time
import alpaca_trade_api as tradeapi

# In-memory cache to limit API calls
cache = {}
CACHE_TTL_SECONDS = 900 # 15 minutes

class FMPFetcher:
    def __init__(self):
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def get_analyst_ratings(self, symbol: str) -> dict:
        cache_key = f"{symbol}_ratings"
        if cache_key in cache and (time.time() - cache[cache_key]['timestamp']) < CACHE_TTL_SECONDS:
            return cache[cache_key]['data']
        
        print(f"[FMP] Fetching new ratings for {symbol}...")
        try:
            url = f"{self.base_url}/analyst-estimates/{symbol}?limit=1&apikey={config.FMP_API_KEY}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if not data: return {"buy_ratio": 0.0}
            
            ratings = data[0]
            buy = ratings.get('ratingBuy', 0) + ratings.get('ratingOverweight', 0) + ratings.get('ratingStrongBuy', 0)
            sell = ratings.get('ratingSell', 0) + ratings.get('ratingUnderweight', 0) + ratings.get('ratingStrongSell', 0)
            hold = ratings.get('ratingHold', 0)
            total = buy + sell + hold
            buy_ratio = buy / total if total > 0 else 0.0
            
            processed_data = {"buy_ratio": buy_ratio}
            cache[cache_key] = {'timestamp': time.time(), 'data': processed_data}
            return processed_data
        except requests.RequestException as e:
            print(f"Error fetching FMP data: {e}")
            return {"buy_ratio": 0.0}

class NewsFetcher:
    def __init__(self, alpaca_api_client: tradeapi.REST):
        self.alpaca_api = alpaca_api_client
    
    def get_latest_headline(self, symbol: str) -> str:
        if config.POLYGON_API_KEY:
            return self._fetch_from_polygon(symbol)
        return self._fetch_from_alpaca(symbol)

    def _fetch_from_polygon(self, symbol: str) -> str:
        print(f"[News] Using Polygon.io...")
        try:
            url = f"https://api.polygon.io/v2/reference/news?ticker={symbol}&limit=1&apiKey={config.POLYGON_API_KEY}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            articles = response.json().get('results', [])
            return articles[0].get('title', '') if articles else ""
        except requests.RequestException:
            return self._fetch_from_alpaca(symbol)

    def _fetch_from_alpaca(self, symbol: str) -> str:
        print(f"[News] Using Alpaca fallback...")
        try:
            news = self.alpaca_api.get_news(symbol=symbol, limit=1)
            return news[0].headline if news else ""
        except Exception:
            return ""