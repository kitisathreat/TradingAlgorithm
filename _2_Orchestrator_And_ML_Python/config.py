import os
from dotenv import load_dotenv

load_dotenv()

APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
APCA_BASE_URL = os.getenv("APCA_BASE_URL", "https://paper-api.alpaca.markets")
FMP_API_KEY = os.getenv("FMP_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

SYMBOLS_TO_TRADE = ['AAPL', 'MSFT', 'NVDA']