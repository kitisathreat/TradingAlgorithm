"""
Main entry point for the Trading Algorithm
This module coordinates between the C++ decision engine and Python components.
"""

from pathlib import Path
import sys
import logging

# Add the orchestrator directory to Python path
REPO_ROOT = Path(__file__).parent
ORCHESTRATOR_PATH = REPO_ROOT / "_2_Orchestrator_And_ML_Python"
sys.path.append(str(ORCHESTRATOR_PATH))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_algorithm.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the trading algorithm"""
    try:
        # Import here to handle potential ImportError gracefully
        from _2_Orchestrator_And_ML_Python.live_trader import LiveTrader
        from _2_Orchestrator_And_ML_Python.market_analyzer import MarketAnalyzer
        
        # Initialize components
        market_analyzer = MarketAnalyzer()
        trading_bot = LiveTrader(market_analyzer)
        
        logger.info("Trading Algorithm initialized successfully")
        return trading_bot, market_analyzer
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.info("Please ensure the C++ decision engine is built and installed")
        return None, None
    except Exception as e:
        logger.error(f"Error initializing trading algorithm: {e}")
        return None, None

if __name__ == "__main__":
    trading_bot, market_analyzer = main()
    if trading_bot and market_analyzer:
        logger.info("Trading Algorithm is ready to run")
    else:
        logger.error("Trading Algorithm failed to initialize") 