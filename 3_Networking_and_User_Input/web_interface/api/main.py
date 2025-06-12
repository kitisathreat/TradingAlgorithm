from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import json
import asyncio
from typing import Dict, List, Optional
import logging
from datetime import datetime
import sys
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import our trading components
try:
    from _2_Orchestrator_And_ML_Python.live_trader import LiveTrader
    from _2_Orchestrator_And_ML_Python.market_analyzer import MarketAnalyzer
    logger.info("Successfully imported trading components")
except ImportError as e:
    logger.error(f"Error importing trading components: {e}")
    logger.error(f"Python path: {sys.path}")
    logger.error(f"Project root: {PROJECT_ROOT}")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path to the web_interface directory
WEB_INTERFACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(WEB_INTERFACE_DIR, "static")
TEMPLATES_DIR = os.path.join(WEB_INTERFACE_DIR, "templates")

# Create static directory if it doesn't exist
os.makedirs(STATIC_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="Trading Algorithm Control Center")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files with absolute path
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Templates with absolute path
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Global state
trading_bot = None
market_analyzer = None
active_connections = []

MODEL_PATH = Path(os.path.join(WEB_INTERFACE_DIR, "trained_model/model.h5"))
training_in_progress = False
training_progress = 0
training_status_msg = "Not started"

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                self.disconnect(connection)

manager = ConnectionManager()

# Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        return HTMLResponse(content=f"""
            <html>
                <head><title>Trading Algorithm Control Center</title></head>
                <body>
                    <h1>Trading Algorithm Control Center</h1>
                    <p>Error loading interface: {str(e)}</p>
                    <p>Please check the server logs for more information.</p>
                </body>
            </html>
        """)

@app.post("/api/start_trading")
async def start_trading():
    global trading_bot
    try:
        if trading_bot is None:
            trading_bot = LiveTrader()  # Initialize the actual LiveTrader
            await trading_bot.initialize()
            await trading_bot.start()
            return {"status": "success", "message": "Trading bot started successfully"}
        else:
            return {"status": "error", "message": "Trading bot is already running"}
    except Exception as e:
        logger.error(f"Error starting trading bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stop_trading")
async def stop_trading():
    global trading_bot
    try:
        if trading_bot is not None:
            await trading_bot.stop()
            trading_bot = None
            return {"status": "success", "message": "Trading bot stopped successfully"}
        else:
            return {"status": "error", "message": "Trading bot is not running"}
    except Exception as e:
        logger.error(f"Error stopping trading bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trading_status")
async def get_trading_status():
    global trading_bot
    if not trading_bot:
        return {
            "is_running": False,
            "last_update": datetime.now().isoformat(),
            "active_symbols": [],
            "current_positions": {},
            "is_placeholder": False
        }
    
    try:
        positions = trading_bot.trading_states
        return {
            "is_running": True,
            "last_update": datetime.now().isoformat(),
            "active_symbols": list(positions.keys()),
            "current_positions": {
                symbol: {
                    "current_position": state.current_position,
                    "last_update": state.last_update.isoformat(),
                    "last_decision": state.last_decision.__dict__ if state.last_decision else None
                }
                for symbol, state in positions.items()
            },
            "is_placeholder": False
        }
    except Exception as e:
        logger.error(f"Error getting trading status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/performance_metrics")
async def get_performance_metrics():
    if trading_bot is None:
        raise HTTPException(status_code=400, detail="Trading bot is not running")
    try:
        # Get performance metrics from the trading bot
        metrics = {}
        for symbol, state in trading_bot.trading_states.items():
            if state.last_decision:
                metrics[symbol] = {
                    "total_return": state.last_decision.total_return if hasattr(state.last_decision, 'total_return') else 0.0,
                    "sharpe_ratio": state.last_decision.sharpe_ratio if hasattr(state.last_decision, 'sharpe_ratio') else 0.0,
                    "max_drawdown": state.last_decision.max_drawdown if hasattr(state.last_decision, 'max_drawdown') else 0.0,
                    "win_rate": state.last_decision.win_rate if hasattr(state.last_decision, 'win_rate') else 0.0
                }
        return metrics
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market_analysis")
async def get_market_analysis():
    global market_analyzer
    try:
        if market_analyzer is None:
            market_analyzer = MarketAnalyzer()
        
        # Generate market analysis
        high_beta = market_analyzer.analyze_beta_distribution()
        volatility = market_analyzer.analyze_volatility()
        
        return {
            "high_beta_companies": high_beta,
            "volatile_companies": volatility,
            "charts": {
                "market_trend": "/static/market_trend.png",
                "beta_distribution": "/static/beta_distribution.png",
                "volatility_distribution": "/static/volatility_distribution.png"
            }
        }
    except Exception as e:
        logger.error(f"Error performing market analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming WebSocket messages if needed
            await manager.broadcast(f"Message received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Background task to broadcast trading updates
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_trading_updates())

async def broadcast_trading_updates():
    while True:
        if trading_bot is not None:
            try:
                status = await get_trading_status()
                await manager.broadcast(json.dumps(status))
            except Exception as e:
                logger.error(f"Error broadcasting trading updates: {e}")
        await asyncio.sleep(1)  # Update every second

@app.get("/api/model_status")
async def model_status():
    exists = MODEL_PATH.exists()
    return {"exists": exists}

@app.post("/api/start_training")
async def start_training():
    global training_in_progress, training_progress, training_status_msg
    if training_in_progress:
        return {"status": "error", "message": "Training already in progress"}
    training_in_progress = True
    training_progress = 0
    training_status_msg = "Training started"
    # Here you will implement the actual training logic
    # For now, simulate progress
    import threading, time
    def train():
        global training_progress, training_status_msg, training_in_progress
        for i in range(1, 11):
            time.sleep(1)
            training_progress = i * 10
            training_status_msg = f"Training... {training_progress}%"
        training_status_msg = "Training complete"
        training_in_progress = False
        # Simulate model file creation
        MODEL_PATH.parent.mkdir(exist_ok=True)
        MODEL_PATH.touch()
    threading.Thread(target=train, daemon=True).start()
    return {"status": "success", "message": "Training started"}

@app.get("/api/training_status")
async def training_status():
    return {
        "in_progress": training_in_progress,
        "progress": training_progress,
        "status_msg": training_status_msg
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

st.title("Trading Algorithm Control Center")

# Sidebar controls
st.sidebar.title("Controls")
if st.sidebar.button("Start Trading"):
    # Start trading logic
    pass

# Main dashboard
col1, col2 = st.columns(2)

with col1:
    st.subheader("Performance Metrics")
    # Display metrics

with col2:
    st.subheader("Active Positions")
    # Display positions

# Charts
st.subheader("Market Analysis")
# Display interactive charts 