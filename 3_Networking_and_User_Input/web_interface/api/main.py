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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Global state
trading_bot = None  # TODO: Replace with LiveTrader when available
active_connections = []  # TODO: Replace with List[WebSocket] when available
market_analyzer = None  # TODO: Replace with MarketPatternAnalyzer when available

MODEL_PATH = Path("trained_model/model.h5")
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
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/start_trading")
async def start_trading():
    global trading_bot
    try:
        if trading_bot is None:
            trading_bot = LiveTrader()
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
    return {
        "is_running": trading_bot is not None,
        "last_update": datetime.now().isoformat(),
        "active_symbols": trading_bot.active_symbols if trading_bot else [],
        "current_positions": trading_bot.get_current_positions() if trading_bot else {}
    }

@app.get("/api/performance_metrics")
async def get_performance_metrics():
    if trading_bot is None:
        raise HTTPException(status_code=400, detail="Trading bot is not running")
    try:
        metrics = await trading_bot.get_performance_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
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