from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import requests

# Pi code. Need ot check if this could work?

# fastAPI
app = FastAPI(title="Investor Training Orchestrator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# I hate this IP stuff, so I will probably need to find a solution for a good static connection? Idk how I could operate that connection, because wifi I have access to typically has credentials, idk if trying to manage 5ghz is a good idea.
VISION_SERVICE_URL = "http://YOUR_DESKTOP_PC_IP:8001/analyze_face" 
DECISION_LOG_FILE = 'investor_decisions_with_vision.csv'

# load data
try:
    historical_df = pd.read_csv('historical_data.csv')
    class AppState:
        scenario_index = 0
    app_state = AppState()
except FileNotFoundError:
    print("FATAL ERROR: historical_data.csv not found.")
    exit()

# API endpoints
@app.get("/get_next_scenario")
def get_next_scenario():
    if app_state.scenario_index >= len(historical_df):
        raise HTTPException(status_code=404, detail="All scenarios completed!")
    scenario = historical_df.iloc[app_state.scenario_index].to_dict()
    app_state.scenario_index += 1
    return scenario

@app.post("/log_decision")
async def log_decision(
    image: UploadFile = File(...),
    Date: str = Form(...),
    Symbol: str = Form(...),
    Close_Price: float = Form(...),
    Analyst_Buy_Ratio: float = Form(...),
    News_Headline: str = Form(...),
    Investor_Action: str = Form(...)
):
    facial_sentiment = "error"
    try:
        files = {'image': (image.filename, await image.read(), image.content_type)}
        response = requests.post(VISION_SERVICE_URL, files=files, timeout=10)
        response.raise_for_status()
        facial_sentiment = response.json().get('dominant_emotion', 'error')
    except requests.RequestException as e:
        print(f"Error communicating with vision service: {e}")
        facial_sentiment = "vision_service_offline"

    decision_data = {
        'Date': Date, 'Symbol': Symbol, 'Close_Price': Close_Price,
        'Analyst_Buy_Ratio': Analyst_Buy_Ratio, 'News_Headline': News_Headline,
        'Investor_Action': Investor_Action, 'Facial_Sentiment': facial_sentiment
    }
    decision_df = pd.DataFrame([decision_data])
    
    if not os.path.exists(DECISION_LOG_FILE):
        decision_df.to_csv(DECISION_LOG_FILE, index=False, mode='w')
    else:
        decision_df.to_csv(DECISION_LOG_FILE, index=False, mode='a', header=False)
    
    print(f"Logged: {Symbol} -> {Investor_Action} (Facial: {facial_sentiment})")
    return {"status": "success"}
