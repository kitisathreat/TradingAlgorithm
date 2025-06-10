from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from deepface import DeepFace

# This script runs on a powerful PC, not the Raspberry Pi.
# Command to run: uvicorn vision_service:app --host 0.0.0.0 --port 8001

app = FastAPI(title="Facial Sentiment Analysis Service")

@app.post("/analyze_face")
async def analyze_face(image: UploadFile = File(...)):
    """
    Receives an image, analyzes it for the dominant emotion, and returns the result.
    """
    dominant_emotion = "error"
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        analysis = DeepFace.analyze(
            img_path=img,
            actions=['emotion'],
            enforce_detection=False # Don't fail if a face isn't perfectly detected
        )
        
        dominant_emotion = analysis[0]['dominant_emotion']
        print(f"[Vision Service] Analysis successful. Emotion: {dominant_emotion}")

    except Exception as e:
        print(f"[Vision Service] Error during analysis: {e}")
        dominant_emotion = "error"

    return {"dominant_emotion": dominant_emotion}