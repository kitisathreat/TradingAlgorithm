from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import cv2
import numpy as np
import onnxruntime as ort
import os

# --- Initialize FastAPI App ---
app = FastAPI(title="Advanced Facial Analysis Service")

# Only runs after serer startup

# 1. OpenCV's Caffe
proto_path = os.path.join(cv2.dnn.readNetFromCaffe.__loader__.path, "..", "model", "deploy.prototxt")
model_path = os.path.join(cv2.dnn.readNetFromCaffe.__loader__.path, "..", "model", "res10_300x300_ssd_iter_140000.caffemodel")
face_detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# ONNX Emotion Recognition Model
emotion_model_path = "emotion-ferplus-8.onnx"
if not os.path.exists(emotion_model_path):
    raise FileNotFoundError(f"Emotion model not found at {emotion_model_path}.  download it.")

emotion_classifier = ort.InferenceSession(emotion_model_path)
EMOTION_LIST = ['neutral', 'happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'contempt'] #again, can add more, this was just base

#print("Face detector and emotion classifier loaded successfully.")
#test statement

# Helper for single image 
def analyze_single_image(image: np.ndarray):
    """Analyzes one image to find a face and its emotion."""
    (h, w) = image.shape[:2]
    # Create a blob 
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_detector.setInput(blob)
    detections = face_detector.forward()

    # Highest confidence
    best_detection_index = np.argmax(detections[0, 0, :, 2])
    confidence = detections[0, 0, best_detection_index, 2]

    if confidence < 0.5: # Confidence threshold
        return None 

    # Bounding box and face ROI
    box = detections[0, 0, best_detection_index, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    
    face_roi = image[startY:endY, startX:endX]
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    resized_face = cv2.resize(gray_face, (64, 64))
    
    # Prepare image for the ONNX model
    processed_input = resized_face.astype(np.float32).reshape(1, 1, 64, 64)
    
    # Inference
    input_name = emotion_classifier.get_inputs()[0].name
    output_name = emotion_classifier.get_outputs()[0].name
    result = emotion_classifier.run([output_name], {input_name: processed_input})[0]
    
    emotion_index = np.argmax(result)
    dominant_emotion = EMOTION_LIST[emotion_index]
    
    return {
        "dominant_emotion": dominant_emotion,
        "confidence": float(confidence),
        "box_size": (endX - startX) * (endY - startY) # Area of the face box
    }

# Might need to edit this? Not entirely sure if this API endpoint is a good idea, maybe just calling this as a class method from an object would be easier? idk
@app.post("/analyze_vision")
async def analyze_vision(
    image_left: UploadFile = File(...),
    image_right: UploadFile = File(...),
    depth_map: UploadFile = File(None) # Kinect input provision
):
    """
    Receives dual camera images, analyzes both, and returns the best result.
    Optionally accepts a depth map for future use.
    """
    # Decode images
    contents_left = await image_left.read()
    nparr_left = np.frombuffer(contents_left, np.uint8)
    img_left = cv2.imdecode(nparr_left, cv2.IMREAD_COLOR)

    contents_right = await image_right.read()
    nparr_right = np.frombuffer(contents_right, np.uint8)
    img_right = cv2.imdecode(nparr_right, cv2.IMREAD_COLOR)

    # Analyze both images
    result_left = analyze_single_image(img_left)
    result_right = analyze_single_image(img_right)

    # Prefers detection with the larger face for now
    best_result = None
    if result_left and result_right:
        best_result = result_left if result_left['box_size'] >= result_right['box_size'] else result_right
    elif result_left:
        best_result = result_left
    elif result_right:
        best_result = result_right
    else:
        # No face detected in either image
        return {"dominant_emotion": "not_found", "confidence": 0.0}

    # --- Placeholder for future Kinect Depth Data Fusion ---
    if depth_map:
        depth_data = await depth_map.read()
        print(f"[Vision Service] Received depth map of size {len(depth_data)} bytes. Processing logic can be added here.")
        # depth data to calculate for distortion from wide angle pi cameras
        best_result['depth_processed'] = True # Add a flag indicating depth was used

    return {
        "dominant_emotion": best_result['dominant_emotion'],
        "confidence": best_result['confidence']
    }
