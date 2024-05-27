import cv2
import mediapipe as mp
import numpy as np
import pickle
import simpleaudio as sa
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from threading import Thread
from queue import Queue
import logging
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger()

app = FastAPI()

# Load the KNN model
try:
    with open('knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    logger.info("KNN model loaded successfully.")
except FileNotFoundError as e:
    logger.error("KNN model file not found: %s", e)
    raise HTTPException(status_code=500, detail="KNN model file not found.")
except Exception as e:
    logger.error("Error loading KNN model: %s", e)
    raise HTTPException(status_code=500, detail="Error loading KNN model.")

# Load the sound files
sound_files = {}
try:
    sound_files = {
        "Sit Up Straight": sa.WaveObject.from_wave_file('sounds/sit_up_straight.wav'),
        "Straighten Head": sa.WaveObject.from_wave_file('sounds/straighten_your_head.wav')
    }
    logger.info("Sound files loaded successfully.")
except FileNotFoundError as e:
    logger.error("Sound file not found: %s", e)
    raise HTTPException(status_code=500, detail="Sound file not found.")
except Exception as e:
    logger.error("Error loading sound files: %s", e)
    raise HTTPException(status_code=500, detail="Error loading sound files.")

# Initialize MediaPipe
mp_pose = mp.solutions.pose

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Function to calculate midpoint
def calc_midpoint(point1, point2):
    return [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]

# Function to calculate angle between three points (in degrees)
def calc_angle(pointA, pointB, pointC):
    AB = np.array(pointB) - np.array(pointA)
    BC = np.array(pointB) - np.array(pointC)
    cos_angle = np.dot(AB, BC) / (np.linalg.norm(AB) * np.linalg.norm(BC))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

# Function to extract features from landmarks
def extract_features(landmarks):
    nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
    
    dist_left_ear_shoulder = euclidean_distance(left_ear, left_shoulder)
    dist_right_ear_shoulder = euclidean_distance(right_ear, right_shoulder)
    shoulder_midpoint = calc_midpoint(left_shoulder, right_shoulder)
    dist_shoulder_nose = euclidean_distance(shoulder_midpoint, nose)
    angle_nose_left_shoulder_right_shoulder = calc_angle(nose, left_shoulder, right_shoulder)
    
    return [dist_left_ear_shoulder, dist_right_ear_shoulder, dist_shoulder_nose, angle_nose_left_shoulder_right_shoulder]

# Function to process the video frame by frame
def process_video(queue):
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        object_detected = False
        prev_predicted_label = None
        current_sound = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture image from camera!")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                try:
                    landmarks = results.pose_landmarks.landmark
                    object_detected = True
                    features = extract_features(landmarks)
                    predicted_label = knn_model.predict([features])[0]
                    labels = {0: "Looks Good", 1: "Sit Up Straight", 2: "Straighten Head"}
                    
                    cv2.putText(image, f'Predicted Label: {labels[predicted_label]}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    if predicted_label != prev_predicted_label:
                        if current_sound is not None:
                            current_sound.stop()
                        if labels[predicted_label] != "Looks Good":
                            current_sound = sound_files[labels[predicted_label]].play()
                        prev_predicted_label = predicted_label

                except Exception as e:
                    logger.error("Exception during pose processing: %s", e)
                    
                mp.solutions.drawing_utils.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
            
            else:
                cv2.putText(image, "No object detected!", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                if object_detected:
                    if current_sound is not None:
                        current_sound.stop()
                    object_detected = False
                    prev_predicted_label = None
            
            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            queue.put(frame)

        cap.release()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Posture Correction API"}

@app.get("/video_feed")
def video_feed():
    def generate():
        queue = Queue()
        thread = Thread(target=process_video, args=(queue,))
        thread.start()
        
        while True:
            frame = queue.get()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error("Exception during uvicorn run: %s", e)
