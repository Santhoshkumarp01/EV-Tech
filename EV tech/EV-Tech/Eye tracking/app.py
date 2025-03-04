import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import winsound
import warnings
import time
from flask import Flask, render_template, Response
from collections import deque
from concurrent.futures import ThreadPoolExecutor

warnings.simplefilter("ignore")

app = Flask(__name__)

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
executor = ThreadPoolExecutor(max_workers=3)

def speak(alert_text):
    """ Speaks alert messages in a separate thread """
    try:
        engine.say(alert_text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")

# Initialize MediaPipe models
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Constants
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EYE_AR_THRESH = 0.25  
EYE_AR_CONSEC_FRAMES = 15  # Reduced for faster detection
COUNTER = deque(maxlen=EYE_AR_CONSEC_FRAMES)  

def estimate_distance(height):
    """ Estimates distance based on the detected height of the person """
    return 4000 / height if height > 0 else float('inf')

def eye_aspect_ratio(eye_points, landmarks):
    """ Calculates the eye aspect ratio (EAR) to detect drowsiness """
    try:
        A = np.linalg.norm(np.array(landmarks[eye_points[1]]) - np.array(landmarks[eye_points[5]]))
        B = np.linalg.norm(np.array(landmarks[eye_points[2]]) - np.array(landmarks[eye_points[4]]))
        C = np.linalg.norm(np.array(landmarks[eye_points[0]]) - np.array(landmarks[eye_points[3]]))
        return (A + B) / (2.0 * C)
    except Exception:
        return 1.0  

def generate_frames():
    """ Captures frames, processes pose and face landmarks, and applies alerts """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            # Process pose and face landmarks
            results_face = face_mesh.process(rgb_frame)
            results_pose = pose.process(rgb_frame)

            if results_pose.pose_landmarks:
                try:
                    landmarks = results_pose.pose_landmarks.landmark
                    nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y * h
                    ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * h
                    person_height = abs(ankle_y - nose_y)
                    distance = estimate_distance(person_height)

                    cv2.putText(frame, f"Distance: {distance:.2f} m", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if distance <= 3:
                        executor.submit(speak, "Alert! Person is too close.")
                except Exception as e:
                    print(f"Pose Processing Error: {e}")

            if results_face.multi_face_landmarks:
                try:
                    for face_landmarks in results_face.multi_face_landmarks:
                        landmarks = [(int(l.x * w), int(l.y * h)) for l in face_landmarks.landmark]
                        left_ear = eye_aspect_ratio(LEFT_EYE, landmarks)
                        right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks)
                        ear = (left_ear + right_ear) / 2.0

                        if ear < EYE_AR_THRESH:
                            COUNTER.append(1)
                            if len(COUNTER) >= EYE_AR_CONSEC_FRAMES:
                                cv2.putText(frame, "DROWSINESS ALERT!", (100, 150),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                                executor.submit(winsound.Beep, 1000, 1000)
                        else:
                            COUNTER.clear()
                except Exception as e:
                    print(f"Face Processing Error: {e}")

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
