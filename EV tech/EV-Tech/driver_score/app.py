import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
def speak(alert_text):
    threading.Thread(target=lambda: (engine.say(alert_text), engine.runAndWait()), daemon=True).start()

# Initialize MediaPipe Pose and Face Mesh
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Score Parameters
initial_score = 10  # Start with a perfect score
initial_credit = 0   # Start with zero credit points
penalties = {"drowsiness": 2, "distraction": 2, "over_speeding": 3, "helmet":2}
rewards = {"safe_speed": 3, "alertness": 2, "focus": 2, "helmet":2}

# Flags for voice alerts
drowsiness_alerted = False
distraction_alerted = False
over_speed_alerted = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['GET'])
def start_session():
    try:
        has_helmet = request.form.get("helmet") == "yes"
        final_score, final_credit = run_detection(has_helmet)
        return render_template('result.html', score=final_score, credit=final_credit)
    except Exception as e:
        return f"Error: {str(e)}", 500

def run_detection(has_helmet):
    global initial_score, initial_credit
    global drowsiness_alerted, distraction_alerted, over_speed_alerted

    SCORE = initial_score
    CREDIT = initial_credit

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return SCORE, CREDIT  # Return default score and credit

    prev_nose_x = None
    prev_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            results_face = mp_face_mesh.process(rgb_frame)
            results_pose = pose.process(rgb_frame)

            # Speed Estimation (Using Nose Movement)
            if results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark
                nose_x = landmarks[mp_pose.PoseLandmark.NOSE].x * w
                curr_time = time.time()

                if prev_nose_x is not None and (curr_time - prev_time) > 0:
                    speed = abs(nose_x - prev_nose_x) / (curr_time - prev_time)
                    speed = min(speed, 100)

                    if speed > 60 and not over_speed_alerted:
                        SCORE -= penalties["over_speeding"]
                        speak("You are over speeding. Please slow down.")
                        over_speed_alerted = True
                    else:
                        CREDIT += rewards["safe_speed"]

                prev_nose_x = nose_x
                prev_time = curr_time

            # Drowsiness Detection (Based on Head Tilts)
            if results_pose.pose_landmarks:
                left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE].y * h
                right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE].y * h
                nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y * h

                if abs(left_eye - right_eye) < 5 and nose_y > (h * 0.55):  # Head nodding down
                    if not drowsiness_alerted:
                        SCORE -= penalties["drowsiness"]
                        speak("You seem drowsy. Please take a break.")
                        drowsiness_alerted = True
                else:
                    CREDIT += rewards["alertness"]
                    drowsiness_alerted = False

            # Distraction Detection (Looking Away Detection)
            if results_pose.pose_landmarks:
                left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR].x * w
                right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x * w
                nose_x = landmarks[mp_pose.PoseLandmark.NOSE].x * w

                if nose_x < left_ear or nose_x > right_ear:
                    if not distraction_alerted:
                        SCORE -= penalties["distraction"]
                        speak("Please keep your eyes on the road.")
                        distraction_alerted = True
                else:
                    CREDIT += rewards["focus"]
                    distraction_alerted = False

            cv2.imshow("Driver Safety Dashboard", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error during processing: {str(e)}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return max(SCORE, 0), max(CREDIT, 0)

if __name__ == '__main__':
    app.run(debug=True)
