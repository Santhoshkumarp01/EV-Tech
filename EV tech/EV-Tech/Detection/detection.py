import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
import pygame  # For playing the external alarm sound
import winsound  # For Beep sound
import warnings

warnings.simplefilter("ignore")  # Suppress warnings
engine = pyttsx3.init()
pygame.mixer.init()

ALARM_SOUND_PATH = "Alert.wav"  # Specify the path to your downloaded alarm file

def play_alarm():
    """Play the external alarm sound."""
    pygame.mixer.music.load(ALARM_SOUND_PATH)
    pygame.mixer.music.play()

def speak(alert_text):
    """Voice alert function using threading."""
    threading.Thread(target=lambda: (engine.say(alert_text), engine.runAndWait()), daemon=True).start()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EYE_AR_THRESH = 0.23  # Optimized threshold for better accuracy
EYE_AR_CONSEC_FRAMES = 15  # Reduced frames for faster detection
COUNTER = 0
alert_triggered_drowsy = False

alert_triggered_look = False
last_alert_time_look = 0
alert_triggered_dist = False

def estimate_distance(height):
    """Estimate distance based on face height in pixels."""
    return float('inf') if height == 0 else 4000 / height

def eye_aspect_ratio(eye_points, landmarks):
    """Calculate the eye aspect ratio (EAR)."""
    A = np.linalg.norm(np.array(landmarks[eye_points[1]]) - np.array(landmarks[eye_points[5]]))
    B = np.linalg.norm(np.array(landmarks[eye_points[2]]) - np.array(landmarks[eye_points[4]]))
    C = np.linalg.norm(np.array(landmarks[eye_points[0]]) - np.array(landmarks[eye_points[3]]))
    return (A + B) / (2.0 * C)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    results_face = mp_face_mesh.process(rgb_frame)
    results_pose = pose.process(rgb_frame)

    # Distance estimation
    if results_pose.pose_landmarks:
        landmarks = results_pose.pose_landmarks.landmark
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y * h
        ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * h

        person_height = abs(ankle_y - nose_y)
        distance = estimate_distance(person_height)

        cv2.putText(frame, f"Distance: {distance:.2f} m", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if distance <= 3:
            cv2.putText(frame, "ALERT! Person too close!", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            if not alert_triggered_dist:
                speak("Alert! Person is too close.")
                alert_triggered_dist = True
        else:
            alert_triggered_dist = False

    # Drowsiness detection
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            landmarks = [(int(l.x * w), int(l.y * h)) for l in face_landmarks.landmark]

            left_ear = eye_aspect_ratio(LEFT_EYE, landmarks)
            right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks)
            ear = (left_ear + right_ear) / 2.0

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (100, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    if not alert_triggered_drowsy:
                        speak("You are feeling drowsy. Stay alert!")
                        play_alarm()
                        alert_triggered_drowsy = True
            else:
                COUNTER = 0
                alert_triggered_drowsy = False
                pygame.mixer.music.stop()

            for point in LEFT_EYE + RIGHT_EYE:
                cv2.circle(frame, landmarks[point], 2, (0, 255, 0), -1)

    # Look at the road alert
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[1]
            x, y = int(nose_tip.x * w), int(nose_tip.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            if 0.4 < nose_tip.x < 0.6:
                alert_triggered_look = False
            else:
                if not alert_triggered_look or time.time() - last_alert_time_look > 2:
                    last_alert_time_look = time.time()
                    alert_triggered_look = True
                    speak("Please look at the road")
                    winsound.Beep(1000, 700)  # **Beep sound added**
                    cv2.putText(frame, "LOOK AT THE ROAD!", (50, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Safety & Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
