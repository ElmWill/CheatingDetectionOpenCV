import streamlit as st
import cv2
import dlib
import numpy as np
import time

# Load the face detector and facial landmarks predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_gaze_ratio(eye_points, facial_landmarks, frame_gray):
    eye_region = np.array([(facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in eye_points], np.int32)
    mask = np.zeros((frame_gray.shape[0], frame_gray.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [eye_region], 255)
    eye_frame = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)
    _, thresh_eye = cv2.threshold(eye_frame, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(contour)
        gaze_ratio = (x + w // 2) / frame_gray.shape[1]
        return gaze_ratio
    return None

def get_head_pose(landmarks, frame):
    model_points = np.array([
        (0.0, 0.0, 0.0),         # Nose tip
        (0.0, -330.0, -65.0),    # Chin
        (-225.0, 170.0, -135.0), # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0), # Left Mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ], dtype='double')

    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
        (landmarks.part(8).x, landmarks.part(8).y),    # Chin
        (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
        (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
        (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
        (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
    ], dtype='double')

    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                               [0, focal_length, center[1]],
                               [0, 0, 1]], dtype='double')

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    rmat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rmat, translation_vector))
    euler_angles = cv2.decomposeProjectionMatrix(pose_mat)[6]

    return euler_angles

# Streamlit App
st.title("Cheating Detection")

# Initialize webcam feed
cap = cv2.VideoCapture(0)

look_away_logged = {}
head_turn_logged = {}
HEAD_TURN_THRESHOLD = 15  # degrees for yaw to consider it a significant turn

# Initialize log messages list
log_messages = []

def log(message):
    log_messages.append(message)

# Create a button to start/stop the webcam feed
start_stop_button = st.button("Start", key="start_stop_button")

# Create two columns for layout
col1, col2 = st.columns([2, 1])  # Adjust the ratio as needed

with col1:
    frame_placeholder = st.empty()  # Placeholder for video feed

with col2:
    st.subheader("Live Log")
    # Set a fixed height for the log display and enable scrolling
    log_display = st.empty()

webcam_running = False

if start_stop_button:
    webcam_running = not webcam_running  # Toggle webcam state

while webcam_running:
    ret, frame = cap.read()
    if not ret:
        st.write("No frame detected!")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for i, face in enumerate(faces):
        landmarks = predictor(gray, face)

        left_eye_gaze_ratio = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, gray)
        right_eye_gaze_ratio = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, gray)

        # Handle look away detection
        if left_eye_gaze_ratio is not None and right_eye_gaze_ratio is not None:
            avg_gaze_ratio = (left_eye_gaze_ratio + right_eye_gaze_ratio) / 2

            if avg_gaze_ratio < 0.3:  # Adjust threshold
                if i not in look_away_logged:  # Log once when looking away
                    look_away_logged[i] = True
            else:
                if i in look_away_logged:  # Reset state when looking 
                    look_away_logged.pop(i)

        # Head pose estimation
        euler_angles = get_head_pose(landmarks, frame)
        yaw_angle = np.abs(euler_angles[1])

        # Handle head turn detection
        if yaw_angle > HEAD_TURN_THRESHOLD:
            if i not in head_turn_logged:  # Log once when head is turned
                head_turn_logged[i] = True
                log(f"Person {i+1} has turned their head at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            if i in head_turn_logged :  # Reset state when head turns back
                head_turn_logged.pop(i)

        # Draw rectangles and labels on the frame
        cv2.rectangle(frame , (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        if yaw_angle > HEAD_TURN_THRESHOLD:
            cv2.putText(frame, "SUS!", (face.left(), face.top() - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Convert the frame to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")

    # Update the log display
    log_display.markdown(
        f"""
        <div style="height: 400px; overflow-y: scroll;">
        {"<br>".join(log_messages)}
        </div>
        """, unsafe_allow_html=True
    )

cap.release()
cv2.destroyAllWindows()
