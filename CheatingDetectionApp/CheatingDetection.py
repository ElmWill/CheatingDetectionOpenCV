import streamlit as st
import cv2
import dlib
import time
import os
from datetime import datetime
from PIL import Image

# Load pre-trained face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Simulated credentials
USER_CREDENTIALS = {"admin": "1234"}

# Create session folder for saving suspicious movement pictures
SESSION_FOLDER = f"suspicious_images/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(SESSION_FOLDER, exist_ok=True)

# Utility function to check if eyes are looking away
def is_eyes_not_looking(landmarks):
    """
    Determines if eyes are not looking at the screen.
    """
    left_eye_x = (landmarks.part(36).x + landmarks.part(39).x) / 2
    right_eye_x = (landmarks.part(42).x + landmarks.part(45).x) / 2
    nose_x = landmarks.part(30).x

    # Check if nose is not centered between eyes
    if abs(nose_x - (left_eye_x + right_eye_x) / 2) > 30:
        return True  # Eyes not centered (looking away)
    return False

# Save suspicious image
def save_suspicious_image(frame, reason):
    """Save the frame as an image in the session folder with a timestamp and reason."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SESSION_FOLDER}/{reason}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    return filename

# Login system
def login():
    """Display login form and handle login."""
    st.sidebar.header("Login Required")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state['logged_in'] = True
            st.sidebar.success("Login Successful!")
            time.sleep(1)
            st.rerun()
        else:
            st.sidebar.error("Invalid Username or Password")

# Main Dashboard
def main_dashboard():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # State variables
    log_messages = []
    no_face_start_time = None
    eyes_not_looking_start_time = None

    def log(message):
        """Add a log entry."""
        log_messages.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}")

    # Layout placeholders
    dashboard_placeholder = st.empty()
    frame_placeholder = st.empty()
    logs_placeholder = st.empty()
    folder_placeholder = st.empty()

    stop_button = st.button("Stop Webcam", key="stop_button")

    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("No frame detected!")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)
        face_detected = len(faces) > 0
        eyes_not_looking = False  # Default

        # Check for face and eyes status
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            landmarks = shape_predictor(gray, face)
            if is_eyes_not_looking(landmarks):
                eyes_not_looking = True

        # Handle no face detection
        if not face_detected:
            if no_face_start_time is None:
                no_face_start_time = time.time()
            elif time.time() - no_face_start_time > 5:
                log("⚠️ Suspicious Movement")
                save_suspicious_image(frame, "no_face")
                no_face_start_time = None  # Reset timer
        else:
            no_face_start_time = None  # Reset if face is detected

        # Handle eyes not looking
        if eyes_not_looking:
            if eyes_not_looking_start_time is None:
                eyes_not_looking_start_time = time.time()
            elif time.time() - eyes_not_looking_start_time > 5:
                log("⚠️ Suspicious Movement")
                save_suspicious_image(frame, "eyes_away")
                eyes_not_looking_start_time = None  # Reset timer
        else:
            eyes_not_looking_start_time = None  # Reset if eyes are looking

        # Display video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Logs
        logs_placeholder.markdown(
            f"""<div style="height: 300px; background-color:#000000; color:white; 
                overflow-y: scroll; padding:10px; border:1px solid #333;">
                {"<br>".join(log_messages[-15:])}
            </div>""",
            unsafe_allow_html=True
        )

    cap.release()
    cv2.destroyAllWindows()
    st.success("Webcam stopped successfully.")

    # Display folder contents
    st.header("Suspicious Movement Images")
    images = [f for f in os.listdir(SESSION_FOLDER) if f.endswith(".jpg")]
    for image_file in images:
        image_path = os.path.join(SESSION_FOLDER, image_file)
        img = Image.open(image_path)
        st.image(img, caption=image_file, width=200)

# Main flow
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

st.title("Suspicious Movement Monitoring System")

if not st.session_state['logged_in']:
    login()
else:
    main_dashboard()
