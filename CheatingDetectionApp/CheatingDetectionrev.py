import streamlit as st
import cv2
import numpy as np
import time

# Function for movement detection
def detect_movement(prev_frame, current_frame, threshold=3000):
    """Detect movement by comparing consecutive frames."""
    diff = cv2.absdiff(prev_frame, current_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    movement_score = np.sum(thresh) / 255
    return movement_score > threshold

# Simulated credentials
USER_CREDENTIALS = {"admin": "1234"}  # Replace with secure credentials

# Login system
def login():
    """Display login form and handle login."""
    st.sidebar.header("Login Required")
    username = st.sidebar.text_input("Username", key="username")
    password = st.sidebar.text_input("Password", type="password", key="password")

    if st.sidebar.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state['logged_in'] = True
            st.sidebar.success("Login Successful!")
            time.sleep(1)
            st.experimental_rerun()
        else:
            st.sidebar.error("Invalid Username or Password")

# Main dashboard
def main_dashboard():
    

    # Initialize webcam feed
    cap = cv2.VideoCapture(0)

    # State variables
    prev_frame = None
    movement_detected = False
    log_messages = []

    def log(message):
        log_messages.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}")

    # Layout
    dashboard_placeholder = st.empty()
    frame_placeholder = st.empty()
    logs_placeholder = st.empty()

    stop_button = st.button("Stop Webcam", key="stop_button")

    while stop_button is False:
        ret, frame = cap.read()
        if not ret:
            st.warning("No frame detected!")
            break
        
        # Movement detection
        if prev_frame is not None:
            movement_detected = detect_movement(prev_frame, frame)
            if movement_detected:
                log("Movement Detected!")
        prev_frame = frame.copy()

        # Dashboard
        dashboard_placeholder.markdown(f"""
        <div style="background-color:#000000; color:white; padding:10px; border-radius:5px; text-align:center;">
            <h2 style="margin:0;">Movement board</h2>
            <h3 style="margin:0;">Movement: {"DETECTED" if movement_detected else "NO MOVEMENT"}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Display video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Logs
        logs_placeholder.markdown(
            f"""<div style="height: 300px; background-color:#000000; color:white; 
                overflow-y: scroll; padding:10px; border:1px solid #333;">
                {"<br>".join(log_messages)}
            </div>""",
            unsafe_allow_html=True
        )
    
    cap.release()
    cv2.destroyAllWindows()
    st.success("Webcam stopped successfully.")

# Main flow
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

st.title("Cheating Detection System")

if not st.session_state['logged_in']:
    if st.button("Start Webcam"):
        st.session_state['trigger_login'] = True
    if st.session_state.get('trigger_login', False):
        login()
else:
    main_dashboard()
