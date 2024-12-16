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

def get_head_pose(landmarks):
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

# Initialize webcam feed
cap = cv2.VideoCapture(0)

look_away_logged = {}
head_turn_logged = {}
ALERT_THRESHOLD = 3  # in seconds
HEAD_TURN_THRESHOLD = 15  # degrees for yaw to consider it a significant turn
log_file_path = "gaze_log.txt"

# Open log file
with open(log_file_path, "a") as log_file:
    while True:
        ret, frame = cap.read()
        if not ret:
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
                        log_file.write(f"Person {i+1} is looking away at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        log_file.flush()  # Ensure immediate logging
                else:
                    if i in look_away_logged:  # Reset state when looking back
                        log_file.write(f"Person {i+1} is no longer looking away at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        log_file.flush()  # Ensure immediate logging
                        look_away_logged.pop(i)

            # Head pose estimation
            euler_angles = get_head_pose(landmarks)
            yaw_angle = np.abs(euler_angles[1])

            # Handle head turn detection
            if yaw_angle > HEAD_TURN_THRESHOLD:
                if i not in head_turn_logged:  # Log once when head is turned
                    head_turn_logged[i] = True
                    log_file.write(f"Person {i+1} has turned their head at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_file.flush()  # Ensure immediate logging
            else:
                if i in head_turn_logged:  # Reset state when head turns back
                    log_file.write(f"Person {i+1} has turned back at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_file.flush()  # Ensure immediate logging
                    head_turn_logged.pop(i)

            # Draw rectangles and labels on the frame
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            if yaw_angle > HEAD_TURN_THRESHOLD:
                cv2.putText(frame, "Head Turned!", (face.left(), face.top() - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

