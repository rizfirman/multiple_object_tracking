import cv2
import mediapipe as mp
from pythonosc import udp_client

# Setup OSC client
osc_client = udp_client.SimpleUDPClient("127.0.0.1", 8000)  # Localhost, port 8000 for TouchDesigner

# Setup Mediapipe for pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (Mediapipe works with RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for pose detection
    result = pose.process(rgb_frame)

    # If pose landmarks are detected
    if result.pose_landmarks:
        # Extract landmark positions for legs and feet
        left_ankle = result.pose_landmarks.landmark[27]  # Left ankle
        right_ankle = result.pose_landmarks.landmark[28]  # Right ankle
        left_foot_index = result.pose_landmarks.landmark[31]  # Left foot index
        right_foot_index = result.pose_landmarks.landmark[32]  # Right foot index

        # Check if both feet are detected properly (confidence threshold)
        if left_ankle.visibility > 0.5 and right_ankle.visibility > 0.5:
            # Send OSC message for left ankle and foot
            osc_client.send_message("/pose/foot/left_ankle", [left_ankle.x, left_ankle.y, left_ankle.z])
            osc_client.send_message("/pose/foot/left_foot_index", [left_foot_index.x, left_foot_index.y, left_foot_index.z])

            # Send OSC message for right ankle and foot
            osc_client.send_message("/pose/foot/right_ankle", [right_ankle.x, right_ankle.y, right_ankle.z])
            osc_client.send_message("/pose/foot/right_foot_index", [right_foot_index.x, right_foot_index.y, right_foot_index.z])

        # Draw the landmarks on the frame
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame with pose detection
    cv2.imshow('Foot Tracking', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
