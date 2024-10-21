# a both of hand detector

# import cv2
# import mediapipe as mp
# from pythonosc import udp_client

# # Setup OSC client
# osc_client = udp_client.SimpleUDPClient("127.0.0.1", 8000)  # 127.0.0.1 is localhost, 8000 is the port TouchDesigner will listen to

# # Setup Mediapipe for hand tracking
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
# mp_draw = mp.solutions.drawing_utils

# # Start capturing video from webcam
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         print("Ignoring empty camera frame.")
#         break

#     # Flip the frame horizontally
#     frame = cv2.flip(frame, 1)

#     # Convert the frame to RGB (Mediapipe works with RGB)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process the frame for hand tracking
#     result = hands.process(rgb_frame)

#     # Draw hand landmarks if hand is detected
#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             # Send the landmark positions via OSC
#             for i, landmark in enumerate(hand_landmarks.landmark):
#                 osc_client.send_message(f"/hand/{i}", [landmark.x, landmark.y, landmark.z])

#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     # Display the frame with hand tracking
#     cv2.imshow('Hand Tracking', frame)

#     # Break the loop on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Quitting...")
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# only one hand detector
import cv2
import mediapipe as mp
from pythonosc import udp_client

# Setup OSC client
osc_client = udp_client.SimpleUDPClient("127.0.0.1", 8000)  # 127.0.0.1 is localhost, 8000 is the port TouchDesigner will listen to

# Setup Mediapipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
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

    # Process the frame for hand tracking
    result = hands.process(rgb_frame)

    # If hand landmarks are detected
    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, hand_handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            hand_label = hand_handedness.classification[0].label  # 'Left' or 'Right'
            # print(hand_label)

            # Only process if the detected hand is the right hand
            # if hand_label == 'Right':
            if hand_label == 'Left':
                # Detect only the landmarks for the index finger (telunjuk), which are 5 to 8
                index_finger_landmarks = [hand_landmarks.landmark[i] for i in range(5, 9)]
                for i, landmark in enumerate(index_finger_landmarks):
                    osc_client.send_message(f"/hand/index_finger/{i}", [landmark.x, landmark.y, landmark.z])

                # Optionally, you can still draw all the landmarks for the right hand
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame with hand tracking
    cv2.imshow('Hand Tracking', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

