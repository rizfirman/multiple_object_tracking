import torch
from pythonosc import udp_client
import cv2
import ssl
import numpy as np

# Disable SSL certificate verification (temporary fix)
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize OSC client to send data to TouchDesigner
osc_client = udp_client.SimpleUDPClient("127.0.0.1", 8000)  # IP localhost and port TouchDesigner listens to

# Load pre-trained YOLOv5 model (yolov5s.pt)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# Set camera input (0 for webcam)
url = 0
cap = cv2.VideoCapture(url)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Initialize list of OpenCV trackers
trackers = cv2.legacy.MultiTracker_create()

detection_refresh_interval = 10  # Refresh detection every 10 frames
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video source.")
        break

    frame_count += 1

    # Refresh detection every 'detection_refresh_interval' frames
    if frame_count % detection_refresh_interval == 0 or len(trackers.getObjects()) == 0:
        # Clear existing trackers and reinitialize
        trackers = cv2.legacy.MultiTracker_create()

        # YOLOv5 Detection
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()

        for det in detections:
            if len(det) == 6:  # Ensure that detection has all necessary components (6 values: x1, y1, x2, y2, conf, cls)
                x1, y1, x2, y2, conf, cls = det
                if int(cls) == 0:  # Only track 'person' class (class 0 in COCO)
                    bbox = (x1, y1, x2 - x1, y2 - y1)  # Convert to [x, y, w, h]
                    tracker = cv2.legacy.TrackerCSRT_create()  # Create a new tracker for each detected person
                    trackers.add(tracker, frame, bbox)

    # Update trackers
    success, boxes = trackers.update(frame)

    for i, new_box in enumerate(boxes):
        x, y, w, h = new_box
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {i}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Normalizing the coordinates (0 to 1 range)
        norm_x = (x1 + x2) / 2 / frame.shape[1]
        norm_y = (y1 + y2) / 2 / frame.shape[0]

        # Send OSC messages to TouchDesigner
        osc_client.send_message(f"/person/{i}/position", [norm_x, norm_y])
        osc_client.send_message(f"/person/{i}/bbox", [x1, y1, x2, y2])

    # Display the frame
    cv2.imshow("Webcam Feed", frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
