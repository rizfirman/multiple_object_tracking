from ultralytics import YOLO
from pythonosc import udp_client
import cv2
import ssl
import numpy as np
from sort import Sort

# Disable SSL certificate verification (temporary fix)
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize OSC client to send data to TouchDesigner
osc_client = udp_client.SimpleUDPClient("127.0.0.1", 8000)  # IP localhost and port TouchDesigner listens to

# Load pre-trained YOLOv8 model (yolov8n.pt)
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' for a lightweight model, or 'yolov8s.pt' for a standard model

# Set camera input (0 for webcam or URL for a video stream)
url = 0
cap = cv2.VideoCapture(url)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Initialize SORT tracker
tracker = Sort()

# Counting variables
count = 0
counted_ids = set()  # Set to keep track of counted object IDs

# Line position (y-coordinate)
line_position = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2 )

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video source.")
        break

    # YOLOv8 Detection
    results = model(frame, conf=0.25)  # Perform inference with confidence threshold
    detections = results[0].boxes.data.cpu().numpy()  # Extract detected bounding boxes

    # Prepare the detections for SORT: [x1, y1, x2, y2, score]
    sort_input = []
    for det in detections:
        if len(det) == 6:  # Ensure detection has 6 components: x1, y1, x2, y2, conf, cls
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0:  # Only track 'person' class (class 0 in COCO)
                # Append to sort_input with the expected format
                sort_input.append([x1, y1, x2, y2, conf])

    # Convert to numpy array and check if sort_input is not empty
    sort_input = np.array(sort_input)

    # Check if sort_input has valid detections
    if sort_input.size > 0:
        # Update tracker
        tracked_objects = tracker.update(sort_input)

        # Process the tracked objects
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj  # Extract bounding box and ID
            center_y = int((y1 + y2) / 2)

            # Draw the bounding box and ID
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {int(obj_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Check if the person crosses the line and has not been counted yet
            if center_y > line_position - 10 and center_y < line_position + 10:
                if int(obj_id) not in counted_ids:
                    counted_ids.add(int(obj_id))
                    count += 1
                    print(f"Person count: {count}")

                    # Send OSC messages to TouchDesigner
                    osc_client.send_message("/person_count", count)

    # Draw the imaginary line
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (255, 0, 0), 2)

    # Display the count on the frame
    cv2.putText(frame, f'Count: {count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Webcam Feed", frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
