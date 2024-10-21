import torch
from pythonosc import udp_client
import cv2
import ssl

# Disable SSL certificate verification (temporary fix)
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize OSC client to send data to TouchDesigner
osc_client = udp_client.SimpleUDPClient("127.0.0.1", 8000)  # IP localhost and port TouchDesigner listens to

# Load pre-trained YOLOv5 model (yolov5s.pt)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# Set camera input (0 for webcam)
source = 0  # 0 for webcam
cap = cv2.VideoCapture(source)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    # Inference on the frame
    results = model(frame)

    # Parse the detection results
    detected_persons = []
    for i, det in enumerate(results.xyxy[0]):  # results.xyxy[0] contains [x1, y1, x2, y2, confidence, class]
        if det[5] == 0:  # Class '0' corresponds to 'person' in COCO dataset
            x1, y1, x2, y2, conf, cls = det
            person_id = i  # Assign a unique ID for each person
            detected_persons.append((person_id, x1.item(), y1.item(), x2.item(), y2.item(), conf.item()))

            # Draw bounding box on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {person_id} Conf: {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Send detected person data to TouchDesigner via OSC
    for person_id, x1, y1, x2, y2, conf in detected_persons:
        # Normalizing the coordinates (0 to 1 range) to send via OSC
        norm_x = (x1 + x2) / 2 / frame.shape[1]  # x center
        norm_y = (y1 + y2) / 2 / frame.shape[0]  # y center
        osc_client.send_message(f"/person/{person_id}/position", [norm_x, norm_y])  # Send position to TouchDesigner
        osc_client.send_message(f"/person/{person_id}/bbox", [x1, y1, x2, y2])  # Send bounding box info
        osc_client.send_message(f"/person/{person_id}/confidence", conf)  # Send confidence score

    # Display the frame with bounding boxes
    cv2.imshow("Webcam Feed", frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
