import torch
import cv2
import numpy as np
from pathlib import Path

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # atau 'yolov5m', 'yolov5l', 'yolov5x' tergantung kebutuhan

# Inisialisasi webcam (index 0 untuk webcam default)
url = 0
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Tidak dapat mengakses webcam.")
    exit()

while True:
    # Membaca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak dapat membaca frame.")
        break

    # Mengubah format BGR (OpenCV) ke RGB (YOLOv5 membutuhkan format RGB)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Melakukan inferensi menggunakan model YOLOv5
    results = model(img_rgb)

    # Mengambil hasil dari deteksi
    detections = results.pandas().xyxy[0]  # Mendapatkan hasil dalam format pandas DataFrame

    # Iterasi setiap deteksi untuk menampilkan kotak deteksi di frame
    for idx, row in detections.iterrows():
        x1, y1, x2, y2, conf, cls = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], row['class']
        label = f"{results.names[int(cls)]} {conf:.2f}"

        # Gambar kotak bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Tulis label pada bounding box
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tampilkan frame dengan kotak deteksi
    cv2.imshow("YOLOv5 Real-Time Object Detection", frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan webcam dan tutup jendela
cap.release()
cv2.destroyAllWindows()
