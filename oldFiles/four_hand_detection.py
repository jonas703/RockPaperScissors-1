import cv2
import numpy as np
from ultralytics import YOLO


modelPath = "runs/detect/rps_yolo11/weights/best.pt" # Path to your trained model weights,  "runs/detect/rps_yolo11/weights/best.pt"

model = YOLO(modelPath)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Divide frame into 4 quadrants to detect up to 4 hands
    q1 = frame[0:h//2, 0:w//2]
    q2 = frame[0:h//2, w//2:w]
    q3 = frame[h//2:h, 0:w//2]
    q4 = frame[h//2:h, w//2:w]

    quadrants = [q1, q2, q3, q4]
    outputs = []

    # Process each quadrant
    for q in quadrants:
        results = model(q, verbose=False)
        # Draw bounding boxes and labels
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{model.names[cls]} {conf:.2f}"

                cv2.rectangle(q, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(q, label, (x1, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        outputs.append(q)

    # Combine quadrants back into a single frame
    top = np.hstack((outputs[0], outputs[1]))
    bottom = np.hstack((outputs[2], outputs[3]))
    final_view = np.vstack((top, bottom))

    cv2.imshow("4-Hand Classifier (YOLO)", final_view)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
