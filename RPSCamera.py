import cv2
import numpy as np
from ultralytics import YOLO
from RPSLogic import RPSMinus1  # ← IMPORT THE LOGIC ENGINE

modelPath = "runs/detect/rps_yolo11/weights/best.pt"
model = YOLO(modelPath)

cap = cv2.VideoCapture(1)

# Map YOLO class IDs to R/P/S
gesture_map = {
    0: "P",   # Paper
    1: "R",   # Rock
    2: "S"    # Scissors
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame = cv2.flip(frame, 1) # Mirror the frame for natural interaction

    # Divide frame into 4 quadrants
    q1 = frame[0:h//2, 0:w//2]
    q2 = frame[0:h//2, w//2:w]
    q3 = frame[h//2:h, 0:w//2]
    q4 = frame[h//2:h, w//2:w]

    quadrants = [q1, q2, q3, q4]
    p1_choices = []
    p2_choices = []

    # Process each quadrant
    for idx, q in enumerate(quadrants):
        results = model(q, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                gesture = gesture_map.get(cls, "?")

                # LEFT side (quadrants 1 & 3) = Player 1
                if idx in [0, 2]:
                    p1_choices.append(gesture)
                # RIGHT side (quadrants 2 & 4) = Player 2
                else:
                    p2_choices.append(gesture)

                label = f"{gesture} {conf:.2f}"

                cv2.rectangle(q, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(q, label, (x1 + 5, y1 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Combine quadrants for output display
    top = np.hstack((quadrants[0], quadrants[1]))
    bottom = np.hstack((quadrants[2], quadrants[3]))
    final_view = np.vstack((top, bottom))

    # -----------------------------
    # Apply RPS Logic if valid data
    # -----------------------------
    if len(p1_choices) > 0 and len(p2_choices) > 0:
        logic = RPSMinus1(p1_choices, p2_choices, strategy=3)

        if logic["status"] == "win":
            text = "ALL ROADS LEAD TO VICTORY"
        else:
            text = f"DROP: {logic['recommended_drop']}"

        # Overlay text

        cv2.putText(final_view, text, (850, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 3)
    
    cv2.putText(final_view,"Player 1", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(final_view,"Player 2", (1700, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.imshow("4-Hand Classifier (YOLO + RPS Logic)", final_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
