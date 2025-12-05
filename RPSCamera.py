import cv2
import time
import numpy as np
from ultralytics import YOLO
from RPSLogic import RPSMinus1

modelPath = "rps_yolo11/weights/best.pt"
model = YOLO(modelPath)
cap = cv2.VideoCapture(1)

gesture_map = {0: "P", 1: "R", 2: "S"}

# ----------------------------------------------
# Helper: Black text with white outline
# ----------------------------------------------
def outlined_text(img, text, pos, scale=1.5, thickness=3):
    cv2.putText(img, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness, cv2.LINE_AA)

# Start timer
startTime = time.time()

# Phase states
phase = 0
p1_final = None
p2_final = None

# Store bounding boxes so they persist
p1_box = None
p2_box = None

# For logic during phase 1
p1_choices = []
p2_choices = []

# Current game strategy
current_strategy = 3   # default
print("Current Strategy:", current_strategy)

while True:

    now = time.time()
    elapsed = now - startTime

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # ======================================================
    # PHASE 0 — First 5 seconds: detect FOUR hands
    # ======================================================
    if elapsed < 5:

        p1_choices = []
        p2_choices = []

        q1 = frame[0:h//2, 0:w//2]
        q2 = frame[0:h//2, w//2:w]
        q3 = frame[h//2:h, 0:w//2]
        q4 = frame[h//2:h, w//2:w]
        quadrants = [q1, q2, q3, q4]

        for idx, q in enumerate(quadrants):
            results = model(q, verbose=False)

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < 0.35:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls = int(box.cls[0])
                    gesture = gesture_map[cls]
                    label = f"{gesture} {conf:.2f}"

                    if idx in [0, 2]:
                        p1_choices.append(gesture)
                    else:
                        p2_choices.append(gesture)

                    cv2.rectangle(q, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(q, label, (x1+5, y1+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        final_view = np.vstack((
            np.hstack((q1, q2)),
            np.hstack((q3, q4))
        ))

        outlined_text(final_view,
                      f"SHOW BOTH HANDS ({5 - int(elapsed)})",
                      (850, 40), 1.5, 3)

        cv2.imshow("4-Hand Classifier", final_view)

    # ======================================================
    # PHASE 1 — 5 to 7 seconds: Ask Player 1 to drop a hand
    # ======================================================
    elif elapsed < 7:

        drop_text = "Analyzing..."
        if len(p1_choices) > 0 and len(p2_choices) > 0:
            logic = RPSMinus1(p1_choices, p2_choices, strategy=current_strategy)

            if logic["status"] == "win":
                drop_text = "ALL ROADS LEAD TO VICTORY"
            else:
                drop_text = f"PLAYER 1 DROP: {logic['recommended_drop']}"

        outlined_text(frame, "REMOVE ONE HAND!", (350, 180), 2, 4)
        outlined_text(frame, drop_text, (350, 260), 1.8, 4)

        cv2.imshow("4-Hand Classifier", frame)

    # ======================================================
    # PHASE 2 — 7 to 12 seconds: FINAL SINGLE HAND DETECTION
    # Continuously update best gesture for entire 5 seconds
    # ======================================================
    elif elapsed < 12:

        left_half = frame[:, :w//2]
        right_half = frame[:, w//2:]

        # Player 1 final hand (continuously choose highest-confidence detection)
        best_conf = 0
        best_cls = None
        best_box = None

        r1 = model(left_half, verbose=False)
        for r in r1:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > 0.35 and conf > best_conf:
                    best_conf = conf
                    best_cls = int(box.cls[0])
                    best_box = box.xyxy[0].cpu().numpy().astype(int)

        if best_box is not None:
            p1_final = gesture_map[best_cls]
            p1_box = tuple(best_box)

        # Player 2 final hand
        best_conf = 0
        best_cls = None
        best_box = None

        r2 = model(right_half, verbose=False)
        for r in r2:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > 0.35 and conf > best_conf:
                    best_conf = conf
                    best_cls = int(box.cls[0])
                    best_box = box.xyxy[0].cpu().numpy().astype(int)

        if best_box is not None:
            best_box[0] += w//2
            best_box[2] += w//2
            p2_final = gesture_map[best_cls]
            p2_box = tuple(best_box)

        # Draw live-updating boxes + labels
        if p1_final and p1_box is not None:
            x1, y1, x2, y2 = p1_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.putText(frame, p1_final, (x1+5, y1+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        if p2_final and p2_box is not None:
            x1, y1, x2, y2 = p2_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(frame, p2_final, (x1+5, y1+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        outlined_text(frame,
                      f"FINAL DETECTION ({12 - int(elapsed)})",
                      (30, 50), 1.5, 3)

        cv2.imshow("4-Hand Classifier", frame)

    # ======================================================
    # PHASE 3 — After 12 seconds: Show results
    # ======================================================
    else:

        if p1_final and p2_final:
            outcome = RPSMinus1([p1_final], [p2_final], strategy=current_strategy)

            if outcome.get("status") == "win":
                message = "PLAYER 1 WINS!"
            elif len(outcome["winning"]) > 0:
                message = "PLAYER 1 WINS!"
            elif len(outcome["losing"]) > 0:
                message = "PLAYER 2 WINS!"
            else:
                message = "DRAW!"

        else:
            message = "COULD NOT DETECT BOTH HANDS"

        outlined_text(frame, message, (200, 200), 2, 4)
        cv2.imshow("4-Hand Classifier", frame)

    # ======================================================
    # KEYBOARD CONTROLS — Reset & Strategy Selection
    # ======================================================
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # Reset game (keep same strategy)
    if key == ord('r'):
        startTime = time.time()
        p1_final = p2_final = None
        p1_box = p2_box = None
        print("Game reset (same strategy:", current_strategy, ")")
        continue

    # Reset & apply new strategy
    if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
        current_strategy = int(chr(key))
        startTime = time.time()
        p1_final = p2_final = None
        p1_box = p2_box = None
        print("Game reset with new strategy:", current_strategy)
        continue

cap.release()
cv2.destroyAllWindows()
