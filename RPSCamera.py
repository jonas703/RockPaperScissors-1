import cv2
import time
import numpy as np
from ultralytics import YOLO
from RPSLogic import RPSMinus1

# ----------------------------------------------------
# Model
# ----------------------------------------------------
modelPath = "rps_yolo11/weights/best.pt"
model = YOLO(modelPath)

cv2Camera = 1 # 0 for all windows computers, set to 1 for mac

gesture_map = {0: "P", 1: "R", 2: "S"}

# ----------------------------------------------------
# Runtime Mode Flags
# ----------------------------------------------------
use_images = False
use_image_phase0 = False
use_image_phase2 = False

image_phase0 = None
image_phase2 = None

waiting_for_phase2_image = False


# ----------------------------------------------------
# Menu Before Starting Game
# ----------------------------------------------------
print("\n--- RPS 4-Hand Classifier ---")
print("Select input mode:")
print("1. Use camera")
print("2. Use images")

choice = input("Enter choice (1 or 2): ").strip()

if choice == "2":
    use_images = True
    use_image_phase0 = True

    # Load PHASE 0 image
    img0_path = input("\nEnter PHASE 0 image path (4 hands): ")
    image_phase0 = cv2.imread(img0_path)
    if image_phase0 is None:
        raise ValueError("ERROR: Could not load Phase 0 image.")

    # Load PHASE 2 image
    img2_path = input("Enter PHASE 2 image path (2 hands after drop): ")
    image_phase2 = cv2.imread(img2_path)
    if image_phase2 is None:
        raise ValueError("ERROR: Could not load Phase 2 image.")

    use_image_phase2 = True
    print("\nImages loaded successfully.")

else:
    print("\nUsing CAMERA input...")
    cap = cv2.VideoCapture(cv2Camera)


# ----------------------------------------------------
# Strategy Selection
# ----------------------------------------------------
print("\nSelect game strategy (1:Agressive, 2:Defensive, 3:Balanced, 4:Default):")
strategy_input = input("Strategy: ").strip()
if strategy_input not in ["1", "2", "3", "4"]:
    print("Invalid input — defaulting to strategy 4.")
    current_strategy = 4
else:
    current_strategy = int(strategy_input)

match current_strategy:
    case 1:
        strategy = "Aggressive"
    case 2:
        strategy = "Defensive"
    case 3:
        strategy = "Balanced"
    case 4:
        strategy = "Default"


print(f"\nUsing Strategy {strategy}\n")


# ----------------------------------------------------
# Helper for Outlined Text
# ----------------------------------------------------
def outlined_text(img, text, pos, scale=1.5, thickness=3):
    cv2.putText(img, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness, cv2.LINE_AA)


# ----------------------------------------------------
# Initialize Game State
# ----------------------------------------------------
startTime = time.time()

p1_final = None
p2_final = None

p1_box = None
p2_box = None

p1_choices = []
p2_choices = []


# ----------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------
while True:

    now = time.time()
    elapsed = now - startTime

    # ------------------------------------------
    # Frame Acquisition Depending on Mode
    # ------------------------------------------
    if use_images:
        if elapsed < 5:
            frame = image_phase0.copy()
        elif 7 <= elapsed < 12:
            frame = image_phase2.copy()
        else:
            frame = image_phase2.copy()
        h, w = frame.shape[:2]

    else:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

    # ---------------------------------------------------------
    # PHASE 0: Detect 4 hands (first 5 seconds)
    # ---------------------------------------------------------
    if elapsed < 10:

        p1_choices = []
        p2_choices = []

        # split into quadrants
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

                    # classify P1 or P2 based on quadrant
                    if idx in [0, 2]:
                        p1_choices.append(gesture)
                    else:
                        p2_choices.append(gesture)

                    cv2.rectangle(q, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(q, label, (x1+5, y1+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0,0,255), 2)

        final_view = np.vstack((np.hstack((q1, q2)),
                                np.hstack((q3, q4))))

        outlined_text(final_view,
                      f"SHOW BOTH HANDS ({10 - int(elapsed)})",
                      (850, 40), 1.5, 3)

        cv2.imshow("4-Hand Classifier", final_view)

    # ---------------------------------------------------------
    # PHASE 1: Player 1 must drop a hand (5–7 sec)
    # ---------------------------------------------------------
    elif elapsed < 15:

        drop_text = "Analyzing..."

        if not waiting_for_phase2_image:
            logic = RPSMinus1(p1_choices, p2_choices, strategy=current_strategy)
            print(f"\nPlayer 1 should DROP: {logic['recommended_drop']}")
            waiting_for_phase2_image = True

        if len(p1_choices) > 0 and len(p2_choices) > 0:
            logic = RPSMinus1(p1_choices, p2_choices, strategy=current_strategy)
            drop_text = f"PLAYER 1 DROP: {logic['recommended_drop']}"

        outlined_text(frame, "REMOVE ONE HAND!", (350, 180), 2, 4)
        outlined_text(frame, drop_text, (350, 260), 1.8, 4)

        cv2.imshow("4-Hand Classifier", frame)

    # ---------------------------------------------------------
    # PHASE 2: Final single-hand detection (7–12 sec)
    # ---------------------------------------------------------
    elif elapsed < 25:

        left_half  = frame[:, :w//2]
        right_half = frame[:, w//2:]

        # ------------------
        # Player 1 detection
        # ------------------
        best_conf = 0
        best_cls  = None
        best_box  = None

        r1 = model(left_half, verbose=False)
        for r in r1:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > 0.35 and conf > best_conf:
                    best_conf = conf
                    best_cls  = int(box.cls[0])
                    best_box  = box.xyxy[0].cpu().numpy().astype(int)

        if best_box is not None:
            p1_final = gesture_map[best_cls]
            p1_box   = tuple(best_box)

        # ------------------
        # Player 2 detection
        # ------------------
        best_conf = 0
        best_cls  = None
        best_box  = None

        r2 = model(right_half, verbose=False)
        for r in r2:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > 0.35 and conf > best_conf:
                    best_conf = conf
                    best_cls  = int(box.cls[0])
                    best_box  = box.xyxy[0].cpu().numpy().astype(int)

        if best_box is not None:
            best_box[0] += w//2
            best_box[2] += w//2
            p2_final = gesture_map[best_cls]
            p2_box   = tuple(best_box)

        # Draw detected hands
        if p1_box:
            x1,y1,x2,y2 = p1_box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(frame, p1_final, (x1+5,y1+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        if p2_box:
            x1,y1,x2,y2 = p2_box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(frame, p2_final, (x1+5,y1+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        outlined_text(frame,
                      f"FINAL DETECTION ({25 - int(elapsed)})",
                      (30, 50), 1.5, 3)

        cv2.imshow("4-Hand Classifier", frame)

    # ---------------------------------------------------------
    # PHASE 3: Final result
    # ---------------------------------------------------------
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
        outlined_text(frame, "Press 'R' to Reset", (200, 400), 2, 4)
        cv2.imshow("4-Hand Classifier", frame)

    # ---------------------------------------------------------
    # Keyboard Controls
    # ---------------------------------------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('r'):
        print("Game reset.")
        startTime = time.time()
        p1_final = p2_final = None
        p1_box = p2_box = None
        continue


# Cleanup
if not use_images:
    cap.release()
cv2.destroyAllWindows()
