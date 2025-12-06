import cv2
import time
import numpy as np
from ultralytics import YOLO
from RPSLogic import RPSMinus1


# ============================================================
#  MODEL LOADING
# ============================================================
modelPath = "rps_yolo11/weights/best.pt"
model = YOLO(modelPath)

cv2Camera = 0  # Windows laptops = 0, Mac external camera often = 1
gesture_map = {0: "P", 1: "R", 2: "S"}



# ============================================================
#  Text Drawing Function - uses text size scaled to camera resolution
# ============================================================

def draw_centered_text(img, text, y_ratio, scale_ratio=0.018, thickness_ratio=0.0015):
    h, w = img.shape[:2]

    # Compute scaled font size 
    scale = h * scale_ratio
    scale = max(0.3, min(scale, 1.2))  

    # Compute scaled thickness
    thickness = max(1, int(h * thickness_ratio))

    # Measure text for centering
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)

    x = int((w - tw) / 2)
    y = int(h * y_ratio)

    # Outline
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness + 2, cv2.LINE_AA)

    # Foreground text
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness, cv2.LINE_AA)



# ============================================================
#  INPUT MODE SELECTION
# ============================================================
print("\n--- RPS 4-Hand Classifier ---")
print("1. Use camera")
print("2. Use images")
choice = input("Enter choice (1 or 2): ").strip()

use_images = choices_loaded = False
image_phase0 = image_phase2 = None

if choice == "2":
    use_images = True

    # Load PHASE 0 Image (4 hands)
    img0_path = input("\nEnter PHASE 0 image path (4 hands): ")
    image_phase0 = cv2.imread(img0_path)
    if image_phase0 is None:
        raise ValueError("ERROR: Could not load Phase 0 image.")

    # Load PHASE 2 Image (2 hands)
    img2_path = input("Enter PHASE 2 image path (2 hands after drop): ")
    image_phase2 = cv2.imread(img2_path)
    if image_phase2 is None:
        raise ValueError("ERROR: Could not load Phase 2 image.")

    choices_loaded = True
    print("Images loaded successfully.\n")

else:
    # Find the camera and its resolution -- Resolution used for text scaling and placement
    print("\nUsing CAMERA input...\n")
    
    test_cap = cv2.VideoCapture(cv2Camera)
    CAM_W = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    CAM_H = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    test_cap.release()
    print(f"Camera Resolution Detected: {CAM_W} x {CAM_H}")

    cap = cv2.VideoCapture(cv2Camera)
    
# ============================================================
#  STRATEGY SELECTION
# ============================================================
print("Select game strategy (1:Agressive, 2:Defensive, 3:Balanced, 4:Default):")
strategy_input = input("Strategy: ").strip()

current_strategy = int(strategy_input) if strategy_input in ["1", "2", "3", "4"] else 4
strategy_names = {1: "Aggressive", 2: "Defensive", 3: "Balanced", 4: "Default"}
print(f"\nUsing Strategy: {strategy_names[current_strategy]}\n")


# ============================================================
#  GAME STATE VARIABLES
# ============================================================
startTime = time.time()

p1_final = p2_final = None
p1_box = p2_box = None
p1_choices = []
p2_choices = []
waiting_for_phase2_image = False


# ============================================================
#  MAIN LOOP
# ============================================================
while True:

    now = time.time()
    elapsed = now - startTime

    # --------------------------------------------------------
    # Acquire a Frame (Camera or Image Mode)
    # --------------------------------------------------------
    if use_images:
        if elapsed < 5:
            frame = image_phase0.copy()
        else:
            frame = image_phase2.copy()
    else:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    # ========================================================
    #  PHASE 0 — Detect 4 Hands
    # ========================================================
    if elapsed < 10:

        p1_choices = []
        p2_choices = []

        # Split into quadrants
        q1 = frame[0:h//2, 0:w//2]
        q2 = frame[0:h//2, w//2:w]
        q3 = frame[h//2:h, 0:w//2]
        q4 = frame[h//2:h, w//2:w]
        quadrants = [q1, q2, q3, q4]

        # Detect in each quadrant
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

                    # Assign player based on quadrant
                    if idx in [0, 2]:
                        p1_choices.append(gesture)
                    else:
                        p2_choices.append(gesture)

                    cv2.rectangle(q, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(q, gesture, (x1+5, y1+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        final_view = np.vstack((np.hstack((q1, q2)),
                                np.hstack((q3, q4))))

        draw_centered_text(final_view,
                           f"SHOW BOTH HANDS ({10 - int(elapsed)})",
                           y_ratio=0.08)

        cv2.imshow("4-Hand Classifier", final_view)


    # ========================================================
    #  PHASE 1 — Player 1 Must Drop a Hand
    # ========================================================
    elif elapsed < 15:

        if not waiting_for_phase2_image:
            logic = RPSMinus1(p1_choices, p2_choices, strategy=current_strategy)
            print("\nPlayer 1 should DROP:", logic["recommended_drop"])
            waiting_for_phase2_image = True

        drop_text = f"PLAYER 1 DROP: {logic['recommended_drop']}"

        draw_centered_text(frame, "REMOVE ONE HAND!", y_ratio=0.22, scale_ratio=0.07)
        draw_centered_text(frame, drop_text, y_ratio=0.32, scale_ratio=0.06)

        cv2.imshow("4-Hand Classifier", frame)


    # ========================================================
    #  PHASE 2 — Final One-Hand Detection
    # ========================================================
    elif elapsed < 25:

        left_half  = frame[:, :w//2]
        right_half = frame[:, w//2:]

        # ---------------------------
        # Detect Player 1 (Left Side)
        # ---------------------------
        best_conf = 0
        p1_box = None
        r1 = model(left_half, verbose=False)

        for r in r1:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > 0.35 and conf > best_conf:
                    best_conf = conf
                    p1_final = gesture_map[int(box.cls[0])]
                    p1_box = box.xyxy[0].cpu().numpy().astype(int)

        # ---------------------------
        # Detect Player 2 (Right Side)
        # ---------------------------
        best_conf = 0
        p2_box = None
        r2 = model(right_half, verbose=False)

        for r in r2:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > 0.35 and conf > best_conf:
                    best_conf = conf
                    p2_final = gesture_map[int(box.cls[0])]
                    p2_box = box.xyxy[0].cpu().numpy().astype(int)

                    # Adjust for right side offset
                    p2_box[0] += w//2
                    p2_box[2] += w//2

        # Draw P1/P2 boxes
        if p1_box is not None:
            x1,y1,x2,y2 = p1_box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(frame, p1_final, (x1+5, y1+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        if p2_box is not None:
            x1,y1,x2,y2 = p2_box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(frame, p2_final, (x1+5, y1+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        draw_centered_text(frame,
                           f"FINAL DETECTION ({25 - int(elapsed)})",
                           y_ratio=0.08)

        cv2.imshow("4-Hand Classifier", frame)


    # ========================================================
    #  PHASE 3 — Final Winner
    # ========================================================
    else:

        if p1_final and p2_final:
            result = RPSMinus1([p1_final], [p2_final], strategy=current_strategy)

            if result.get("status") == "win":
                message = "PLAYER 1 WINS!"
            elif len(result["winning"]) > 0:
                message = "PLAYER 1 WINS!"
            elif len(result["losing"]) > 0:
                message = "PLAYER 2 WINS!"
            else:
                message = "DRAW!"
        else:
            message = "ERROR: COULD NOT DETECT BOTH HANDS"

        draw_centered_text(frame, message, y_ratio=0.30, scale_ratio=0.08)
        draw_centered_text(frame, "Press 'R' to Reset",
                           y_ratio=0.50, scale_ratio=0.05)

        cv2.imshow("4-Hand Classifier", frame)


    # ========================================================
    #  KEYBOARD CONTROLS
    # ========================================================
    key = cv2.waitKey(1) & 0xFF

    # Quit program
    if key == ord('q'):
        break

    # Reset game
    if key == ord('r'):
        print("Game reset.\n")
        startTime = time.time()
        p1_final = p2_final = None
        p1_box = p2_box = None
        continue


# Cleanup
if not use_images:
    cap.release()

cv2.destroyAllWindows()
