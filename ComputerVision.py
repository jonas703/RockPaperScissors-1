import cv2 as cv
import mediapipe as mp

"""
An older version of python might need to be used to run mediapipe.
I got it to work with python 3.11.4

winget install --id Python.Python.3.11 --version 3.11.4 -e
py -3.11 -m pip install --upgrade pip setuptools wheel
py -3.11 -m pip install opencv-python
py -3.11 -m pip install mediapipe opencv-python
"""

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles


# Determine player's move based on hand landmarks. 
# mediapipe documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer

def getPlayerMove(hand_landmarks):
    landmarks = hand_landmarks.landmark
    # attempted implementation of 'middle finger to exit' gesture (not currently working)
    if (landmarks[12].y > landmarks[11].y and landmarks[16].y < landmarks[14].y and 
        landmarks[20].y < landmarks[18].y and landmarks[8].y < landmarks[6].y and
        (landmarks[5].x > landmarks[0].x and landmarks[0] > landmarks[17].x
         or landmarks[5].x < landmarks[0].x and landmarks[0] < landmarks[17].x)):
        return "Exit Game" # middle finger gesture to exit game
    
    if landmarks[0].x < landmarks[9].x:  # Left hand
        # Flip x-coordinates for left hand to standardize orientation
        for lm in landmarks:
            lm.x = -lm.x
    if all([landmarks[i+1].x < landmarks[i+3].x for i in range(9,20,4)]): 
        return "Rock" # uses coordinates of joints to determine if fingers are curled for rock
    elif (landmarks[14].y < landmarks[16].y and landmarks[18].y < landmarks[20].y) or (landmarks[14].x < landmarks[16].x and landmarks[18].x < landmarks[20].x):
        return "Scissors" # uses coordinates of joints to determine if ring and pinkie fingers are curled for scissors
    elif all([landmarks[i+2].x > landmarks[i+3].x for i in range(9,20,4)]):
        return "Paper" # check joints for if paper
    else: 
        return "Unknown"



# location of video feed may vary
vid = cv.VideoCapture(0)


#initialize game variables
clock = 0
p1_move = p2_move = None
gametext = "" 
success = True


# Mediapipe hands model setup
with mp_hands.Hands(model_complexity = 0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    
    #Game loop
    while True:
        ret, frame = vid.read()

        if not ret or frame is None: break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        results = hands.process(frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
        frame = cv.flip(frame, 1)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)


        """
        Currently the game recognizes two 'players' each with one move. 
        This will have to be modified to take into accout multiple rounds and the 'minus one' rule.
        player segmentation will also have to be implemented so player 1 and player 2 are correctly identified consistently.

        """

        gestures = results.multi_hand_landmarks

        if 0 <= clock < 20:
            gametext = "Get Ready!"
            success = True
        elif clock <30: gametext = "3"
        elif clock <50: gametext = "2"
        elif clock <70: gametext = "1"
        elif clock <90:
            gestures = results.multi_hand_landmarks
            if gestures and len(gestures) == 2:
                p1_move = getPlayerMove(gestures[0])
                p2_move = getPlayerMove(gestures[1])
                gametext = f"P1: {p1_move}  P2: {p2_move}"
            else:
                gametext = "Both players show your move!"
                success = False
        elif clock <130:
            if success and p1_move != "Unknown" and p2_move != "Unknown":
                gametext = f"P1: {p1_move}  P2: {p2_move}"
                if p1_move == p2_move:
                    gametext += "  It's a tie!"
                elif (p1_move == "Rock" and p2_move == "Scissors") or \
                     (p1_move == "Paper" and p2_move == "Rock") or \
                        (p1_move == "Scissors" and p2_move == "Paper"):
                    gametext += "  Player 1 wins!"
                else:   
                    gametext += "  Player 2 wins!"
            else:
                gametext = "Round failed! Try again."

        # Display clock and game text on video feed
        cv.putText(frame, f"Clock: {clock}", (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv.LINE_AA) 
        cv.putText(frame, gametext, (50,80), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv.LINE_AA)

        clock = (clock + 1) % 100

        cv.imshow('frame', frame)
        
        
        # Close the program by pressing 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if p1_move == "Exit Game" or p2_move == "Exit Game":
            break
        

vid.release()
cv.destroyAllWindows()