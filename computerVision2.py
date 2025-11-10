import cv2 as cv
import mediapipe as mp
import time

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

def getPlayerMove(gestures, gameStyle=1):


    """
    Goal here is to return a dictionary to the project file indicating the seperate moves of each player.
    Player 1 is defined as the player on the bottom half of the screen.
    Player 2 is defined as the player on the top half of the screen.
    
    
    """

    if len(gestures)==2:
        gameStyle = 1
    elif len(gestures)==4:
        gameStyle = 2
    else:
        gameStyle = 3

    

    if gameStyle == 1:
        player1_move = {'player': "Player One",'move1': None}
        player2_move = {'player':"Player Two",'move1': None}
        for gesture in gestures:
            player_dict = player1_move if (gestures[gesture].landmark[0].y) > 240 else player2_move
            player_dict['move1'] = choice(gesture)

        return player1_move, player2_move



    elif gameStyle == 2:
        player1_move = {'player': "Player One",'move1': None,'move2': None} 
        player2_move = {'player': "Player Two",'move1': None,'move2': None} 
        
        for gesture in gestures:
            player_dict = player1_move if (gestures[gesture].landmark[0].y) > 240 else player2_move
            if player_dict['move1'] is None:
                player_dict['move1'] = choice(gesture)
            else:
                player_dict['move2'] = choice(gesture)

        return player1_move, player2_move

    else:
        return "Unknown Style"

def choice(hand_landmarks):
    hand = None

    landmarks = hand_landmarks.landmark
    #thumbTip = landmarks[4]
    indexTip = landmarks[8]
    middleTip = landmarks[12]
    ringTip = landmarks[16]
    pinkyTip = landmarks[20]
    #thumbknuckle = landmarks[2]
    indexknuckle = landmarks[5]
    middleknuckle = landmarks[9]
    ringknuckle = landmarks[13]
    pinkyknuckle = landmarks[17]
    wrist = landmarks[0]
    
    if wrist.y > 240:
        if wrist.x < landmarks[1].x:
            hand = 0    # Left hand
        else:
            hand = 1    # Right hand

        if (indexTip.y < indexknuckle.y and middleTip.y < middleknuckle.y and
            ringTip.y < ringknuckle.y and pinkyTip.y < pinkyknuckle.y):
            return "Paper"
        elif (indexTip.y < indexknuckle.y and middleTip.y < middleknuckle.y and
              ringTip.y > ringknuckle.y and pinkyTip.y > pinkyknuckle.y):
            return "Scissors"
        elif (indexTip.y > indexknuckle.y and middleTip.y > middleknuckle.y and
              ringTip.y > ringknuckle.y and pinkyTip.y > pinkyknuckle.y):
            return "Rock"
        else:
            return "Unknown"
    else:
        if (indexTip.y > indexknuckle.y and middleTip.y > middleknuckle.y and
            ringTip.y > ringknuckle.y and pinkyTip.y > pinkyknuckle.y):
            return "Paper"
        elif (indexTip.y > indexknuckle.y and middleTip.y > middleknuckle.y and
              ringTip.y < ringknuckle.y and pinkyTip.y < pinkyknuckle.y):
            return "Scissors"
        elif (indexTip.y < indexknuckle.y and middleTip.y < middleknuckle.y and
              ringTip.y < ringknuckle.y and pinkyTip.y < pinkyknuckle.y):
            return "Rock"
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


        gestures = results.multi_hand_landmarks
       

        if 0 <= clock < 20:
            gametext = "Get Ready!"
            success = True
        elif clock <30: gametext = "3"
        elif clock <50: gametext = "2"
        elif clock <70: gametext = "1"
        elif clock <90:
            p1_move, p2_move = getPlayerMove(gestures)
            print(p1_move)
            print(p2_move)


        # Display clock and game text on video feed
        cv.putText(frame, f"Clock: {clock}", (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv.LINE_AA) 
        cv.putText(frame, gametext, (50,80), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv.LINE_AA)

        clock = (clock + 1) % 100

        cv.imshow('frame', frame)
        
        
        # Close the program by pressing 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break



vid.release()
cv.destroyAllWindows()