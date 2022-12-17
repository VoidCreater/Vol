import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
import pyautogui as pyg

keyboard = Controller()
cap = cv2.VideoCapture(0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence = 0.7, min_tracking_confidence = 0.5)
TipId = [4, 8, 12, 16, 20]

def drawHLands(image, hand_landmarks):
    if(hand_landmarks):
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

def countFingers(image, hand_landmarks, handNo = 0):
    if(hand_landmarks):
        lands = hand_landmarks[handNo].landmark
        fingers = []
        for lm_index in TipId:
            finger_tip_y = lands[lm_index].y
            finger_bottom_y = lands[lm_index -2].y
            if(lm_index!=4):
                if(finger_tip_y < finger_bottom_y):
                    fingers.append(1)
                if(finger_tip_y > finger_bottom_y):
                    fingers.append(0)
        totalfingers = fingers.count(1)
        if(totalfingers == 1):
            if(finger_tip_y < h-250):
                print("vol increased")
                pyg.press("volumeup")
            if(finger_tip_y > h-250):
                print("vol decreased")
                pyg.press("volumedown")

while(True):
    success, image = cap.read()
    image = cv2.flip(image, 1)
    result = hands.process(image)
    hand_landmarks = result.multi_hand_landmarks

    drawHLands(image, hand_landmarks)

    countFingers(image, hand_landmarks)
    cv2.imshow("mediapipe", image)

    key = cv2.waitKey(1)
    if(key == 27):
        break

cv2.destroyAllWindows()
