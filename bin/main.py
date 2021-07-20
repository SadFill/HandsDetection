"""
https://www.youtube.com/watch?v=01sAkU_NvOY
"""

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  # номер вебки, по умолчанию 0

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results:
        for handlandmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handlandmarks, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
