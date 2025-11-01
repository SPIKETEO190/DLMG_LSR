import cv2 as cv
import time
import mediapipe as mp
import numpy as np

running = True

cap = cv.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpFaceMesh = mp.solutions.face_mesh
face = mpFaceMesh.FaceMesh()
mpDraw = mp.solutions.drawing_utils

current_time = 0
previous_time = 0

while running:
    _, image = cap.read()
    imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results_hands = hands.process(imgRGB)
    results_face = face.process(imgRGB)

    key = cv.waitKey(5) & 0xFF

    if results_hands.multi_hand_landmarks :
        for hands_landmarks in results_hands.multi_hand_landmarks:
            mpDraw.draw_landmarks(image, hands_landmarks, mpHands.HAND_CONNECTIONS)
            lm = hands_landmarks.landmark
            wrist = np.array([lm[0].x,lm[0].y,lm[0].z])
            index_base = np.array([lm[5].x,lm[5].y,lm[5].z])
            pinky_base = np.array([lm[17].x,lm[17].y,lm[17].z])
            v1 = index_base - wrist
            v2 = pinky_base - wrist

            normala = np.cross(v1,v2)

            



    if results_face.multi_face_landmarks :
        for face_landmarks in results_face.multi_face_landmarks:
            top_lip_y = face_landmarks.landmark[13].y
            bottom_lip_y = face_landmarks.landmark[14].y
            mouth_opening = abs(top_lip_y - bottom_lip_y)

            #print(mouth_opening)
            # if(mouth_opening > 0.01):
            #     print("open")
            # else:
            #     print("close")

    if key == ord("q"):
        running = False
        cap.release()
        cv.destroyAllWindows()
        print("Exiting...")
        break

    current_time = time.time()
    elapsed_time = current_time - previous_time
    fps = 1/elapsed_time
    previous_time = current_time

    cv.putText(image, str(int(fps)), (10,70), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv.imshow('DLMG_LSR', image)
    cv.waitKey(1)