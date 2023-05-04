import cv2
import numpy as np
import os
import tensorflow as tf
from keras import models

np.set_printoptions(threshold = np.inf)
np.set_printoptions(suppress=True)

mode = input("Suit or Value:")
folder = input("Card Type ex(clubs):")
card = input("Card Type ex(3H):")
numImg = 0

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

white_lower = np.array([0, 0, 200], dtype=np.uint8)
white_upper = np.array([255, 30, 255], dtype=np.uint8)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    edges = cv2.Canny(frame, 100, 200)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True)*0.02, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 50:
                print("--------Found card--------")
                numImg = numImg + 1
                if numImg > 15:
                    os._exit(0)
                card_image = frame[y:y + h, x:x + w]
                card_image_resized = cv2.resize(card_image, (200, 200))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                if mode == 'suit':
                    cv2.imwrite("suitDataset/"+str(folder)+"/"+str(card)+str(numImg)+".jpg", card_image)
                else:
                    cv2.imwrite("valueDataset/"+str(folder)+"/"+str(card)+str(numImg)+".jpg", card_image)


    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()