import cv2
import numpy as np
import tensorflow as tf
from keras import models

suitArray = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
valueArray = ['10', '2', '3', '4', '5', '6', '7', '8', '9', 'Ace', 'J', 'JOKER', 'King', 'Queen']
bestSuit = ''
bestValue = ''

np.set_printoptions(threshold = np.inf)
np.set_printoptions(suppress=True)

model_value = models.load_model('cardModelValue.h5')
model_value.load_weights(tf.train.latest_checkpoint("checkpoints_value"))

model_suit = models.load_model('cardModelSuit.h5')
model_suit.load_weights(tf.train.latest_checkpoint("checkpoints_suit"))

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
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                img = cv2.resize(frame[y:y+h, x:x+w], (200, 200))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0)
                networkOutput_value = model_value.predict(img_array)
                networkOutput_suit = model_suit.predict(img_array)
                value_index = np.argmax(networkOutput_value)
                print(networkOutput_value)
                suit_index = np.argmax(networkOutput_suit)
                print(networkOutput_suit)
                bestValue = valueArray[value_index]
                bestSuit = suitArray[suit_index]

    if bestValue == 'JOKER':
        frame = cv2.putText(frame, bestValue, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    elif bestValue != '':
        frame = cv2.putText(frame, bestValue+' of '+bestSuit, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()