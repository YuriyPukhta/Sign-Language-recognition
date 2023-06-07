import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import torch
from model import SimpleCNN, load_model, min_max_scale
import numpy as np

model = SimpleCNN(26)
model = load_model(model)
model.eval()
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
mapper = {0: 'a',
          1: 'b',
          2: 'c',
          3: 'd',
          4: 'e',
          5: 'f',
          6: 'g',
          7: 'h',
          8: 'i',
          9: 'j',
          10: 'k',
          11: 'l',
          12: 'm',
          13: 'n',
          14: 'o',
          15: 'p',
          16: 'q',
          17: 'r',
          18: 's',
          19: 't',
          20: 'u',
          21: 'v',
          22: 'w',
          23: 'x',
          24: 'y',
          25: 'z'}

cap = cv2.VideoCapture(0)
_, frame = cap.read()
h, w, c = frame.shape

while True:
    _, frame = cap.read()

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            new_hande = []
            # print(handLMs)
            X_arr = []
            Y_arr = []
            Z_arr = []
            for hl in handLMs.landmark:
                X_arr.append(hl.x)
                Y_arr.append(hl.y)
                Z_arr.append(hl.z)
                # new_hande.append(hl.x)
                # new_hande.append(hl.y)
                # new_hande.append(hl.z)
            X_arr = min_max_scale(X_arr)
            Y_arr = min_max_scale(Y_arr)
            Z_arr = min_max_scale(Z_arr)

            for i in range(len(X_arr)):
                new_hande.append(X_arr[i])
                new_hande.append(Y_arr[i])
                new_hande.append(Z_arr[i])

            new_hande = np.array(new_hande)
            sign_data = torch.Tensor(new_hande.reshape(1, 21, 3).astype(float))
            sign_data = torch.unsqueeze(sign_data, 0)
            output = model(sign_data)
            _, preds = torch.max(output.data, 1)
            print(mapper[preds.item()])
            cv2.imshow("Frame", frame)

cap.release()
cv2.destroyAllWindows()
