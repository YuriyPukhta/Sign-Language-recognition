import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import mediapipe as mp
import torch
from .model import SimpleCNN, load_model, min_max_scale
import numpy as np

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


class ASLDetector:
    def __init__(self):
        model = SimpleCNN(26)
        self.model = load_model(model)
        self.model.eval()
        self.hands = mp.solutions.hands.Hands()

    def get_letter(self, frame):
        result = self.hands.process(frame)
        hand_landmarks = result.multi_hand_landmarks
        letter = " "
        if hand_landmarks:
            for handLMs in hand_landmarks:
                new_hande = []
                X_arr = []
                Y_arr = []
                Z_arr = []
                for hl in handLMs.landmark:
                    X_arr.append(hl.x)
                    Y_arr.append(hl.y)
                    Z_arr.append(hl.z)
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
                output = self.model(sign_data)
                _, preds = torch.max(output.data, 1)
                letter = mapper[preds.item()]
                break

        return letter
