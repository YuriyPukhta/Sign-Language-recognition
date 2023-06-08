import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import mediapipe as mp
from .model import Resnet50_Fine, load_model, test_transform
import torch
from PIL import Image

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


class ResNetDetector:
    def __init__(self):
        model = Resnet50_Fine(num_classes=26)
        self.model = load_model(model)
        self.model.eval()
        self.hands = mp.solutions.hands.Hands()

    def get_letter(self, frame):
        h, w, c = frame.shape
        result = self.hands.process(frame)
        hand_landmarks = result.multi_hand_landmarks
        letter = " "
        if hand_landmarks:
            for handLMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                X = 0
                Y = 0
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    X += x
                    Y += y
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)

                X = int(X / len(handLMs.landmark))
                Y = int(Y / len(handLMs.landmark))

                l = max(abs(y_max - y_min), abs(x_max - x_min))
                x_min = X - round(l / 2)
                x_max = x_min + l
                y_min = Y - round(l / 2)
                y_max = y_min + l
                y_min = y_min - 10
                y_max = y_max + 20
                x_min = x_min - 10
                x_max = x_max + 20

                if x_max not in range(0, w):
                    letter = None
                    continue
                if x_min not in range(0, w):
                    letter = None
                    continue
                if y_max not in range(0, h):
                    letter = None
                    continue
                if y_min not in range(0, h):
                    letter = None
                    continue

                image = Image.fromarray(frame[y_min:y_max, x_min: x_max, :])
                transformed_image = test_transform(image)
                transformed_image = torch.unsqueeze(transformed_image, 0)
                output = self.model(transformed_image)
                letter = mapper[torch.max(output.data, 1)[1].item()]
                break

        return letter
