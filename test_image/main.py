import os

import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp
import numpy as np
import time
from model import Resnet50_Fine, load_model, test_transform
import torchvision.transforms as transforms
from PIL import Image

#model = Resnet50WithFPN(26)
#model = load_model(model)

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision

import os
from PIL import Image
from tqdm import tqdm
import numpy as np



model = Resnet50_Fine(num_classes=26)

model = load_model(model)


opt_myCNN = optim.Adam(model.parameters(), lr = 0.0001)
criterion = nn.CrossEntropyLoss()
num_epochs=1




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
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
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
            # print(X, Y)
            # print(w,h)
            l = max(abs(y_max - y_min), abs(x_max - x_min))
            x_min = X - round(l / 2)
            x_max = x_min + l
            y_min = Y - round(l / 2)
            y_max = y_min + l
            y_min = y_min - 10
            y_max = y_max + 20
            x_min = x_min - 10
            x_max = x_max + 20
            # analysisframe = analysisframe[y_min:y_max, x_min:x_max]
            if x_max not in range(0, w):
                print("out_of_screan")
                continue
            if x_min not in range(0, w):
                print("out_of_screan")
                continue
            if y_max not in range(0, h):
                print("out_of_screan")
                continue
            if y_min not in range(0, h):
                print("out_of_screan")
                continue

            #print(frame.shape, frame[y_min:y_max, x_min: x_max, :].shape)

            image =Image.fromarray(frame[y_min:y_max, x_min: x_max, :])
            #image = Image.open("D:\\project\\dataset\\gen_dataset\\a\\a_0.png")
            #image_t = torch.tensor(image)
            #image_t = torch.transpose(image_t, 0, 2).transpose(1, 2)

            transformed_image = test_transform(image)
            #print(transformed_image[:])
            transformed_image = torch.unsqueeze(transformed_image, 0)

            #print(transformed_image.shape)
            output = model(transformed_image)
            print(mapper[torch.max(output.data, 1)[1].item()])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            #mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)


            #transformed_image = test_transform(image)
            cv2.imshow("Frame", frame[y_min:y_max, x_min: x_max, :])

cap.release()
cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()