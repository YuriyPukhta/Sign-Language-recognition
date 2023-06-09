import os

import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp
import numpy as np
import time
#from model import Resnet50WithFPN, load_model, test_transform
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
from model import  CNNmove, load_my_model
import os
from PIL import Image
from tqdm import tqdm
import numpy as np

model = load_my_model()

sign_name = [ "Helow","NO", "THANK YOU", "YES"]
Landmarks = []
def Get_point(handLMs, pose):
    new_hand = []
    for hl in handLMs:
        new_hand.append(hl.x)
        new_hand.append(hl.y)
        new_hand.append(hl.z)
    for hl in pose:
        new_hand.append(hl.x)
        new_hand.append(hl.y)
        new_hand.append(hl.z)
    return new_hand




mphands = mp.solutions.holistic
hands = mphands.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
_, frame = cap.read()
h, w, c = frame.shape
skip_time = 0
while True:
    _, frame = cap.read()

    _, frame = cap.read()
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    left_hand = result.left_hand_landmarks
    pose = result.pose_landmarks
    if not left_hand or not pose:
        print("skip")
        if skip_time > 10:
            Landmarks = []
            skip_time = 0
        continue
    if left_hand and pose:
        hand1 = Get_point(left_hand.landmark, pose.landmark)

    Landmarks.append(hand1)
    if len(Landmarks) == 20:
        data_Array = np.array(Landmarks)
        #print(data_Array.shape)
        data_Array = data_Array.reshape(1, 20, 54, 3)
        output = model.predict(data_Array, verbose = 0)
        output[0][1] = output[0][1] / 6
        print(output)
        print(np.argmax(output[0]))
        Landmarks.pop(0)

cap.release()
cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()