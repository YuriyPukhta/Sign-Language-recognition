import os
import torch
import cv2
import mediapipe as mp
import numpy as np
import time
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D, BatchNormalization
from keras.models import Model
from PIL import Image
import pandas as pd
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
SHOW_LANDMARKS = False
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
h, w, c = 00, 99 ,99

num_of_skip_frame = 5
skip_next = False
curent_frame_skip = 0

image_per_sing = 150
counter_image = 999
sign = -1
dataset = "D:\\project\\dataset\\ASL_Dataset\\asl_dataset"
save_dataset_path = "D:\\project\\dataset\\gen_land_marks_dataset"
save_sign_path = ""

def min_max_scale(lst):
    min_val = min(lst)
    max_val = max(lst)
    scaled_lst = [(x - min_val) / (max_val - min_val) for x in lst]
    return scaled_lst
Landmarks = []


if not os.path.exists(save_dataset_path):
    os.makedirs(save_dataset_path)

ref_file = []
name_sign = []
for subdir in os.listdir(dataset):
    sing_path = os.path.join(dataset, subdir)
    for path_image in os.listdir(sing_path):
        path = os.path.join(sing_path, path_image)
        image = cv2.imread(os.path.join(sing_path, path_image))
        #cv2.imshow("org Frame", image)
        framergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        hand_landmarks = result.multi_hand_landmarks
        if hand_landmarks:
            handLMs = hand_landmarks[0]
            if SHOW_LANDMARKS:
                for  hl in enumerate(handLMs):
                    mp_drawing.draw_landmarks(
                        image,
                        hl,
                        mphands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            

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
            new_hande.append(subdir)
            Landmarks.append(new_hande)

columns = []
for i in range(21):
    columns.append(f"x{i}")
    columns.append(f"y{i}")
    columns.append(f"z{i}")
columns.append("sign")
df = pd.DataFrame(Landmarks,
                  columns=columns)
df.to_csv(os.path.join(save_dataset_path, 'LM_min_max .csv'), index=False)

