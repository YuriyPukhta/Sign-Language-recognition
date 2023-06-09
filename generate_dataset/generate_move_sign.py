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


def min_max_scale(lst):
    min_val = min(lst)
    max_val = max(lst)
    scaled_lst = [(x - min_val) / (max_val - min_val) for x in lst]
    return scaled_lst

def min_max_scale_hand_move(lst):
    min_val = min([min(arr) for arr in lst])
    max_val = max([max(arr) for arr in lst])
    #print(min_val, max_val)
    scaled_lst = [[(x - min_val) / (max_val - min_val) for x in hand] for hand in lst]
    return scaled_lst

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



save_dataset_path = "D:\\project\\dataset\\gen_land_marks_dataset"
sign_name = ["YES", "NO", "Helow","THANK YOU" ]#"THANK YOU",] #, "HELP", "PLEASE", "MORE", "THANK YOU", "NO", "STOP", "NOW", "STAND UP", "WORK", "SORY" ]
Landmarks = []
def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()


if not os.path.exists(save_dataset_path):
    os.makedirs(save_dataset_path)
num_of_frame = 20
sq_num = 20



Landmarks = []
for sign in sign_name:
    print(sign)
    #time.sleep(5)
    for sq in range(sq_num):
        hand1_move_X = []
        hand1_move_Y = []
        hand1_move_Z = []

        hand2_move_X = []
        hand2_move_Y = []
        hand2_move_Z = []

        fr = 0
        while fr in range(num_of_frame):
            _, frame = cap.read()
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(framergb)
            left_hand = result.left_hand_landmarks
            pose = result.pose_landmarks
            if not left_hand or not pose:
                print("skip")
                continue
            if left_hand and pose:
                hand1 = Get_point(left_hand.landmark, pose.landmark)


            fr += 1
            text = "Frame"
            cv2.putText(frame, "{}: {:.2f}".format(text, fr), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

            print(f"{sq} frame {fr} reedy")
            time.sleep(0.05)



            hands_move =hand1
            hands_move.append(fr)
            hands_move.append(sign)
            Landmarks.append(hands_move)

        print("sleep")
        time.sleep(1.5)


columns = []
for i in range(21):
    columns.append(f"hx{i}")
    columns.append(f"hy{i}")
    columns.append(f"hz{i}")
for i in range(33):
    columns.append(f"px{i}")
    columns.append(f"py{i}")
    columns.append(f"pz{i}")
columns.append("frame")
columns.append("sign")
df = pd.DataFrame(Landmarks,
                  columns=columns)
df.to_csv('D:\\project\\dataset\\gen_land_marks_dataset\\FrameLM_1.csv', index=False)

cap.release()
cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()