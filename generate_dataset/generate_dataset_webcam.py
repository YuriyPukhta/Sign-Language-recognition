import os
import torch
import cv2
import mediapipe as mp
import numpy as np
import time
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D, BatchNormalization
from keras.models import Model
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd


mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
_, frame = cap.read()
h, w, c = frame.shape

num_of_skip_frame = 5
skip_next = False
curent_frame_skip = 0

image_per_sing = 150
counter_image = 999
sign = -1
reference_dataset = "D:\\project\\dataset\\ASL_Dataset\\asl_dataset"
save_dataset_path = "D:\\project\\dataset\\gen_dataset"
save_sign_path = ""


Landmarks = []
def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

if not os.path.exists(save_dataset_path):
    os.makedirs(save_dataset_path)

ref_file = []
name_sign = []
for subdir in os.listdir(reference_dataset):
    image___ = os.path.join(reference_dataset, subdir)
    ref_file.append(os.path.join(image___, os.listdir(image___)[1]))

    name_sign.append(subdir)
while True:
    _, frame = cap.read()

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        columns = []
        for i in range(21):
            columns.append(f"x{i}")
            columns.append(f"y{i}")
            columns.append(f"z{i}")
        columns.append("sign")
        df = pd.DataFrame(Landmarks,
                          columns=columns)
        print(df.head(20))
        df.to_csv('LM.csv', index=False)
        print("Escape hit, closing...")
        break

    if(sign >=  len(ref_file)):
        columns = []
        for i in range(21):
            columns.append(f"x{i}")
            columns.append(f"y{i}")
            columns.append(f"z{i}")
        columns.append("sign")
        df = pd.DataFrame(Landmarks,
                          columns=columns)
        print(df.head(20))
        df.to_csv('LM.csv', index=False)
    if(image_per_sing <= counter_image):
        sign += 1
        print("Next sign")
        cv2.imshow("ref frame", cv2.imread(ref_file[sign]))
        print(name_sign[sign])
        counter_image = 0
        input("sign true ?")
        save_sign_path = os.path.join(save_dataset_path, name_sign[sign])
        if not os.path.exists(save_sign_path):
            os.makedirs(save_sign_path)
        continue

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        handLMs = hand_landmarks[0]
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

        X = int(X/len(handLMs.landmark))
        Y = int(Y/len(handLMs.landmark))
        l = max(abs(y_max - y_min), abs(x_max - x_min))
        x_min = X - round(l/2)
        x_max = x_min + l
        y_min = Y - round(l/2)
        y_max = y_min + l
        y_min = y_min - 10
        y_max = y_max + 20
        x_min = x_min - 10
        x_max = x_max + 20
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
        new_frame = frame[y_min:y_max, x_min: x_max, :]
        image = new_frame.copy()
        image = cv2.resize(image, (200,200), interpolation = cv2.INTER_AREA)


        gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)

        text = "Blurry"
        cv2.putText(frame, "{}: {:.2f}".format(text, fm), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        start_point = (x_min, y_min )
        end_point = (x_max , y_max)

        frame = cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)
        if fm > 50:
            if (curent_frame_skip >= num_of_skip_frame):
                curent_frame_skip = 0
                img_name = f"{name_sign[sign]}_{counter_image}.png"
                counter_image += 1
                cv2.imshow("hande frame", image)
                print(f"image save {img_name}")
                cv2.imshow("Frame", frame)
                image = Image.fromarray(frame[y_min:y_max, x_min: x_max, :])
                new_hande = []
                for l in handLMs.landmark:
                    new_hande.append(l.x)
                    new_hande.append(l.y)
                    new_hande.append(l.z)
                new_hande.append(name_sign[sign])
                Landmarks.append(new_hande)
            else:
                curent_frame_skip += 1
                cv2.imshow("Frame", frame)
                continue
        else:
            print("blure")




cap.release()
cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()