import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
from keras.models import load_model


checkpoint_path = "D:\\project\\asl\\my_model.h5"



from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
CNNmove = Sequential()

CNNmove.add(Conv2D(64, (3, 3), activation='relu', input_shape=(20, 54,3)))
CNNmove.add(MaxPooling2D(pool_size=(2, 2)))
CNNmove.add(Conv2D(128, (3, 3), activation='relu'))
CNNmove.add(MaxPooling2D(pool_size=(2, 2)))
CNNmove.add(Flatten())
CNNmove.add(Dense(128, activation='relu'))

CNNmove.add(Dense(4, activation='softmax'))


def load_my_model():
    model = load_model(checkpoint_path)
    return model