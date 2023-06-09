# Sign Language Recognition
This project aims to develop a Sign Language Recognition system using machine learning techniques. The system is designed to recognize hand gestures and translate them into corresponding textual representations, enabling communication between individuals who are proficient in sign language and those who are not.

Prerequisites
Before getting started with this project, ensure that you have the following prerequisites installed on your machine:
```
Python (version >= 3.6)
Pytorch (version >= 2.0)
Mediapipe (version >= 0.2)
OpenCV (version >= 4.0)
Numpy (version >= 1.16)
```
# How to Reproduce
To reproduce the Sign Language Recognition system, follow the steps below:

Clone this repository to your local machine:
```
git clone https://github.com/your-username/sign-language-recognition.git
```
Change into the project directory:
```
cd sign-language-recognition
```
Install the required dependencies:
```
pip install -r requirements.txt
```
Run the application:
```
python user/main.py
```
# How create your own sign set
You may want to create your own character set for another language.
## Dataset generation for training model on hand images
You need a folder with a sub-catalogue of symbols that you want to make, in each such folder there should be one image of this symbol that will be used as a reference when creating the program

Change ```generate_dataset/generate_dataset_webcam.py``` the path in the file to yours 
and specify how many characters you want for each character ```images_per_sign = 100```

``` python 
reference_dataset = "reference dataset path"
save_dataset_path = "save path"
```
Run 
```
python generate_dataset/generate_dataset_landmarks.py
```
Show the sign that was selected as a reference, the program will show which sign should be shown right now, try to show all possible hand positions for this sign.

## Dataset generation for training model on hand landmarks 
You should do the first step with the generation of hand image or have a ready dataset with images
in ```generate_dataset/generate_dataset_landmarks.py``` change patht to yours.

```
dataset = "yours dataset"
save_dataset_path = "save_path"
```

Run 
```
python generate_dataset/generate_dataset_landmarks.py
```
When the program finish, you will receive a file with a set of 21 landmarks for your signs.
# How to Train a Model
if you have a dataset with features you can train your model

## Hand images model
Change ```train_file/Train_image/Train_resnet_GenData.ipynd``` the path in the file to yours.

```
trainData_Dir = "yours_dataset_path"
```
and save model path 
```
checkpoint_path = "save_model_path"
```
Run file
Wait for the training process to complete. The trained model will be saved at the specified save_model_path.

## Hand landmarks model
Change ```train_file/Train_landmarks/Train_CNN_landmarks.ipynd``` the path in the file to yours CSV file.

```
trainData_Dir = "yours_dataset_path.CSV"
```
and save model path 
```
checkpoint_path = "save_model_path"
```
Run file
Wait for the training process to complete. The trained model will be saved at the specified save_model_path.

# How to Preprocess the Data & Generate Features
Hand images model:
For this, you need to crop the image of the hand in the picture, this picture will be transformed and resize(224,224) into a tensor and the model will be able to process it.
To generate features, we use pre-trained ResNet50.

Hand landmarks model:
For this, you need to process keypoints on the hand iamge with Mediapipe and cut them into a fotmat [x1, y1, z1... x21, y21, z21] 
The landmark models requires only 21 hand keypoints.
