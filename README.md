TASK 2 DRIVE LINK: https://drive.google.com/file/d/19tH42frzVwvnuyXjxK6HMlwzd8Z1Em0o/view?usp=drive_link

TASK 3 DRIVE LINK:https://drive.google.com/file/d/1SnjWSa7ip8Kd6fbTpFEPOWGF1UB0GC8G/view?usp=drive_link

TASK 2: CODE

import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from google.colab import files


uploaded_files = files.upload()


def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def save_uploaded_files(uploaded_files):
    file_paths = []
    for filename in uploaded_files.keys():
        with open(filename, 'wb') as f:
            f.write(uploaded_files[filename])
        file_paths.append(filename)
    return file_paths

file_paths = save_uploaded_files(uploaded_files)


if len(file_paths) < 2:
    raise ValueError("Not enough files uploaded. Please upload at least two audio files for training.")


data = []
labels_gender = []
labels_age = []
labels_emotion = []

for file_path in file_paths:
    features = extract_features(file_path)
    data.append(features)

    gender = "Male" if "Male" in file_path else "Female"
    emotion = np.random.randint(1, 9)
    age = np.random.randint(20, 80)
    labels_gender.append(gender)
    labels_emotion.append(emotion)
    labels_age.append(age)

data = np.array(data)
labels_gender = np.array(labels_gender)
labels_emotion = np.array(labels_emotion)
labels_age = np.array(labels_age)


gender_encoder = LabelEncoder()
labels_gender_encoded = gender_encoder.fit_transform(labels_gender)

emotion_encoder = LabelEncoder()
labels_emotion_encoded = emotion_encoder.fit_transform(labels_emotion)


scaler = StandardScaler()
data = scaler.fit_transform(data)


X_train, X_test, y_gender_train, y_gender_test = train_test_split(data, labels_gender_encoded, test_size=0.2, random_state=42)
X_train, X_test, y_age_train, y_age_test = train_test_split(data, labels_age, test_size=0.2, random_state=42)
X_train, X_test, y_emotion_train, y_emotion_test = train_test_split(data, labels_emotion_encoded, test_size=0.2, random_state=42)

def build_model(output_units, output_activation):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(data.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(output_units, activation=output_activation)
    ])
    model.compile(loss="sparse_categorical_crossentropy" if output_units > 1 else "mse",
                  optimizer="adam",
                  metrics=["accuracy"] if output_units > 1 else ["mae"])
    return model


gender_model = build_model(2, "softmax")
gender_model.fit(X_train, y_gender_train, epochs=10, batch_size=8, verbose=1)


gender_model.save("gender_model.h5")


emotion_model = build_model(len(np.unique(labels_emotion_encoded)), "softmax")
emotion_model.fit(X_train, y_emotion_train, epochs=10, batch_size=8, verbose=1)


emotion_model.save("emotion_model.h5")


age_model = build_model(1, "linear")
age_model.fit(X_train, y_age_train, epochs=10, batch_size=8, verbose=1)


age_model.save("age_model.h5")


def process_audio(file_path):
    features = extract_features(file_path)
    features = scaler.transform(features.reshape(1, -1))


    gender_pred = np.argmax(gender_model.predict(features))
    gender = gender_encoder.inverse_transform([gender_pred])[0]


    age_pred = age_model.predict(features)[0][0]


    emotion_pred = np.argmax(emotion_model.predict(features))
    emotion = emotion_encoder.inverse_transform([emotion_pred])[0]

    return f"Predicted Gender: {gender}, Predicted Age: {int(age_pred)}, Predicted Emotion: {emotion}"

if uploaded_files:
    for uploaded_filename in uploaded_files.keys():
        result = process_audio(uploaded_filename)
        print(result)
else:
    print("Please upload an audio file for processing.")

TASK 3: CODE

import cv2
import numpy as np
from sklearn.cluster import KMeans
import urllib.request
import os
from google.colab import files
import matplotlib.pyplot as plt

def download_yolo():
    if not os.path.exists("yolov3.cfg"):
        urllib.request.urlretrieve("https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg", "yolov3.cfg")
    if not os.path.exists("yolov3.weights"):
        urllib.request.urlretrieve("https://pjreddie.com/media/files/yolov3.weights", "yolov3.weights")
    if not os.path.exists("coco.names"):
        urllib.request.urlretrieve("https://github.com/pjreddie/darknet/raw/master/data/coco.names", "coco.names")

download_yolo()

yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
yolo_layer_names = yolo_net.getLayerNames()
yolo_output_layers = [yolo_layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

def detect_color(image):
    if image.shape[0] == 0 or image.shape[1] == 0:
        return [0, 0, 0]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (100, 100))
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(pixels)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    return dominant_color

def classify_color(dominant_color):
    r, g, b = dominant_color
    if b > r and b > g:
        return "Blue"
    elif r > b and r > g:
        return "Red"
    elif g > r and g > b:
        return "Green"
    elif abs(r - g) < 20 and abs(g - b) < 20:
        return "Gray/Black/White"
    else:
        return "Other"

def detect_objects(image):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    detections = yolo_net.forward(yolo_output_layers)

    cars = []
    persons = person_cascade.detectMultiScale(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 1.1, 4)

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2:
                center_x, center_y, w, h = (obj[:4] * np.array([width, height, width, height])).astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                cars.append((x, y, w, h))

    return cars, persons

def process_image(image):
    cars, persons = detect_objects(image)
    person_count = len(persons)
    for (x, y, w, h) in cars:
        car_roi = image[y:y+h, x:x+w]
        if car_roi.size == 0:
            continue
        dominant_color = detect_color(car_roi)
        car_color = classify_color(dominant_color)
        rectangle_color = (0, 0, 255) if car_color == "Blue" else (255, 0, 0)
        cv2.rectangle(image, (x, y), (x+w, y+h), rectangle_color, 3)
        cv2.putText(image, car_color, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rectangle_color, 2)

    for (x, y, w, h) in persons:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(image, f'People: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image

def select_image():
    uploaded = files.upload()
    for filename in uploaded.keys():
        image = cv2.imread(filename)
        processed_image = process_image(image)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        plt.imshow(processed_image)
        plt.axis('off')
        plt.show()

select_image()
