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
