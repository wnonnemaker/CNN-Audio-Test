# src/data_preprocessing.py
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

def mp3_to_spectrogram(mp3_file):
    y, sr = librosa.load(mp3_file, sr=22050)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

def load_data(data_dir, input_shape=(128, 128)):
    X = []
    y = []

    for label, folder in enumerate(['bass', 'synth']):
        folder_path = os.path.join(data_dir, folder)
        for file in os.listdir(folder_path):
            if file.endswith('.mp3'):
                spectrogram = mp3_to_spectrogram(os.path.join(folder_path, file))
                spectrogram = np.resize(spectrogram, input_shape)
                X.append(spectrogram)
                y.append(label)
    
    X = np.array(X)
    y = np.array(y)

    # Normalize data
    X = X / np.max(X)

    return X, y

def get_train_test_split(X, y, test_size=0.2):
    # One-hot encode the labels
    from tensorflow.keras.utils import to_categorical
    y = to_categorical(y, num_classes=2)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Reshape data to fit the CNN input
    X_train = X_train.reshape((-1, 128, 128, 1))
    X_test = X_test.reshape((-1, 128, 128, 1))

    return X_train, X_test, y_train, y_test
