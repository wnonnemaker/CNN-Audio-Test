# src/predict.py
import numpy as np
from tensorflow.keras.models import load_model
from src.data_preprocessing import mp3_to_spectrogram

def predict_sound(model_path, mp3_file):
    # Load the trained model
    model = load_model(model_path)

    # Convert the MP3 file to a spectrogram
    spectrogram = mp3_to_spectrogram(mp3_file)
    spectrogram = np.resize(spectrogram, (128, 128))
    spectrogram = spectrogram.reshape((1, 128, 128, 1))  # Reshape to fit the CNN input

    # Make a prediction
    prediction = model.predict(spectrogram)
    predicted_class = np.argmax(prediction)

    if predicted_class == 0:
        print("Bass sound detected")
    else:
        print("Synth sound detected")
