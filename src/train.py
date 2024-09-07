# src/train.py
from src.data_preprocessing import load_data, get_train_test_split
from src.model import create_cnn_model
import os

def train_model(data_dir, model_save_path, epochs=20, batch_size=32):
    # Load data
    X, y = load_data(data_dir)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    # Create model
    input_shape = (128, 128, 1)
    model = create_cnn_model(input_shape)

    # Train model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # Save the trained model
    model.save(model_save_path)

    print(f"Model saved at {model_save_path}")
