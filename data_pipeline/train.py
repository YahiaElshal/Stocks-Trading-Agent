import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fetch_data import fetch_yahoo_data 
from preprocess import preprocess_data
import joblib
import numpy as np


def build_and_train_lstm_model(x_train, y_train, epochs=100, batch_size=32, verbose=1, callback=None):
    """
    Builds and trains an LSTM model for stock price prediction.
    
    Parameters:
    - x_train: Training input sequences
    - y_train: Training target values
    - epochs: Number of training epochs (default: 100)
    - batch_size: Batch size for training (default: 32)
    - verbose: Verbosity level (0: silent, 1: progress bar, 2: one line per epoch)
    - callback: Optional callback function to report progress
    
    Returns:
    - Trained LSTM model
    """
    model = Sequential([
        LSTM(60, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
        Dropout(0.2),
        LSTM(80, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(
        optimizer='adam', 
        loss='mean_squared_error', 
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    
    # Train with or without progress reporting
    if callback:
        for epoch in range(epochs):
            model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=0)
            callback(epoch / epochs)
    else:
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    return model

def save_model(model, filepath):
    """
    Saves the trained model to disk.
    
    Parameters:
    - model: Trained Keras model
    - filepath: Path to save the model
    """
    model.save(filepath)


if __name__ == "__main__":
    # Step 1: Fetch data from Yahoo Finance
    ticker = "TCOM"  
    start_date = "2020-01-01"
    end_date = "2025-4-27"
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = fetch_yahoo_data(ticker, start_date)
    
    if data is None:
        print(f"No data available for {ticker}. Exiting.")
        exit()

    print(f"Data fetched successfully. Shape: {data.shape}")

    # Step 2: Preprocess the data
    sequence_length = 100
    print("Preprocessing data...")
    processed_data = preprocess_data(data, sequence_length=sequence_length, ticker=ticker)
    x_train = processed_data['x_train']
    y_train = processed_data['y_train']
    x_test = processed_data['x_test']
    y_test = processed_data['y_test']

    print(f"Training data shape: {x_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {x_test.shape}, {y_test.shape}")

    # Step 3: Train the LSTM model
    epochs = 100  # Number of epochs for training
    batch_size = 32
    print("Training the LSTM model...")
    model = build_and_train_lstm_model(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Step 4: Evaluate the model on the test set
    print("Evaluating the model on the test set...")
    y_pred = model.predict(x_test, verbose=0)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Absolute Error (MAE) on Test Set: {mae}")
    print(f"Mean Squared Error (MSE) on Test Set: {mse}")

    # Step 5: Save the model
    model_path = f"models/{ticker}.keras"
    save_model(model, model_path)
    print(f"Model saved to {model_path}")

    # Step 6: Load the scaler and make predictions for the next 5 days
    print("Loading the scaler...")
    scaler_path = f"models/{ticker}_scaler.pkl"
    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded from {scaler_path}")

    # Final evaluation results
    evaluation_results = {
        "mae": mae,
        "mse": mse
    }
    print("Evaluation Results:", evaluation_results)