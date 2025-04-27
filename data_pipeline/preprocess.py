import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib  # Import joblib for saving the scaler

def preprocess_data(data, features=None, train_split=0.7, sequence_length=100, ticker=None):
    """
    Preprocesses the stock data for training, validation, and testing.

    Parameters:
    - data: DataFrame containing stock data
    - features: List of features to use (default: ['Open', 'High', 'Low', 'Close', 'Volume'])
    - train_split: Fraction of data to use for training (default: 0.7)
    - sequence_length: Length of input sequences for LSTM (default: 100)
    - ticker: Stock ticker symbol (optional, for saving scaler)

    Returns:
    - Dictionary containing preprocessed training and testing data, scaler, and feature names
    """
    if features is None:
        features = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Filter the data to include only the selected features
    data = data[features]

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Save the scaler for future use
    if ticker:
        scaler_path = f"models/{ticker}_scaler.pkl"
        joblib.dump(scaler, scaler_path)

    # Create sequences for LSTM
    def create_sequences(data, step=sequence_length):
        x, y = [], []
        for i in range(len(data) - step):
            x.append(data[i:i + step])
            y.append(data[i + step, 3])  # 'Close' is the 4th feature (index 3)
        return np.array(x), np.array(y)

    x, y = create_sequences(scaled_data)

    # Split into training, validation, and testing sets
    train_size = int(len(x) * train_split)
    test_size = int(len(x) * (1 - train_split))
    x_train, y_train = x[:train_size], y[:train_size]
    x_test, y_test = x[train_size + test_size:], y[train_size + test_size:]

    return {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'scaler': scaler,
        'features': features  # Include feature names
    }
