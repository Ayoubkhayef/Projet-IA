from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_model(data):
    price_data, _ = data  # Ignorer les tweets pour simplifier
    X = price_data['Log_Returns'].values[:-1].reshape(-1, 1, 1)
    y = price_data['Log_Returns'].values[1:]

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, 1)),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=32)
    return model
