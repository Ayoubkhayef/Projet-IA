from data_fetching import fetch_historical_data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def plot_predictions(prices, predicted_prices, future_dates):
    df = pd.DataFrame(prices, columns=['Date', 'Price'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index('Date', inplace=True)

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Price'], label='Prix Historique', color='blue')
    
    # Ajouter les prédictions au graphique
    future_df = pd.DataFrame({'Date': future_dates, 'Price': predicted_prices})
    future_df.set_index('Date', inplace=True)
    
    plt.plot(future_df.index, future_df['Price'], label='Prédictions', color='red', linestyle='--')
    
    plt.title('Prédiction des prix du Dogecoin')
    plt.xlabel('Date')
    plt.ylabel('Prix (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

def prepare_data(prices):
    df = pd.DataFrame(prices, columns=['Date', 'Price'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index('Date', inplace=True)

    # Normalisation des prix
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Price']])

    # Préparer les séquences pour LSTM
    X, y = [], []
    sequence_length = 30  # Utiliser 30 jours précédents pour prédire le suivant
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    return np.array(X), np.array(y), scaler, df

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future(model, last_sequence, scaler, future_days=30):
    future_predictions = []
    current_sequence = last_sequence

    for _ in range(future_days):
        prediction = model.predict(current_sequence[np.newaxis, :, :])[0, 0]
        future_predictions.append(prediction)
        current_sequence = np.append(current_sequence[1:], prediction)
    
    # Reconvertir les données prédictives à l'échelle d'origine
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions.flatten()

# Utilisation
prices = fetch_historical_data('dogecoin', days=365)
X, y, scaler, df = prepare_data(prices)
X = X.reshape((X.shape[0], X.shape[1], 1))

model = build_lstm_model((X.shape[1], 1))
model.fit(X, y, epochs=10, batch_size=32)

last_sequence = X[-1]  # Dernière séquence des données réelles
future_prices = predict_future(model, last_sequence, scaler, future_days=30)

# Générer des dates pour les prédictions
future_dates = pd.date_range(start=df.index[-1], periods=31, freq='D')[1:]

# Tracer les prédictions
plot_predictions(prices, future_prices, future_dates)
