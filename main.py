from scripts.data_fetching import fetch_historical_data, fetch_tweets
from scripts.preprocessing import preprocess_data
from scripts.model_training import train_model

def main():
    # Récupérer les données de prix et de sentiment
    price_data = fetch_historical_data('DOGE-USD', '2021-01-01', '2023-12-01')
    tweets = fetch_tweets('Dogecoin', 100)

    # Prétraiter les données
    processed_data = preprocess_data(price_data, tweets)

    # Entraîner le modèle
    model = train_model(processed_data)

    # Sauvegarder le modèle
    model.save('models/meme_coin_model.h5')

if __name__ == "__main__":
    main()
