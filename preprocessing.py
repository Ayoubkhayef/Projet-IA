import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def preprocess_data(price_data, tweets):
    # Calcul des rendements log
    price_data['Log_Returns'] = (price_data['Close'] / price_data['Close'].shift(1)).apply(lambda x: np.log(x))
    price_data.dropna(inplace=True)

    # Analyse des tweets
    vectorizer = CountVectorizer(max_features=1000)
    tweet_vectors = vectorizer.fit_transform(tweets)

    # Retourner les données combinées
    return price_data, tweet_vectors
