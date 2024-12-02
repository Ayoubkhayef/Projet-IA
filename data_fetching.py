from pycoingecko import CoinGeckoAPI

def fetch_historical_data(coin_id, vs_currency='usd', days='max'):
    cg = CoinGeckoAPI()
    data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency=vs_currency, days=days)
    prices = data['prices']
    return prices

# Exemple
prices = fetch_historical_data('dogecoin', days=365)
print(prices[:5])
