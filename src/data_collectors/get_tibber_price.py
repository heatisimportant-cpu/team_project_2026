import requests
import pandas as pd

base_url = "https://api.energy-charts.info/price"
bzn = "DE-LU"
url = f"{base_url}?bzn={bzn}"

data = requests.get(url).json()

df = pd.DataFrame({
    "timestamp": pd.to_datetime(data["unix_seconds"], unit="s"),
    "price_eur_mwh": data["price"]
})

#tibber price for darmstadt in eur/kwh
df["tibber_kwh"] = ((df["price_eur_mwh"] / 10) * 1.19) + 17.17

print(df.head())
df.to_csv("data/tibber_price.csv",index=False)
