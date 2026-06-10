import requests
import pandas as pd

base_url = "https://api.energy-charts.info/price"
bzn = "DE-LU"

data = requests.get(
    base_url,
    params={"bzn": bzn}
).json()

# 15-minute prices
df = pd.DataFrame({
    "timestamp": pd.to_datetime(data["unix_seconds"], unit="s"),
    "price_eur_mwh": data["price"]
})

# Tibber price (ct/kWh)
df["tibber_kwh_hour"] = ((df["price_eur_mwh"] / 10) * 1.19) + 17.17

# Hourly average prices
hourly_df = (
    df.set_index("timestamp")
      .resample("1h")
      .mean()
      .reset_index()
)

print(hourly_df.head())

hourly_df.to_csv("../data/tibber_price_hourly.csv", index=False)