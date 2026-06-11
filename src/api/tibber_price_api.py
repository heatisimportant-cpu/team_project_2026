# /api/tibber-prices-15min → returns the original 15-minute prices.
# /api/tibber-prices-hourly → returns hourly averaged prices.


# sample data

# {
#   "timestamps": [
#     "2026-06-10 00:00",
#     "2026-06-10 00:15",
#     "2026-06-10 00:30",
#     "2026-06-10 00:45"
#   ],
#   "prices": [22.1, 22.4, 22.0, 21.8]
# }

from fastapi import FastAPI
import requests
import pandas as pd

app = FastAPI()

BASE_URL = "https://api.energy-charts.info/price"
BZN = "DE-LU"


def get_price_dataframe():
    data = requests.get(
        BASE_URL,
        params={"bzn": BZN}
    ).json()

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(
            data["unix_seconds"],
            unit="s"
        ),
        "price_eur_mwh": data["price"]
    })

    # Tibber Darmstadt formula (ct/kWh)
    df["tibber_price_ct_kwh"] = (
        (df["price_eur_mwh"] / 10) * 1.19
    ) + 17.17

    return df


@app.get("/api/tibber-prices-15min")
def get_tibber_prices_15min():

    df = get_price_dataframe()

    return {
        "timestamps": (
            df["timestamp"]
            .dt.strftime("%Y-%m-%d %H:%M")
            .tolist()
        ),
        "prices": (
            df["tibber_price_ct_kwh"]
            .round(2)
            .tolist()
        )
    }


@app.get("/api/tibber-prices-hourly")
def get_tibber_prices_hourly():

    df = get_price_dataframe()

    hourly_df = (
        df.set_index("timestamp")
        .resample("1h")
        .mean()
        .reset_index()
    )

    return {
        "timestamps": (
            hourly_df["timestamp"]
            .dt.strftime("%Y-%m-%d %H:%M")
            .tolist()
        ),
        "prices": (
            hourly_df["tibber_price_ct_kwh"]
            .round(2)
            .tolist()
        )
    }