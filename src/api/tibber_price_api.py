from fastapi import FastAPI
import requests
import pandas as pd

app = FastAPI()

BASE_URL = "https://api.energy-charts.info/price"
BZN = "DE-LU"


@app.get("/api/tibber-prices")
def get_tibber_prices():

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

    # Tibber Darmstadt formula
    df["tibber_price_ct_kwh"] = (
        (df["price_eur_mwh"] / 10) * 1.19
    ) + 17.17

    return {
        "timestamps":
            df["timestamp"]
            .dt.strftime("%Y-%m-%d %H:%M")
            .tolist(),

        "prices":
            df["tibber_price_ct_kwh"]
            .round(2)
            .tolist()
    }