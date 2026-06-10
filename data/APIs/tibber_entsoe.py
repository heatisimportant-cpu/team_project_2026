
# 15-minute next-day prices: 
# http://127.0.0.1:8000/api/tibber-nextday-prices-15min
# Hourly next-day prices: 
# http://127.0.0.1:8000/api/tibber-nextday-prices-hourly





from fastapi import FastAPI
from entsoe import EntsoePandasClient
from dotenv import load_dotenv

import pandas as pd
import os

load_dotenv()
app = FastAPI()

API_KEY = os.getenv("ENTSOE_API_KEY")


def get_price_dataframe():

    client = EntsoePandasClient(api_key=API_KEY)

    # Tomorrow in Germany
    start = (
        pd.Timestamp.now(tz="Europe/Berlin")
        .normalize()
        + pd.Timedelta(days=1)
    )
    end = start + pd.Timedelta(days=1)

    prices = client.query_day_ahead_prices(
        country_code="DE_LU",
        start=start,
        end=end
    )

    df = pd.DataFrame({
        "timestamp": prices.index.tz_localize(None),
        "price_eur_mwh": prices.values
    })

    # Tibber Darmstadt formula (ct/kWh)
    df["tibber_price_ct_kwh"] = (
        (df["price_eur_mwh"] / 10) * 1.19
    ) + 17.17

    return df


@app.get("/api/tibber-nextday-prices-15min")
def get_tibber_nextday_prices_15min():

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


@app.get("/api/tibber-nextday-prices-hourly")
def get_tibber_nextday_prices_hourly():

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