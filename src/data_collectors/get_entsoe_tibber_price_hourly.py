# single code block
"""
    this code uses entsoe api to get the day ahead prices and convert it to 
    tibber prices

    parametes: 
"""

from entsoe import EntsoePandasClient
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("ENTSOE_API_KEY")

def get_tomorrow_tibber_prices():
    client = EntsoePandasClient(api_key=API_KEY)

    # Tomorrow time
    start = (
        pd.Timestamp.now(tz="Europe/Berlin")
        .normalize() + pd.Timedelta(days=1)
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

    # Convert 15-minute prices to hourly averages
    hourly_df = (
        df.set_index("timestamp")
          .resample("1h")
          .mean()
          .reset_index()
    )

    # Tibber price (ct/kWh)
    hourly_df["tibber_kwh_hour"] = (
        (hourly_df["price_eur_mwh"] / 10) * 1.19
    ) + 17.17

    return hourly_df


df = get_tomorrow_tibber_prices()
print(df)
# df.to_csv("data/entsoe_tibber_price_hourly.csv",index=False)