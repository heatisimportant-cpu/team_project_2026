import requests
from datetime import datetime
import pandas as pd


def get_todays_weather(
    station_id=None,
    station_type="wmo",  # "wmo" or "dwd"
    lat=None,
    lon=None
):
   

    url = "https://api.brightsky.dev/weather"

    from datetime import timedelta
#from current time to end of the day

    now = datetime.now()
    start = now.replace(minute=0, second=0, microsecond=0)
    end = now.replace(hour=23, minute=59, second=59, microsecond=0)


    

    # tomorrow = datetime.now() + timedelta(days=1)
    # start = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
    # end = tomorrow.replace(hour=23, minute=59, second=59, microsecond=0)

    params = {
        "date": start.isoformat(),
        "last_date": end.isoformat(),
        "tz": "Europe/Berlin",
        "units": "dwd"
    }


    # Station-based query
    if station_id:
        if station_type.lower() == "wmo":
            params["wmo_station_id"] = station_id
        elif station_type.lower() == "dwd":
            params["dwd_station_id"] = station_id
        else:
            raise ValueError(
                "station_type must be either 'wmo' or 'dwd'"
            )

    # Coordinate-based query
    elif lat is not None and lon is not None:
        params["lat"] = lat
        params["lon"] = lon

    else:
        raise ValueError(
            "Provide either station_id or lat/lon coordinates."
        )

    response = requests.get(url, params=params)
    response.raise_for_status()

    weather_records = response.json().get("weather", [])

    df = pd.DataFrame([
        {
            "timestamp": record.get("timestamp"),
            "temperature": record.get("temperature")
        }
        for record in weather_records
    ])

    # Convert timestamp format
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"])
        .dt.tz_localize(None)
    )

    return df




df = get_todays_weather(
    station_id="10338",
    station_type="wmo"
)


# # DWD station
# df = get_todays_weather(
#     station_id="01766",
#     station_type="dwd"
# )

# #Coordinates
# df = get_todays_weather(
#     lat=52.3759,
#     lon=9.7320
# )

print(df)




