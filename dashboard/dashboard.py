# 1. Inject df into builtins to prevent NameError in get_temp.py during import
import builtins
builtins.df = None

# 2. Monkeypatch EntsoePandasClient to handle early-morning queries and cache responses
# to prevent slow imports and API failures on Streamlit reruns
try:
    from entsoe import EntsoePandasClient
    import pandas as pd

    if not hasattr(EntsoePandasClient, "_query_cache"):
        EntsoePandasClient._query_cache = {}

    _original_query = EntsoePandasClient.query_day_ahead_prices

    def _patched_query(self, country_code, start, end, *args, **kwargs):
        cache_key = (country_code, pd.Timestamp(start), pd.Timestamp(end))
        if cache_key in EntsoePandasClient._query_cache:
            return EntsoePandasClient._query_cache[cache_key]

        try:
            res = _original_query(self, country_code, start, end, *args, **kwargs)
            EntsoePandasClient._query_cache[cache_key] = res
            return res
        except Exception as e:
            today_start = pd.Timestamp.now(tz="Europe/Berlin").normalize()
            today_end = today_start + pd.Timedelta(days=1)
            # Avoid infinite recursion if querying today's date also fails
            if start.normalize() == today_start:
                raise e
            
            fallback_key = (country_code, today_start, today_end)
            if fallback_key in EntsoePandasClient._query_cache:
                return EntsoePandasClient._query_cache[fallback_key]

            res = _original_query(self, country_code, today_start, today_end, *args, **kwargs)
            EntsoePandasClient._query_cache[fallback_key] = res
            return res

    EntsoePandasClient.query_day_ahead_prices = _patched_query
except Exception:
    pass

import sys
from pathlib import Path

# Add project root directory to sys.path so 'src' can be imported
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import datetime

from src.data_collectors.get_entsoe_tibber_price_hourly import get_tomorrow_tibber_prices
from src.data_collectors.get_temp import get_todays_weather

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Energy & Weather Hub",
    page_icon="⚡",
    layout="wide"
)

# -----------------------------
# Cached API Functions
# -----------------------------
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def cached_get_tomorrow_tibber_prices():
    return get_tomorrow_tibber_prices()

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def cached_get_todays_weather(station_id=None, station_type="wmo", lat=None, lon=None):
    return get_todays_weather(station_id=station_id, station_type=station_type, lat=lat, lon=lon)

# Helper function to clear all caches and force a reload
def force_refresh_data():
    st.cache_data.clear()
    if hasattr(EntsoePandasClient, "_query_cache"):
        EntsoePandasClient._query_cache.clear()
    st.rerun()

# -----------------------------
# Session State Navigation
# -----------------------------
if "active_page" not in st.session_state:
    st.session_state.active_page = "Electricity Prices"

# -----------------------------
# Centered Top Navigation Buttons
# -----------------------------
col_left, col_btn1, col_btn2, col_right = st.columns([3.5, 2.5, 2.5, 3.5])

with col_btn1:
    btn1_type = "primary" if st.session_state.active_page == "Electricity Prices" else "secondary"
    if st.button("⚡ Electricity Prices", type=btn1_type, use_container_width=True):
        st.session_state.active_page = "Electricity Prices"
        st.rerun()

with col_btn2:
    btn2_type = "primary" if st.session_state.active_page == "Temperature Data" else "secondary"
    if st.button("🌡️ Temperature Data", type=btn2_type, use_container_width=True):
        st.session_state.active_page = "Temperature Data"
        st.rerun()

# st.markdown("---")  # Horizontal line separating nav buttons from page content

page = st.session_state.active_page

# -----------------------------
# PAGE: Electricity Prices
# -----------------------------
if page == "Electricity Prices":
    # Header with Title and Refresh Button side-by-side
    col_title, col_refresh = st.columns([9, 1])
    with col_title:
        st.title("⚡ Electricity Prices")
        st.caption("Hourly Day-Ahead prices and Tibber Equivalent cost projections")
    with col_refresh:
        # Align refresh button visually with the header
        st.write("<style>div.stButton > button { margin-top: 24px; float: right; }</style>", unsafe_allow_html=True)
        if st.button("🔄", key="refresh_prices", help="Force refresh data from API"):
            force_refresh_data()

    with st.spinner("Fetching electricity prices..."):
        try:
            df = cached_get_tomorrow_tibber_prices()
        except Exception as e:
            st.error(f"Failed to fetch electricity prices: {e}")
            df = None

    if df is not None:
        # KPIs row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg ENTSO-E Price", f"{df['price_eur_mwh'].mean():.2f} €/MWh")
        with col2:
            st.metric("Max ENTSO-E Price", f"{df['price_eur_mwh'].max():.2f} €/MWh")
        with col3:
            st.metric("Avg Tibber Price", f"{df['tibber_kwh_hour'].mean():.2f} ct/kWh")
        with col4:
            st.metric("Max Tibber Price", f"{df['tibber_kwh_hour'].max():.2f} ct/kWh")
            
        st.markdown("<br>", unsafe_allow_html=True)

        # Plotly Main Chart
        fig = px.line(
            df,
            x="timestamp",
            y=["price_eur_mwh", "tibber_kwh_hour"],
            markers=True,
            title="Hourly Day-Ahead Price Trends",
            labels={
                "value": "Price (Metric)",
                "variable": "Price Type",
                "timestamp": "Time"
            }
        )
        
        fig.update_layout(
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Separate Charts Side-by-Side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ENTSO-E Day-Ahead Price")
            fig1 = px.bar(
                df,
                x="timestamp",
                y="price_eur_mwh"
            )
            fig1.update_layout(height=350)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("Tibber Equivalent Price")
            fig2 = px.bar(
                df,
                x="timestamp",
                y="tibber_kwh_hour"
            )
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

        # Raw Data Section
        st.markdown("---")
        with st.expander("View Raw Electricity Price Data Table"):
            st.dataframe(df, use_container_width=True, hide_index=True)

# -----------------------------
# PAGE: Temperature Data
# -----------------------------
elif page == "Temperature Data":
    # Header with Title and Refresh Button side-by-side
    col_title, col_refresh = st.columns([9, 1])
    with col_title:
        st.title("🌡️ Temperature Projections")
        st.caption("Hourly temperature forecast from Bright Sky API")
    with col_refresh:
        # Align refresh button visually with the header
        st.write("<style>div.stButton > button { margin-top: 24px; float: right; }</style>", unsafe_allow_html=True)
        if st.button("🔄", key="refresh_temp", help="Force refresh data from API"):
            force_refresh_data()

    # Query Type Selection and Inputs in Sidebar
    with st.sidebar:
        st.header("Weather Settings")
        query_type = st.selectbox(
            "Select Weather Input Type:",
            ["Station ID", "Coordinates"]
        )
        
        # Input panel depending on select option
        if query_type == "Station ID":
            station_id = st.text_input("Station ID:", value="10338")  # default Hannover WMO
            station_type = st.radio("Station Type:", ["wmo", "dwd"], horizontal=True)
            lat, lon = None, None
        else:
            lat = st.number_input("Latitude:", value=49.8728, format="%.4f")  # default Darmstadt
            lon = st.number_input("Longitude:", value=8.6512, format="%.4f")
            station_id, station_type = None, None

    st.markdown("<br>", unsafe_allow_html=True)

    with st.spinner("Fetching temperature forecast..."):
        try:
            if query_type == "Station ID":
                df_temp = cached_get_todays_weather(station_id=station_id, station_type=station_type)
            else:
                df_temp = cached_get_todays_weather(lat=lat, lon=lon)
        except Exception as e:
            st.error(f"Failed to fetch weather data: {e}")
            df_temp = None

    if df_temp is not None and not df_temp.empty:
        # KPIs row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Temperature", f"{df_temp['temperature'].mean():.1f} °C")
        with col2:
            st.metric("Maximum Temperature", f"{df_temp['temperature'].max():.1f} °C")
        with col3:
            st.metric("Minimum Temperature", f"{df_temp['temperature'].min():.1f} °C")

        st.markdown("<br>", unsafe_allow_html=True)

        # Plotly Main Chart
        fig_temp = px.area(
            df_temp,
            x="timestamp",
            y="temperature",
            title="Hourly Temperature Forecast",
            labels={
                "temperature": "Temperature (°C)",
                "timestamp": "Time"
            }
        )
        
        fig_temp.update_layout(
            height=500,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_temp, use_container_width=True)

        # Secondary bar chart for temperature distribution
        st.subheader("Temperature Progression Profile")
        fig_temp_bar = px.bar(
            df_temp,
            x="timestamp",
            y="temperature"
        )
        fig_temp_bar.update_layout(height=350)
        st.plotly_chart(fig_temp_bar, use_container_width=True)

        # Raw Data Section
        st.markdown("---")
        with st.expander("View Raw Temperature Data Table"):
            st.dataframe(df_temp, use_container_width=True, hide_index=True)
    else:
        st.warning("No weather records returned for the selected location/station. Please check the inputs or try another location.")