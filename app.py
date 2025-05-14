import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import json
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Algalteria Galactic Exchange (AGE)", layout="wide")

# Load persisted pause/resume state
loaded_running_state = True
try:
    if os.path.exists("market_state.json"):
        with open("market_state.json", "r") as f:
            loaded_state = json.load(f)
            loaded_running_state = loaded_state.get("running", True)
except Exception:
    st.warning("⚠️ Corrupted market_state.json file. Using default (RUNNING).")

if "running" not in st.session_state:
    st.session_state.running = loaded_running_state

# Initialize stocks
if "stocks" not in st.session_state:
    st.session_state.stocks = pd.DataFrame({
        "Ticker": ["DTF", "GMG", "USF", "TTT", "GFU", "IWI", "EE", "TMF"],
        "Name": [
            "Directorate Tech Fund", 
            "Galactic Mining Guild", 
            "Universal Services Fund", 
            "The Textile Team", 
            "Galactic Farmers Union", 
            "Imperial Weapons Industry", 
            "Epsilon Exchange", 
            "Total Market Fund"
        ],
        "Price": [105.0, 95.0, 87.5, 76.0, 82.0, 132.0, 151.0, 100.0],
        "Volatility": [0.04, 0.035, 0.015, 0.02, 0.025, 0.03, 0.06, 0.018]
    })

if "price_history" not in st.session_state:
    st.session_state.price_history = pd.DataFrame(columns=["Timestamp", "Ticker", "Price"])

# Title and control
st.title("🌌 Algalteria Galactic Exchange (AGE)")
st.subheader(f"📈 Market Status: {'🟢 RUNNING' if st.session_state.running else '🔴 PAUSED'}")

if st.button("⏯ Pause / Resume Market"):
    st.session_state.running = not st.session_state.running
    with open("market_state.json", "w") as f:
        json.dump({"running": st.session_state.running}, f)

# Price updater
def update_prices():
    df = st.session_state.stocks.copy()
    prev_prices = df["Price"].copy()
    df["Price"] = df["Price"] * (1 + np.random.uniform(-df["Volatility"], df["Volatility"]))
    df["$ Change"] = df["Price"] - prev_prices
    df["% Change"] = ((df["$ Change"] / prev_prices) * 100).round(2)
    st.session_state.stocks = df

    timestamp = datetime.now().strftime("%H:%M:%S")
    for _, row in df.iterrows():
        st.session_state.price_history.loc[len(st.session_state.price_history)] = {
            "Timestamp": timestamp,
            "Ticker": row["Ticker"],
            "Price": row["Price"]
        }

# Autorefresh every 10s
count = st_autorefresh(interval=10 * 1000, key="market_heartbeat")

# Only update prices once per refresh tick
if "last_refresh_count" not in st.session_state:
    st.session_state.last_refresh_count = -1

if st.session_state.running and count != st.session_state.last_refresh_count:
    update_prices()
    st.session_state.last_update_time = time.time()
    st.session_state.last_refresh_count = count

# Show update timing
if "last_update_time" in st.session_state:
    time_since = int(time.time() - st.session_state.last_update_time)
    next_tick = max(0, 10 - time_since)
    st.caption(f"⏱ Last update: {time_since}s ago — Next in: {next_tick}s")
else:
    st.caption("⏱ Market has not updated yet.")

# Show stock table with % and $ change
styled_df = st.session_state.stocks[["Ticker", "Name", "Price", "$ Change", "% Change"]].copy()
st.dataframe(
    styled_df.style.format({"Price": "{:.2f}", "$ Change": "{:+.2f}", "% Change": "{:+.2f}%"}),
    use_container_width=True
)

# Select a stock to show its chart
st.markdown("### 📊 Select a stock to view price history")
selected_ticker = st.selectbox("Choose a stock", st.session_state.stocks["Ticker"])

if selected_ticker:
    history = st.session_state.price_history[
        st.session_state.price_history["Ticker"] == selected_ticker
    ]
    if not history.empty:
        st.line_chart(history.set_index("Timestamp")[["Price"]], height=250, use_container_width=True)
