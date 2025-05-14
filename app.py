import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import json
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Algalteria Galactic Exchange (AGE)", layout="wide")

# Load persisted state from file (pause/resume status)
loaded_running_state = True  # fallback default
try:
    if os.path.exists("market_state.json"):
        with open("market_state.json", "r") as f:
            loaded_state = json.load(f)
            loaded_running_state = loaded_state.get("running", True)
except Exception as e:
    st.warning("⚠️ Corrupted market_state.json file. Using default (RUNNING).")

# Only initialize once
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

# Initialize price history
if "price_history" not in st.session_state:
    st.session_state.price_history = pd.DataFrame(columns=["Timestamp", "Ticker", "Price"])

# Title and market status
st.title("🌌 Algalteria Galactic Exchange (AGE)")
st.subheader(f"📈 Market Status: {'🟢 RUNNING' if st.session_state.running else '🔴 PAUSED'}")

# Pause/Resume toggle
if st.button("⏯ Pause / Resume Market"):
    st.session_state.running = not st.session_state.running
    with open("market_state.json", "w") as f:
        json.dump({"running": st.session_state.running}, f)

# Update prices function
def update_prices():
    df = st.session_state.stocks.copy()
    df["Price"] = df["Price"] * (1 + np.random.uniform(-df["Volatility"], df["Volatility"]))
    st.session_state.stocks = df

    timestamp = datetime.now().strftime("%H:%M:%S")
    for _, row in df.iterrows():
        st.session_state.price_history.loc[len(st.session_state.price_history)] = {
            "Timestamp": timestamp,
            "Ticker": row["Ticker"],
            "Price": row["Price"]
        }

# Autorefresh every 10 seconds
count = st_autorefresh(interval=1 * 1000, key="market_heartbeat")

# Track last update tick
if "last_refresh_count" not in st.session_state:
    st.session_state.last_refresh_count = -1

# Only update once per refresh tick
if st.session_state.running and count != st.session_state.last_refresh_count:
    update_prices()
    st.session_state.last_update_time = time.time()
    st.session_state.last_refresh_count = count

# Run update only if running
if st.session_state.running:
    update_prices()
    st.session_state.last_update_time = time.time()

# Show time since last update
if "last_update_time" in st.session_state:
    time_since = int(time.time() - st.session_state.last_update_time)
    next_tick = max(0, 10 - time_since)
    st.caption(f"⏱ Last update: {time_since}s ago — Next in: {next_tick}s")
else:
    st.caption("⏱ Market has not updated yet.")

# Show market data
st.dataframe(
    st.session_state.stocks.style.format({"Price": "{:.2f}"}),
    use_container_width=True
)

# Chart historical prices
st.markdown("### 📊 Price History")
for ticker in st.session_state.stocks["Ticker"]:
    history = st.session_state.price_history[st.session_state.price_history["Ticker"] == ticker]
    if not history.empty:
        st.line_chart(data=history.set_index("Timestamp")[["Price"]], height=200, use_container_width=True)
