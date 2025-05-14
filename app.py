import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import json

st.set_page_config(page_title="Algalteria Galactic Exchange (AGE)", layout="wide")

if "running" not in st.session_state:
    if os.path.exists("market_state.json"):
        with open("market_state.json", "r") as f:
            saved_state = json.load(f)
            st.session_state.running = saved_state.get("running", True)
    else:
        st.session_state.running = True
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

# Title and status
st.title("🌌 Algalteria Galactic Exchange (AGE)")
st.subheader(f"📈 Market Status: {'🟢 RUNNING' if st.session_state.running else '🔴 PAUSED'}")

# Pause/Resume toggle
if st.button("⏯ Pause / Resume Market"):
    st.session_state.running = not st.session_state.running

# Update logic
def update_prices():
    df = st.session_state.stocks.copy()
    df["Price"] = df["Price"] * (1 + np.random.uniform(-df["Volatility"], df["Volatility"]))
    st.session_state.stocks = df

    timestamp = datetime.now().strftime("%H:%M:%S")
    for i, row in df.iterrows():
        st.session_state.price_history.loc[len(st.session_state.price_history)] = {
            "Timestamp": timestamp,
            "Ticker": row["Ticker"],
            "Price": row["Price"]
        }

# Update if running
if "last_update_time" not in st.session_state:
    st.session_state.last_update_time = time.time()

# Update prices only if market is running and at least 10s passed
current_time = time.time()
if st.session_state.running and current_time - st.session_state.last_update_time >= 10:
    update_prices()
    st.session_state.last_update_time = current_time


# Display market data
st.dataframe(
    st.session_state.stocks.style.format({"Price": "{:.2f}"}),
    use_container_width=True
)

# Chart of historical prices
st.markdown("### 📊 Price History")
for ticker in st.session_state.stocks["Ticker"]:
    history = st.session_state.price_history[st.session_state.price_history["Ticker"] == ticker]
    if not history.empty:
        st.line_chart(data=history.set_index("Timestamp")[["Price"]], height=200, use_container_width=True)

from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 10 seconds if running
if st.session_state.running:
    st_autorefresh(interval=10 * 1000, key="market_refresh")
    
# Save running state
with open("market_state.json", "w") as f:
    json.dump({"running": st.session_state.running}, f)
