import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import json
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Algalteria Galactic Exchange (AGE)", layout="wide")

# Safely load persisted running state
def load_market_state():
    try:
        if os.path.exists("market_state.json"):
            with open("market_state.json", "r") as f:
                saved = json.load(f)
                return saved.get("running", True)
    except Exception:
        st.warning("⚠️ Failed to read market_state.json — using default (RUNNING).")
    return True  # fallback

# Only initialize once
if "running" not in st.session_state:
    st.session_state.running = load_market_state()


# Initial stock setup
tickers = ["DTF", "GMG", "USF", "TTT", "GFU", "IWI", "EE"]
names = [
    "Directorate Tech Fund", 
    "Galactic Mining Guild", 
    "Universal Services Fund", 
    "The Textile Team", 
    "Galactic Farmers Union", 
    "Imperial Weapons Industry", 
    "Epsilon Exchange"
]
initial_prices = [105.0, 95.0, 87.5, 76.0, 82.0, 132.0, 151.0]
volatility = [0.04, 0.035, 0.015, 0.02, 0.025, 0.03, 0.06]

# Initialize state
if "stocks" not in st.session_state:
    df = pd.DataFrame({
        "Ticker": tickers,
        "Name": names,
        "Price": initial_prices,
        "Volatility": volatility
    })
    st.session_state.stocks = df

if "initial_prices" not in st.session_state:
    st.session_state.initial_prices = dict(zip(tickers, initial_prices))

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

    # Update each asset except TMF
    for idx, row in df.iterrows():
        if row["Ticker"] == "TMF":
            continue
        change = np.random.uniform(-row["Volatility"], row["Volatility"])
        df.at[idx, "Price"] *= (1 + change)

        # Log to history
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.price_history.loc[len(st.session_state.price_history)] = {
            "Timestamp": timestamp,
            "Ticker": row["Ticker"],
            "Price": df.at[idx, "Price"]
        }

    # Recalculate TMF
    non_tmf_prices = df[df["Ticker"] != "TMF"]["Price"]
    tmf_price = non_tmf_prices.mean()
    if "TMF" in df["Ticker"].values:
        df.loc[df["Ticker"] == "TMF", "Price"] = tmf_price
    else:
        df.loc[len(df)] = {
            "Ticker": "TMF",
            "Name": "Total Market Fund",
            "Price": tmf_price,
            "Volatility": 0.0
        }
        st.session_state.initial_prices["TMF"] = tmf_price

    st.session_state.stocks = df

# Autorefresh every 10s
count = st_autorefresh(interval=10 * 1000, key="market_heartbeat")
if "last_refresh_count" not in st.session_state:
    st.session_state.last_refresh_count = -1

if st.session_state.running and count != st.session_state.last_refresh_count:
    update_prices()
    st.session_state.last_update_time = time.time()
    st.session_state.last_refresh_count = count

# Time display
if "last_update_time" in st.session_state:
    time_since = int(time.time() - st.session_state.last_update_time)
    next_tick = max(0, 10 - time_since)
    st.caption(f"⏱ Last update: {time_since}s ago — Next in: {next_tick}s")
else:
    st.caption("⏱ Market has not updated yet.")

# Compute all-time $ and % change
df = st.session_state.stocks.copy()
df["$ Change"] = df["Price"] - df["Ticker"].map(st.session_state.initial_prices)
df["% Change"] = (df["$ Change"] / df["Ticker"].map(st.session_state.initial_prices)) * 100
styled_df = df[["Ticker", "Name", "Price", "$ Change", "% Change"]]

# Show table
st.dataframe(
    styled_df.style.format({
        "Price": "{:.2f}", 
        "$ Change": "{:+.2f}", 
        "% Change": "{:+.2f}%"
    }),
    use_container_width=True
)

# Select a stock to view its price chart
st.markdown("### 📊 Select a stock to view price history")
selected_ticker = st.selectbox("Choose a stock", df["Ticker"])

if selected_ticker:
    import altair as alt

    history = st.session_state.price_history[
        st.session_state.price_history["Ticker"] == selected_ticker
    ].copy()

    if not history.empty:
        # Convert Timestamp to full datetime object
        today = datetime.today().date()
        history["Datetime"] = pd.to_datetime(today.strftime("%Y-%m-%d") + " " + history["Timestamp"])

        chart = alt.Chart(history).mark_line(point=False).encode(
            x=alt.X("Datetime:T", axis=alt.Axis(title="Time", format="%H:%M", labelAngle=0)),
            y=alt.Y("Price:Q", axis=alt.Axis(title="Price (cr)")),
            tooltip=["Datetime:T", "Price:Q"]
        ).properties(
            title=f"{selected_ticker} Price History",
            width="container",
            height=300
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No price history available yet.")
