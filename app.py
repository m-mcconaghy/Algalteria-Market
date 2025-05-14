import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time

st.set_page_config(page_title="Starfinder Stock Exchange", layout="wide")

# Initialize session state
if "running" not in st.session_state:
    st.session_state.running = True
if "stocks" not in st.session_state:
    st.session_state.stocks = pd.DataFrame({
        "Ticker": ["ABM", "HZN", "SYL"],
        "Name": ["Abadar Mining", "Horizon Energy", "Sylex Arms"],
        "Price": [100.0, 120.0, 80.0],
        "Volatility": [0.02, 0.03, 0.05]
    })
if "price_history" not in st.session_state:
    st.session_state.price_history = pd.DataFrame(columns=["Timestamp", "Ticker", "Price"])

# Title and status
st.title("🌌 Starfinder Interstellar Stock Exchange")
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
if st.session_state.running:
    update_prices()

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

# Auto-refresh every 10 seconds
st.experimental_rerun() if st.session_state.running else time.sleep(10)
