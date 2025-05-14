
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Starfinder Stock Exchange", layout="wide")

# Initialize ticker data
if "running" not in st.session_state:
    st.session_state.running = True
if "stocks" not in st.session_state:
    st.session_state.stocks = pd.DataFrame({
        "Ticker": ["ABM", "HZN", "SYL"],
        "Name": ["Abadar Mining", "Horizon Energy", "Sylex Arms"],
        "Price": [100.0, 120.0, 80.0],
        "Volatility": [0.02, 0.03, 0.05]
    })

st.title("🌌 Starfinder Interstellar Stock Exchange")

# Pause/Resume toggle
if st.button("⏯ Pause / Resume Market"):
    st.session_state.running = not st.session_state.running

# Update prices
if st.session_state.running:
    df = st.session_state.stocks.copy()
    df["Price"] = df["Price"] * (1 + np.random.uniform(-df["Volatility"], df["Volatility"]))
    st.session_state.stocks = df

# Display market data
st.dataframe(
    st.session_state.stocks.style.format({"Price": "{:.2f}"}),
    use_container_width=True
)

# Note
st.caption("Prices update each time the app reloads. Future versions will include real-time timers and trading UI.")
