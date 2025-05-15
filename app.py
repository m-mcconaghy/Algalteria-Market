import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import time
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Algalteria Galactic Exchange (AGE)", layout="wide")

# Connect to SQLite
conn = sqlite3.connect("market.db", check_same_thread=False)
cursor = conn.cursor()

# Ensure required tables exist
cursor.execute("CREATE TABLE IF NOT EXISTS market_status (key TEXT PRIMARY KEY, value TEXT)")
cursor.execute("CREATE TABLE IF NOT EXISTS price_history (Timestamp TEXT, Ticker TEXT, Price REAL)")
cursor.execute("CREATE TABLE IF NOT EXISTS stocks (Ticker TEXT PRIMARY KEY, Name TEXT, Price REAL, Volatility REAL)")
conn.commit()

# Load persisted market status
def load_market_status():
    try:
        cursor.execute("SELECT value FROM market_status WHERE key='running'")
        row = cursor.fetchone()
        return row and row[0] == "True"
    except:
        return True

def save_market_status(running):
    cursor.execute("REPLACE INTO market_status (key, value) VALUES (?, ?)", ("running", str(running)))
    conn.commit()

if "running" not in st.session_state:
    st.session_state.running = load_market_status()

# Stock definitions
tickers = ["DTF", "GMG", "USF", "TTT", "GFU", "IWI", "EE"]
names = [
    "Directorate Tech Fund", "Galactic Mining Guild", "Universal Services Fund",
    "The Textile Team", "Galactic Farmers Union", "Imperial Weapons Industry", "Epsilon Exchange"
]
initial_prices = [105.0, 95.0, 87.5, 76.0, 82.0, 132.0, 151.0]
volatility = [0.04, 0.035, 0.015, 0.02, 0.025, 0.03, 0.06]

# Load or create stock data
if "stocks" not in st.session_state:
    cursor.execute("SELECT * FROM stocks")
    rows = cursor.fetchall()
    if rows:
        df = pd.DataFrame(rows, columns=["Ticker", "Name", "Price", "Volatility"])
    else:
        df = pd.DataFrame({
            "Ticker": tickers,
            "Name": names,
            "Price": initial_prices,
            "Volatility": volatility
        })
        df.to_sql("stocks", conn, if_exists="replace", index=False)
    st.session_state.stocks = df
    st.session_state.initial_prices = dict(zip(df["Ticker"], df["Price"]))

# Load price history
if "price_history" not in st.session_state:
    try:
        st.session_state.price_history = pd.read_sql("SELECT * FROM price_history", conn)
    except:
        st.session_state.price_history = pd.DataFrame(columns=["Timestamp", "Ticker", "Price"])

# UI: Title and toggle
st.title("🌌 Algalteria Galactic Exchange (AGE)")
st.subheader(f"📈 Market Status: {'🟢 RUNNING' if st.session_state.running else '🔴 PAUSED'}")

if st.button("⏯ Pause / Resume Market"):
    st.session_state.running = not st.session_state.running
    save_market_status(st.session_state.running)

# Price update logic
def update_prices():
    df = st.session_state.stocks.copy()

    for idx, row in df.iterrows():
        if row["Ticker"] == "TMF":
            continue
        change = np.random.uniform(-row["Volatility"], row["Volatility"])
        df.at[idx, "Price"] *= (1 + change)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.price_history.loc[len(st.session_state.price_history)] = {
            "Timestamp": timestamp,
            "Ticker": row["Ticker"],
            "Price": df.at[idx, "Price"]
        }

    # Recalculate TMF as average
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
    df.to_sql("stocks", conn, if_exists="replace", index=False)
    st.session_state.price_history.to_sql("price_history", conn, if_exists="replace", index=False)

# Auto-refresh every 10 seconds
count = st_autorefresh(interval=10 * 1000, key="market_heartbeat")
if "last_refresh_count" not in st.session_state:
    st.session_state.last_refresh_count = -1

if st.session_state.running and count != st.session_state.last_refresh_count:
    update_prices()
    st.session_state.last_update_time = time.time()
    st.session_state.last_refresh_count = count

# Show time status
if "last_update_time" in st.session_state:
    time_since = int(time.time() - st.session_state.last_update_time)
    next_tick = max(0, 10 - time_since)
    st.caption(f"⏱ Last update: {time_since}s ago — Next in: {next_tick}s")
else:
    st.caption("⏱ Market has not updated yet.")

# Show price table with change
df = st.session_state.stocks.copy()
df["$ Change"] = df["Price"] - df["Ticker"].map(st.session_state.initial_prices)
df["% Change"] = (df["$ Change"] / df["Ticker"].map(st.session_state.initial_prices)) * 100
styled_df = df[["Ticker", "Name", "Price", "$ Change", "% Change"]]

st.dataframe(
    styled_df.style.format({
        "Price": "{:.2f}", 
        "$ Change": "{:+.2f}", 
        "% Change": "{:+.2f}%"
    }),
    use_container_width=True
)

# Interactive chart
st.markdown("### 📊 Select a stock to view price history")
selected_ticker = st.selectbox("Choose a stock", df["Ticker"])

if selected_ticker:
    import altair as alt

    history = st.session_state.price_history[
        st.session_state.price_history["Ticker"] == selected_ticker
    ].copy()

    if not history.empty:
        history["Datetime"] = pd.to_datetime(history["Timestamp"])
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
