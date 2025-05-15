import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import time
from streamlit_autorefresh import st_autorefresh
import altair as alt

st.set_page_config(page_title="Algalteria Galactic Exchange (AGE)", layout="wide")

# Connect to SQLite
conn = sqlite3.connect("market.db", check_same_thread=False)
cursor = conn.cursor()

# Ensure tables exist with correct schema
cursor.execute("""
CREATE TABLE IF NOT EXISTS stocks (
    Ticker TEXT PRIMARY KEY,
    Name TEXT,
    Price REAL,
    Volatility REAL,
    InitialPrice REAL
)
""")
cursor.execute("CREATE TABLE IF NOT EXISTS price_history (Timestamp TEXT, Ticker TEXT, Price REAL)")
cursor.execute("CREATE TABLE IF NOT EXISTS market_status (key TEXT PRIMARY KEY, value TEXT)")
conn.commit()

# Admin authentication
admin_password = st.secrets.get("ADMIN_PASSWORD", "secret123")
is_admin = st.text_input("Enter admin password", type="password") == admin_password

# Load market status from DB
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

# Initialize base tickers
tickers = ["DTF", "GMG", "USF", "TTT", "GFU", "IWI", "EE"]
names = [
    "Directorate Tech Fund", "Galactic Mining Guild", "Universal Services Fund",
    "The Textile Team", "Galactic Farmers Union", "Imperial Weapons Industry", "Epsilon Exchange"
]
initial_prices = [105.0, 95.0, 87.5, 76.0, 82.0, 132.0, 151.0]
volatility = [0.04, 0.035, 0.015, 0.02, 0.025, 0.03, 0.06]

# Repopulate market (safe insert)
cursor.execute("SELECT COUNT(*) FROM stocks")
row_count = cursor.fetchone()[0]
if row_count == 0:
    for i in range(len(tickers)):
        cursor.execute("""
            INSERT OR REPLACE INTO stocks (Ticker, Name, Price, Volatility, InitialPrice)
            VALUES (?, ?, ?, ?, ?)
        """, (tickers[i], names[i], initial_prices[i], volatility[i], initial_prices[i]))
    # Add TMF
    tmf_price = sum(initial_prices) / len(initial_prices)
    cursor.execute("""
        INSERT OR REPLACE INTO stocks (Ticker, Name, Price, Volatility, InitialPrice)
        VALUES (?, ?, ?, ?, ?)
    """, ("TMF", "Total Market Fund", tmf_price, 0.0, tmf_price))
    conn.commit()

# Header
st.title("🌌 Algalteria Galactic Exchange (AGE)")

if is_admin:
    st.success("🧑‍🚀 Admin mode enabled")
    if st.button("⏯ Pause / Resume Market"):
        st.session_state.running = not st.session_state.running
        save_market_status(st.session_state.running)
else:
    st.info("🛰 Viewer mode — live market feed only")

st.subheader(f"📈 Market Status: {'🟢 RUNNING' if st.session_state.running else '🔴 PAUSED'}")

# Price updater
def update_prices():
    df = pd.read_sql("SELECT * FROM stocks", conn)

    for idx, row in df.iterrows():
        if row["Ticker"] == "TMF":
            continue
        change = np.random.uniform(-row["Volatility"], row["Volatility"])
        new_price = row["Price"] * (1 + change)
        df.at[idx, "Price"] = new_price

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO price_history (Timestamp, Ticker, Price) VALUES (?, ?, ?)",
            (timestamp, row["Ticker"], new_price)
        )

    tmf_price = df[df["Ticker"] != "TMF"]["Price"].mean()
    df.loc[df["Ticker"] == "TMF", "Price"] = tmf_price

    for idx, row in df.iterrows():
        cursor.execute(
            "UPDATE stocks SET Price = ? WHERE Ticker = ?",
            (row["Price"], row["Ticker"])
        )
    conn.commit()

# Auto-refresh every 10s (admin-only updates)
count = st_autorefresh(interval=10 * 1000, key="market_tick")
if "last_refresh_count" not in st.session_state:
    st.session_state.last_refresh_count = -1

if is_admin and st.session_state.running and count != st.session_state.last_refresh_count:
    update_prices()
    st.session_state.last_update_time = time.time()
    st.session_state.last_refresh_count = count

# Time since last update
if "last_update_time" in st.session_state:
    time_since = int(time.time() - st.session_state.last_update_time)
    next_tick = max(0, 10 - time_since)
    st.caption(f"⏱ Last update: {time_since}s ago — Next in: {next_tick}s")
else:
    st.caption("⏱ Market has not updated yet.")

# Show current stock data
df = pd.read_sql("SELECT * FROM stocks", conn)
df["$ Change"] = df["Price"] - df["InitialPrice"]
df["% Change"] = (df["$ Change"] / df["InitialPrice"]) * 100
styled_df = df[["Ticker", "Name", "Price", "$ Change", "% Change"]]

st.dataframe(
    styled_df.style.format({
        "Price": "{:.2f}",
        "$ Change": "{:+.2f}",
        "% Change": "{:+.2f}%"
    }),
    use_container_width=True
)

# Stock price chart
st.markdown("### 📊 Select a stock to view price history")
selected_ticker = st.selectbox("Choose a stock", df["Ticker"])

if selected_ticker:
    history = pd.read_sql(
        "SELECT * FROM price_history WHERE Ticker = ? ORDER BY Timestamp",
        conn,
        params=(selected_ticker,)
    )

    if not history.empty:
        history["Datetime"] = pd.to_datetime(history["Timestamp"])
        chart = alt.Chart(history).mark_line().encode(
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
