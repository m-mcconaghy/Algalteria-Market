import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import time
from streamlit_autorefresh import st_autorefresh
import altair as alt

st.set_page_config(page_title="Algalteria Galactic Exchange (AGE)", layout="wide")

conn = sqlite3.connect("market.db", check_same_thread=False)
cursor = conn.cursor()

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

admin_password = st.secrets.get("ADMIN_PASSWORD", "secret123")
is_admin = st.text_input("Enter admin password", type="password") == admin_password

# Market state
if "running" not in st.session_state:
    try:
        cursor.execute("SELECT value FROM market_status WHERE key='running'")
        row = cursor.fetchone()
        st.session_state.running = row and row[0] == "True"
    except:
        st.session_state.running = True

if "sim_time" not in st.session_state:
    st.session_state.sim_time = 0

if "tick_interval_sec" not in st.session_state:
    st.session_state.tick_interval_sec = 60

if "risk_free_rate" not in st.session_state:
    st.session_state.risk_free_rate = 0.075

if "market_conditions" not in st.session_state:
    st.session_state.market_conditions = "Normal"

# Initial setup
base_tickers = ["DTF", "GMG", "USF", "TTT", "GFU", "IWI", "EE"]
names = [
    "Directorate Tech Fund", "Galactic Mining Guild", "Universal Services Fund",
    "The Textile Team", "Galactic Farmers Union", "Imperial Weapons Industry", "Epsilon Exchange"
]
initial_prices = [105.0, 95.0, 87.5, 76.0, 82.0, 132.0, 151.0]
volatility = [0.04, 0.035, 0.015, 0.02, 0.025, 0.03, 0.06]

cursor.execute("SELECT COUNT(*) FROM stocks")
if cursor.fetchone()[0] == 0:
    for i in range(len(base_tickers)):
        cursor.execute("""
            INSERT INTO stocks (Ticker, Name, Price, Volatility, InitialPrice)
            VALUES (?, ?, ?, ?, ?)
        """, (base_tickers[i], names[i], initial_prices[i], volatility[i], initial_prices[i]))
    tmf_price = sum(initial_prices) / len(initial_prices)
    cursor.execute("""
        INSERT INTO stocks (Ticker, Name, Price, Volatility, InitialPrice)
        VALUES (?, ?, ?, ?, ?)
    """, ("TMF", "Total Market Fund", tmf_price, 0.0, tmf_price))
    conn.commit()

st.title("\U0001F30C Algalteria Galactic Exchange (AGE)")

if is_admin:
    st.success("\U0001F9D1‍\U0001F680 Admin mode enabled")
    if st.button("⏯ Pause / Resume Market"):
        st.session_state.running = not st.session_state.running
        cursor.execute("REPLACE INTO market_status (key, value) VALUES (?, ?)", ("running", str(st.session_state.running)))
        conn.commit()
else:
    st.info("\U0001F6F8 Viewer mode — live market feed only")

st.subheader(f"\U0001F4C8 Market Status: {'🟢 RUNNING' if st.session_state.running else '🔴 PAUSED'}")

# Price update function
def update_prices():
    df = pd.read_sql("SELECT * FROM stocks", conn)
    for idx, row in df.iterrows():
        if row["Ticker"] == "TMF":
            continue
        theta = 0.04
        reversion = theta * (row["InitialPrice"] - row["Price"])
        momentum = np.random.choice([1, -1], p=[0.52, 0.48])
        regime_multiplier = np.random.choice([1, 2.5], p=[0.85, 0.15])
        scaled_vol = row["Volatility"] * np.sqrt(1 / 360)
        noise = np.random.normal(0, scaled_vol * regime_multiplier) * momentum
        drift = (st.session_state.risk_free_rate / 360) * row["Price"]
        new_price = max(row["Price"] + reversion + noise * row["Price"] + drift, 0.01)
        df.at[idx, "Price"] = new_price
        cursor.execute("INSERT INTO price_history (Timestamp, Ticker, Price) VALUES (?, ?, ?)",
                       (str(st.session_state.sim_time), row["Ticker"], new_price))
    df.loc[df["Ticker"] == "TMF", "Price"] = df[df["Ticker"] != "TMF"]["Price"].mean()
    for _, row in df.iterrows():
        cursor.execute("UPDATE stocks SET Price = ? WHERE Ticker = ?", (row["Price"], row["Ticker"]))
    conn.commit()
    st.session_state.sim_time += 1

# Admin controls
if is_admin:
    with st.expander("⚙️ Admin Tools"):
        ticker_to_change = st.selectbox("Select a stock to modify", base_tickers + ["TMF"])
        price_change = st.number_input("New price", min_value=0.01)
        if st.button("Apply Price Change"):
            cursor.execute("UPDATE stocks SET Price = ? WHERE Ticker = ?", (price_change, ticker_to_change))
            cursor.execute("INSERT INTO price_history (Timestamp, Ticker, Price) VALUES (?, ?, ?)",
                           (str(st.session_state.sim_time), ticker_to_change, price_change))
            conn.commit()
            st.success(f"Updated {ticker_to_change} to {price_change:.2f} credits.")
        st.divider()
        st.markdown("#### Advance Simulation")
        if st.button("Advance 1 Hour"):
            for _ in range(60): update_prices()
        if st.button("Advance 1 Day"):
            for _ in range(360): update_prices()
        if st.button("Advance 1 Week"):
            for _ in range(2520): update_prices()
        if st.button("Advance 1 Month"):
            for _ in range(7200): update_prices()
        if st.button("Advance 1 Year"):
            for _ in range(86400): update_prices()
        st.divider()
        st.markdown("#### Stock-Specific Volatility")
        tickers = pd.read_sql("SELECT Ticker FROM stocks", conn)["Ticker"].tolist()
        selected_vol_ticker = st.selectbox("Select a stock to update volatility", tickers)
        current_vol = pd.read_sql("SELECT Volatility FROM stocks WHERE Ticker = ?", conn, params=(selected_vol_ticker,)).iloc[0, 0]
        new_vol = st.number_input("New Volatility", value=current_vol, key=f"vol_{selected_vol_ticker}")
        if st.button("Apply Volatility Change"):
            cursor.execute("UPDATE stocks SET Volatility = ? WHERE Ticker = ?", (new_vol, selected_vol_ticker))
            conn.commit()
            st.success(f"Updated volatility of {selected_vol_ticker} to {new_vol:.3f}")
        st.divider()
        st.markdown("#### Risk-Free Rate")
        new_rfr = st.number_input("Annual Risk-Free Rate", value=st.session_state.risk_free_rate, step=0.0001, format="%.4f")
        st.session_state.risk_free_rate = new_rfr
        tick_rate = st.slider("Tick interval (seconds)", 10, 300, st.session_state.tick_interval_sec, step=10)
        st.session_state.tick_interval_sec = tick_rate

count = st_autorefresh(interval=st.session_state.tick_interval_sec * 1000, key="market_tick")
if "last_refresh_count" not in st.session_state:
    st.session_state.last_refresh_count = -1
if is_admin and st.session_state.running and count != st.session_state.last_refresh_count:
    update_prices()
    st.session_state.last_update_time = time.time()
    st.session_state.last_refresh_count = count

if "last_update_time" in st.session_state:
    elapsed = int(time.time() - st.session_state.last_update_time)
    st.caption(f"⏱ Last update: {elapsed}s ago — Next in: {max(0, st.session_state.tick_interval_sec - elapsed)}s")

stocks_df = pd.read_sql("SELECT * FROM stocks", conn)
stocks_df["$ Change"] = stocks_df["Price"] - stocks_df["InitialPrice"]
stocks_df["% Change"] = (stocks_df["$ Change"] / stocks_df["InitialPrice"]) * 100
st.dataframe(
    stocks_df[["Ticker", "Name", "Price", "Volatility", "$ Change", "% Change"]].style.format({
        "Price": "{:.2f}", "Volatility": "{:.3f}", "$ Change": "+{:.2f}", "% Change": "+{:.2f}%"
    }),
    use_container_width=True
)

st.markdown("### 📊 Select a stock to view price history")
selected_ticker = st.selectbox("Choose a stock", stocks_df["Ticker"])
if selected_ticker:
    hist = pd.read_sql("SELECT * FROM price_history WHERE Ticker = ? ORDER BY CAST(Timestamp AS INTEGER)", conn, params=(selected_ticker,))
    if not hist.empty:
        hist["SimTime"] = pd.to_numeric(hist["Timestamp"], errors="coerce").astype(int)
        low, high = hist["Price"].min(), hist["Price"].max()
        padding = (high - low) * 0.1
        chart = alt.Chart(hist).mark_line().encode(
            x=alt.X("SimTime:Q", axis=alt.Axis(title="Simulated Time (ticks)")),
            y=alt.Y("Price:Q", scale=alt.Scale(domain=[low - padding, high + padding]),
                   axis=alt.Axis(title="Price (cr)", grid=True)),
            tooltip=["SimTime", "Price"]
        ).properties(title=f"{selected_ticker} Price History", width="container", height=300).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No price history available yet.")
