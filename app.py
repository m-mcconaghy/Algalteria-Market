import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import time
from streamlit_autorefresh import st_autorefresh
import altair as alt

TICKS_PER_DAY = 3  # Used for faster simulation during Advance mode

st.set_page_config(page_title="Algalteria Galactic Exchange (AGE)", layout="wide")

# --- Database Connection ---
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

# --- Admin Password and Mode ---
admin_password = st.secrets.get("ADMIN_PASSWORD", "secret123")
is_admin = st.text_input("Enter admin password", type="password") == admin_password

# --- Market State Initialization ---
if "running" not in st.session_state:
    try:
        cursor.execute("SELECT value FROM market_status WHERE key='running'")
        row = cursor.fetchone()
        st.session_state.running = row and row[0] == "True"
    except Exception as e:
        st.error(f"Error loading market status: {e}")
        st.session_state.running = True

if "sim_time" not in st.session_state:
    st.session_state.sim_time = 0

if "tick_interval_sec" not in st.session_state:
    st.session_state.tick_interval_sec = 150  # 2.5 minutes

if "risk_free_rate" not in st.session_state:
    st.session_state.risk_free_rate = 0.075

if "equity_risk_premium" not in st.session_state:
    st.session_state.equity_risk_premium = 0.02

if "market_conditions" not in st.session_state:
    st.session_state.market_conditions = "Normal"

if "market_sentiment" not in st.session_state:
    st.session_state.market_sentiment = "Booming"

# --- Initial Stock Data ---
base_tickers = ["DTF", "GMG", "USF", "TTT", "GFU", "IWI", "EE", "NEC", "ARC", "SOL", "AWE", "ORB", "QNT", "AGX", "LCO", "FMC", "SYX", "VLT", "EXR", "CRB"]
names = ["Directorate Tech Fund", "Galactic Mining Guild", "Universal Services Fund", "The Textile Team", "Galactic Farmers Union", "Imperial Weapons Industry", "Epsilon Exchange", "Nebular Energy Consortium", "Asteroid Resources Collective", "Solar Operations League", "Algalterian Water Exchange", "Orbital Rare Biotech", "Quantum Nexus Trust", "Agricultural Exports Guild", "Lunar Construction Outfit", "Frontier Medical Consortium", "Syphonix Energy Systems", "Veltrax AI Logistics", "Exorium Rare Elements", "Crystalline Banking Network"]
initial_prices = [105.0, 95.0, 87.5, 76.0, 82.0, 132.0, 151.0, 91.0, 87.5, 102.0, 78.0, 113.0, 139.0, 84.0, 62.0, 144.0, 193.0, 119.0, 221.0, 68.0]
volatility = [0.04, 0.035, 0.015, 0.02, 0.025, 0.03, 0.06, 0.018, 0.025, 0.02, 0.015, 0.045, 0.03, 0.017, 0.023, 0.014, 0.055, 0.027, 0.06, 0.018]

# --- Initialize Stocks Table ---
cursor.execute("SELECT COUNT(*) FROM stocks")
if cursor.fetchone()[0] == 0:
    for i in range(len(base_tickers)):
        cursor.execute("""
            INSERT INTO stocks (Ticker, Name, Price, Volatility, InitialPrice)
            VALUES (?, ?, ?, ?, ?)
        """, (base_tickers[i], names[i], initial_prices[i], volatility[i], initial_prices[i]))
    tmf_price = np.average(initial_prices, weights=initial_prices)
    tmf_vol = np.average(volatility, weights=initial_prices)
    cursor.execute("""
        INSERT INTO stocks (Ticker, Name, Price, Volatility, InitialPrice)
        VALUES (?, ?, ?, ?, ?)
    """, ("TMF", "Total Market Fund", tmf_price, tmf_vol, tmf_price))
    conn.commit()

# --- Header and Market Status ---
st.title("\U0001F30C Algalteria Galactic Exchange (AGE)")

col_status, col_admin = st.columns([3, 1])
with col_status:
    st.subheader(f"\U0001F4C8 Market Status: {'<span style=\"color: green;\">🟢 RUNNING</span>' if st.session_state.running else '<span style=\"color: red;\">🔴 PAUSED</span>'}", unsafe_allow_html=True)
with col_admin:
    if is_admin:
        st.success("\U0001F9D1‍\U0001F680 Admin Mode")
        if st.button("⏯ Pause / Resume Market"):
            st.session_state.running = not st.session_state.running
            cursor.execute("REPLACE INTO market_status (key, value) VALUES (?, ?)", ("running", str(st.session_state.running)))
            conn.commit()
    else:
        st.info("\U0001F6F8 Viewer Mode — Live Market Feed Only")

# --- Set the TIME ---
SIM_START_DATE = pd.Timestamp("2200-01-01")

# --- Final update_prices function ---
def update_prices(ticks=1):
    for _ in range(ticks):
        df = pd.read_sql("SELECT * FROM stocks", conn)
        tick_scale = 24 / TICKS_PER_DAY  # how many hours this tick simulates

        for idx, row in df.iterrows():
            if row["Ticker"] == "TMF":
                continue

            # Scaled volatility for the current tick resolution
            regime_multiplier = np.random.choice([1, 1.5], p=[0.95, 0.05])
            scaled_vol = row["Volatility"] * np.sqrt(tick_scale / 24)
            momentum = np.random.choice([1, -1])
            noise = np.random.normal(0, scaled_vol * regime_multiplier) * momentum

            # Drift with scaled time
            financial_drift = st.session_state.risk_free_rate + st.session_state.equity_risk_premium
            sentiment_multiplier = {
                "Bubbling": 0.03,
                "Booming": 0.01,
                "Stagnant": 0.00,
                "Receding": -0.02,
                "Depression": -0.05
            }
            selected_sentiment = st.session_state.get("market_sentiment", "Booming")
            mult = sentiment_multiplier.get(selected_sentiment, 1.0)
            drift_rate = (financial_drift * mult) / 24
            drift = np.clip(drift_rate * tick_scale * row["Price"], -0.002 * row["Price"], 0.002 * row["Price"])

            # Optional: adjust shock chance for tick scale (roughly per day chance = 0.1%)
            daily_shock_chance = 0.001
            shock_chance = 1 - (1 - daily_shock_chance) ** (tick_scale / 24)
            shock_factor = 1.0
            if np.random.rand() < shock_chance:
                shock_factor = np.random.choice([0.95, 1.05], p=[0.5, 0.5])

            base_price = row["Price"] * shock_factor
            new_price = base_price + noise * base_price + drift
            limit = 0.01  # 1% per tick
            new_price = float(np.clip(new_price, row["Price"] * (1 - limit), row["Price"] * (1 + limit)))
            new_price = max(new_price, 0.01)

            # Update InitialPrice once per real day
            if st.session_state.sim_time % (24 / (24 / TICKS_PER_DAY)) == 0: # Update once per 24 simulation hours
                new_initial_price = row["InitialPrice"] * 1.00005 # Slightly reduced growth
                cursor.execute("UPDATE stocks SET InitialPrice = ? WHERE Ticker = ?", (new_initial_price, row["Ticker"]))

            df.at[idx, "Price"] = new_price
            sim_timestamp = SIM_START_DATE + timedelta(hours=(st.session_state.sim_time * (24 / TICKS_PER_DAY))) # Corrected timedelta
            cursor.execute("INSERT INTO price_history (Timestamp, Ticker, Price) VALUES (?, ?, ?)",
                           (sim_timestamp.isoformat(), row["Ticker"], new_price))

        # Update TMF based on weighted average (using volatility as weight)
        tmf_data = df[df["Ticker"] != "TMF"]
        tmf_price = np.average(tmf_data["Price"], weights=tmf_data["Volatility"])
        df.loc[df["Ticker"] == "TMF", "Price"] = tmf_price

        for _, row in df.iterrows():
            cursor.execute("UPDATE stocks SET Price = ? WHERE Ticker = ?", (row["Price"], row["Ticker"]))

        conn.commit()
        st.session_state.sim_time += 1

# --- Auto-refresh and Price Updates ---
count = st_autorefresh(interval=st.session_state.tick_interval_sec * 1000, key="market_tick")
if "last_refresh_count" not in st.session_state:
    st.session_state.last_refresh_count = -1
if is_admin and st.session_state.running and count != st.session_state.last_refresh_count:
    update_prices()
    st.session_state.last_update_time = time.time()
    st.session_state.last_refresh_count = count

if "last_update_time" in st.session_state:
    elapsed = int(time.time() - st.session_state.last_update_time)
    st.caption(f"⏱️ Last update: {elapsed}s ago — Next in: {max(0, st.session_state.tick_interval_sec - elapsed)}s")

# --- Display Stock Data ---
st.markdown("### 📈 Current Stock Prices")
stocks_df = pd.read_sql("SELECT * FROM stocks", conn)
stocks_df["$ Change"] = stocks_df["Price"] - stocks_df["InitialPrice"]
stocks_df["% Change"] = (stocks_df["$ Change"] / stocks_df["InitialPrice"]) * 100
stocks_df = stocks_df.sort_values("Ticker").reset_index(drop=True)

st.dataframe(
    stocks_df[["Ticker", "Name", "Price", "Volatility", "$ Change", "% Change"]]
    .style.format({
        "Price": "{:.2f}",
        "Volatility": "{:.3f}",
        "$ Change": "+{:.2f}" if stocks_df["$ Change"].iloc[0] >= 0 else "{:.2f}",
        "% Change": "+{:.2f}%" if stocks_df["% Change"].iloc[0] >= 0 else "{:.2f}%"
    }),
    use_container_width=True,
    height=400  # Adjusted height
)

# --- Stock Price History ---
st.markdown("### 📊 Stock Price History")
selected_ticker = st.selectbox("Choose a stock", stocks_df["Ticker"])

if selected_ticker:
    hist = pd.read_sql(
        "SELECT * FROM price_history WHERE Ticker = ? ORDER BY Timestamp",
        conn,
        params=(selected_ticker,)
    )

    if not hist.empty:
        hist["Date"] = pd.to_datetime(hist["Timestamp"], errors="coerce")
        hist["SimTime"] = (hist["Date"] - SIM_START_DATE).dt.total_seconds() // 3600

        view_range = st.radio(
            "Select timeframe:",
            ["1 Day", "1 Week", "1 Month", "3 Months", "Year to Date", "1Y", "Alltime"],
            horizontal=True
        )

        now_sim_hours = st.session_state.sim_time * (24 / TICKS_PER_DAY)

        if view_range == "1 Day":
            hist_filtered = hist[hist["SimTime"] >= now_sim_hours - 24]
            x_field = alt.X("Date:T", title="Hour", axis=alt.Axis(format="%H:%M"))
        elif view_range == "1 Week":
            hist_filtered = hist[hist["SimTime"] >= now_sim_hours - 168]
            hist_filtered["Date"] = hist_filtered["Date"].dt.floor("D")
            hist_filtered = hist_filtered.groupby("Date", as_index=False).agg({"Price": "mean"})
            x_field = alt.X("Date:T", title="Date", axis=alt.Axis(format="%b %d"))
        elif view_range == "1 Month":
            hist_filtered = hist[hist["SimTime"] >= now_sim_hours - 720]
            hist_filtered["Date"] = hist_filtered["Date"].dt.floor("D")
            hist_filtered = hist_filtered.groupby("Date", as_index=False).agg({"Price": "mean"})
            x_field = alt.X("Date:T", title="Date", axis=alt.Axis(format="%b %d"))
        elif view_range == "3 Months":
            hist_filtered = hist[hist["SimTime"] >= now_sim_hours - 2160]
            hist_filtered["Date"] = hist_filtered["Date"].dt.floor("W")
            hist_filtered = hist_filtered.groupby("Date", as_index=False).agg({"Price": "mean"})
            x_field = alt.X("Date:T", title="Week", axis=alt.Axis(format="%b %d"))
        elif view_range == "Year to Date":
            hist_filtered = hist[hist["Date"].dt.year == SIM_START_DATE.year]
            hist_filtered["Date"] = hist_filtered["Date"].dt.floor("M")
            hist_filtered = hist_filtered.groupby("Date", as_index=False).agg({"Price": "mean"})
            x_field = alt.X("Date:T", title="Month", axis=alt.Axis(format="%b"))
        elif view_range == "1Y":
            hist_filtered = hist[hist["SimTime"] >= now_sim_hours - 8760]
            hist_filtered["Date"] = hist_filtered["Date"].dt.floor("M")
            hist_filtered = hist_filtered.groupby("Date", as_index=False).agg({"Price": "mean"})
            x_field = alt.X("Date:T", title="Month", axis=alt.Axis(format="%b"))
        else:  # Alltime
            hist_filtered = hist
            hist_filtered["Date"] = hist_filtered["Date"].dt.floor("Y")
            hist_filtered = hist_filtered.groupby("Date", as_index=False).agg({"Price": "mean"})
            x_field = alt.X("Date:T", title="Year", axis=alt.Axis(format="%Y"))

        if not hist_filtered.empty:
            low, high = hist_filtered["Price"].min(), hist_filtered["Price"].max()
            padding = (high - low) * 0.1

            chart = alt.Chart(hist_filtered).mark_line(color="steelblue", size=2).encode(
                x=x_field,
                y=alt.Y("Price:Q", scale=alt.Scale(
