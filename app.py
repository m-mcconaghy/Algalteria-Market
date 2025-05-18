import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import time
from streamlit_autorefresh import st_autorefresh
import altair as alt
import os

TICKS_PER_DAY = 3  # Used for faster simulation during Advance mode

DATABASE_PATH = "market.db"

st.set_page_config(page_title="Algalteria Galactic Exchange (AGE)", layout="wide")

# --- Database Connection ---
conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS stocks (
    Ticker TEXT PRIMARY KEY,
    Name TEXT,
    Price REAL,
    Volatility REAL,
    InitialPrice REAL,
    DriftMultiplier REAL DEFAULT 1.0  -- Added drift multiplier
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


# --- Database Upload ---
if is_admin:
    uploaded_file = st.file_uploader("Upload Database File", type=["db"])
    if uploaded_file is not None:
        with open(DATABASE_PATH, "wb") as f:
            f.write(uploaded_file.read())
        st.success("Database file uploaded successfully! Please refresh the page to load the data.")
        # Re-establish the database connection after upload
        conn.close()
        conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
        cursor = conn.cursor()
        # Clear session state to force reload (optional, but can prevent issues)
        for key in st.session_state.keys():
            del st.session_state[key]

# --- Header and Market Status ---
st.title("\U0001F30C Algalteria Galactic Exchange (AGE)")

col_status, col_admin = st.columns([3, 1])
with col_status:
    st.markdown(f"### \U0001F4C8 Market Status: {'<span style=\"color: green;\">🟢 RUNNING</span>' if st.session_state.running else '<span style=\"color: red;\">🔴 PAUSED</span>'}", unsafe_allow_html=True)
with col_admin:
    if is_admin:
        st.success("\U0001F9D1‍\U0001F680 Admin Mode")
        if st.button("⏯ Pause / Resume Market"):
            st.session_state.running = not st.session_state.running
            cursor.execute("REPLACE INTO market_status (key, value) VALUES (?, ?)", ("running", str(st.session_state.running)))
            conn.commit()
    else:
        st.info("\U0001F6F8 Viewer Mode — Live Market Feed Only")

# --- Download Button ---
if is_admin:
    def download_database():
        with open(DATABASE_PATH, "rb") as f:
            db_bytes = f.read()
        st.download_button(
            label="💾 Download Current Database",
            data=db_bytes,
            file_name="market_data.db",
            mime="application/octet-stream"
        )

download_database()
st.markdown("---")

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
            # Incorporate the stock-specific drift multiplier
            drift_rate = (financial_drift * mult * row["DriftMultiplier"]) / 24
            drift = np.clip(drift_rate * tick_scale * row["Price"], -0.002 * row["Price"], 0.002 * row["Price"])

            # Mean reversion component
            mean_reversion_strength = 0.01  # Adjust this strength as needed
            mean_reversion = mean_reversion_strength * (row["InitialPrice"] - row["Price"])

            # Optional: adjust shock chance for tick scale (roughly per day chance = 0.1%)
            daily_shock_chance = 0.001
            shock_chance = 1 - (1 - daily_shock_chance) ** (tick_scale / 24)
            shock_factor = 1.0
            if np.random.rand() < shock_chance:
                shock_factor = np.random.choice([0.95, 1.05], p=[0.5, 0.5])

            base_price = row["Price"] * shock_factor
            new_price = base_price + noise * base_price + drift + mean_reversion #Added mean reversion
            limit = 0.01  # 1% per tick
            new_price = float(np.clip(new_price, row["Price"] * (1 - limit), row["Price"] * (1 + limit)))
            new_price = max(new_price, 0.01)

            # Update InitialPrice once per real day
            if st.session_state.sim_time % (24 / (24 / TICKS_PER_DAY)) == 0:  # Update once per 24 simulation hours
                new_initial_price = row["InitialPrice"] * 1.00005  # Slightly reduced growth
                cursor.execute("UPDATE stocks SET InitialPrice = ? WHERE Ticker = ?", (new_initial_price, row["Ticker"]))

            df.at[idx, "Price"] = new_price
            sim_timestamp = SIM_START_DATE + timedelta(hours=(st.session_state.sim_time * (24 / TICKS_PER_DAY)))  # Corrected timedelta
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
            hist_filtered["Date"] = hist_filtered["Date"].dt.floor("D")
            hist_filtered = hist_filtered.groupby("Date", as_index=False).agg({"Price": "mean"})
            x_field = alt.X("Date:T", title="Week", axis=alt.Axis(format="%b %d"))
        elif view_range == "Year to Date":
            hist_filtered = hist[hist["Date"].dt.year == SIM_START_DATE.year]
            hist_filtered["Date"] = hist_filtered["Date"].dt.floor("D")
            hist_filtered = hist_filtered.groupby("Date", as_index=False).agg({"Price": "mean"})
            x_field = alt.X("Date:T", title="Month", axis=alt.Axis(format="%b"))
        elif view_range == "1Y":
            hist_filtered = hist[hist["SimTime"] >= now_sim_hours - 8760]
            hist_filtered["Date"] = hist_filtered["Date"].dt.floor("D")
            hist_filtered = hist_filtered.groupby("Date", as_index=False).agg({"Price": "mean"})
            x_field = alt.X("Date:T", title="Month", axis=alt.Axis(format="%b"))
        else:  # Alltime
            hist_filtered = hist
            hist_filtered["Date"] = hist_filtered["Date"].dt.floor("D")
            hist_filtered = hist_filtered.groupby("Date", as_index=False).agg({"Price": "mean"})
            x_field = alt.X("Date:T", title="Year", axis=alt.Axis(format="%Y"))

        if not hist_filtered.empty:
            low, high = hist_filtered["Price"].min(), hist_filtered["Price"].max()
            padding = (high - low) * 0.1

            chart = alt.Chart(hist_filtered).mark_line(color="steelblue", size=2).encode(
                x=x_field,
                y=alt.Y("Price:Q", scale=alt.Scale(domain=[low - padding, high + padding]),
                        axis=alt.Axis(title="Price (cr)", grid=True)),
                tooltip=["Date:T", "Price:Q"]
            ).properties(
                title=f"{selected_ticker} Price History",
                width="container",
                height=300
            ).interactive()  # Added interactivity for zoom/pan

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No price history available for the selected timeframe.")
    else:
        st.info("No price history available yet for this stock.")

# --- Admin Controls ---
if is_admin:
    st.sidebar.header("⚙️ Admin Tools")

    with st.sidebar.expander("🎯 Manual Stock Controls"):
        st.markdown("##### Modify Stock Price")
        col_manual1, col_manual2 = st.columns(2)
        with col_manual1:
            ticker_to_change = st.selectbox("Stock to modify", base_tickers + ["TMF"])
        with col_manual2:
            price_change = st.number_input("New Price", min_value=0.01)
        if st.button("✅ Apply Price Change"):
            cursor.execute("UPDATE stocks SET Price = ? WHERE Ticker = ?", (price_change, ticker_to_change))
            sim_timestamp_now = SIM_START_DATE + timedelta(hours=(st.session_state.sim_time * (24 / TICKS_PER_DAY)))
            cursor.execute("INSERT INTO price_history (Timestamp, Ticker, Price) VALUES (?, ?, ?)",
                            (sim_timestamp_now.isoformat(), ticker_to_change, price_change))
            conn.commit()
            st.success(f"Updated {ticker_to_change} to {price_change:.2f} credits.")

    st.sidebar.divider()
    with st.sidebar.expander("⏩ Advance Simulation Time"):
        st.markdown("##### Step Forward")
        col_advance1, col_advance2 = st.columns(2)
        with col_advance1:
            if st.button("Advance 1 Hour"):
                update_prices()
            if st.button("Advance 1 Day"):
                update_prices(ticks=TICKS_PER_DAY)
        with col_advance2:
            if st.button("Advance 1 Week"):
                update_prices(ticks=7 * TICKS_PER_DAY)
            if st.button("Advance 1 Month"):
                update_prices(ticks=30 * TICKS_PER_DAY)
        if st.button("Advance 1 Year"):
            update_prices(ticks=365 * TICKS_PER_DAY)

    st.sidebar.divider()
    with st.sidebar.expander("📉 Adjust Stock Volatility"):
        st.markdown("##### Change Volatility")
        tickers = pd.read_sql("SELECT Ticker FROM stocks", conn)["Ticker"].tolist()
        selected_vol_ticker = st.selectbox("Select Stock", tickers)
        current_vol = pd.read_sql("SELECT Volatility FROM stocks WHERE Ticker = ?", conn, params=(selected_vol_ticker,)).iloc[0, 0]
        new_vol = st.number_input("New Volatility", value=current_vol, step=0.001, format="%.3f", key=f"vol_{selected_vol_ticker}")
        if st.button("📈 Apply Volatility Change"):
            cursor.execute("UPDATE stocks SET Volatility = ? WHERE Ticker = ?", (new_vol, selected_vol_ticker))
            conn.commit()
            st.success(f"Updated volatility of {selected_vol_ticker} to {new_vol:.3f}")

    st.sidebar.divider()
    with st.sidebar.expander("⚖️ Adjust Stock Drift"):  # New section for drift adjustment
        st.markdown("##### Change Drift Influence")
        drift_tickers = pd.read_sql("SELECT Ticker FROM stocks", conn)["Ticker"].tolist()
        selected_drift_ticker = st.selectbox("Select Stock to Adjust Drift", drift_tickers)
        current_drift = pd.read_sql("SELECT DriftMultiplier FROM stocks WHERE Ticker = ?", conn, params=(selected_drift_ticker,)).iloc[0, 0]
        new_drift = st.number_input("Drift Multiplier (Dependant on Market Sentiment)", value=current_drift, step=0.01, format="%.2f", key=f"drift_{selected_drift_ticker}")
        if st.button("✅ Apply Drift Change"):
            cursor.execute("UPDATE stocks SET DriftMultiplier = ? WHERE Ticker = ?", (new_drift, selected_drift_ticker))
            conn.commit()
            st.success(f"Updated drift multiplier of {selected_drift_ticker} to {new_drift:.2f}")

    st.sidebar.divider()
    with st.sidebar.expander("🏦 Market Parameters"):
        st.markdown("##### Set Global Parameters")
        new_rfr = st.number_input("Risk-Free Rate", value=st.session_state.risk_free_rate, step=0.0001, format="%.4f")
        st.session_state.risk_free_rate = new_rfr
        new_erp = st.number_input("Equity Risk Premium", value=st.session_state.equity_risk_premium, step=0.0001, format="%.4f")
        st.session_state.equity_risk_premium = new_erp
        tick_rate = st.slider("⏱️ Tick Interval (seconds)", 10, 300, st.session_state.tick_interval_sec, step=10)
        st.session_state.tick_interval_sec = tick_rate

    st.sidebar.divider()
    with st.sidebar.expander("🌐 Market Sentiment"):
        st.markdown("##### Influence Market Direction")
        market_sentiment_options = {
            "Bubbling": "📈 Bubbling (Strong Upward)",
            "Booming": "↗️ Booming (Moderately Bullish)",
            "Stagnant": "➡️ Stagnant (Neutral)",
            "Receding": "↘️ Receding (Downward Trend)",
            "Depression": "📉 Depression (Heavy Bearish)"
        }
        selected_sentiment_key = st.selectbox("Set Sentiment", list(market_sentiment_options.keys()), index=list(market_sentiment_options.keys()).index(st.session_state.market_sentiment))
        st.session_state.market_sentiment = selected_sentiment_key
