import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
import time
from streamlit_autorefresh import st_autorefresh
import altair as alt
import os

TICKS_PER_DAY = 3  # Used for faster simulation during Advance mode

st.set_page_config(page_title="Algalteria Galactic Exchange (AGE)", layout="wide")


# --- Database Connection ---
def get_connection():
    """Connect to MySQL database on Google Cloud."""
    try:
        conn = mysql.connector.connect(
            host=st.secrets["DB_HOST"],         # Your Google Cloud MySQL IP
            user=st.secrets["DB_USER"],         # Your MySQL username
            password=st.secrets["DB_PASSWORD"], # Your MySQL password
            database=st.secrets["DB_NAME"]      # Your database name
        )
        return conn
    except Error as e:
        st.error(f"Error connecting to MySQL: {e}")
        return None

def get_cursor(conn):
    """Gets a cursor from the database connection."""
    try:
        cursor = conn.cursor()
        return cursor
    except Exception as e:
        st.error(f"Error getting cursor: {e}")
        return None


# --- Initialize Database Tables ---
def initialize_database():
    """Initializes the database tables if they don't exist."""
    conn = get_connection()
    if conn is None:
        return  # Exit if no connection

    cursor = get_cursor(conn)
    if cursor is None:
        conn.close()
        return

    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stocks (
                Ticker VARCHAR(10) PRIMARY KEY,
                Name VARCHAR(255),
                Price DOUBLE,
                Volatility DOUBLE,
                InitialPrice DOUBLE,
                DriftMultiplier DOUBLE DEFAULT 1.0
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                Timestamp DATETIME,
                Ticker VARCHAR(10),
                Price DOUBLE
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_status (
                `key` VARCHAR(255) PRIMARY KEY,
                `value` VARCHAR(255)
            )
        """)
        conn.commit()
    except Exception as e:
        st.error(f"Error initializing database tables: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


initialize_database()  # Call at the beginning

# --- Load market running state from database ---
def load_market_status():
    """Load market running status from database."""
    conn = get_connection()
    if conn is None:
        return True  # Default to running if can't connect

    cursor = get_cursor(conn)
    if cursor is None:
        conn.close()
        return True

    try:
        cursor.execute("SELECT value FROM market_status WHERE key = 'running'")
        result = cursor.fetchone()
        if result:
            return result[0].lower() == 'true'
        else:
            return True  # Default to running if no record exists
    except Exception as e:
        st.error(f"Error loading market status: {e}")
        return True
    finally:
        cursor.close()
        conn.close()


# --- Session State Initialization ---
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

# Load market running state from database instead of setting a default
if "market_running" not in st.session_state:
    st.session_state.market_running = load_market_status()

# --- Initial Stock Data ---
base_tickers = ["DTF", "GMG", "USF", "TTT", "GFU", "IWI", "EE", "NEC", "ARC", "SOL", "AWE", "ORB", "QNT", "AGX", "LCO", "FMC", "SYX", "VLT", "EXR", "CRB"]
names = ["Directorate Tech Fund", "Galactic Mining Guild", "Universal Services Fund", "The Textile Team", "Galactic Farmers Union", "Imperial Weapons Industry", "Epsilon Exchange", "Nebular Energy Consortium", "Asteroid Resources Collective", "Solar Operations League", "Algalterian Water Exchange", "Orbital Rare Biotech", "Quantum Nexus Trust", "Agricultural Exports Guild", "Lunar Construction Outfit", "Frontier Medical Consortium", "Syphonix Energy Systems", "Veltrax AI Logistics", "Exorium Rare Elements", "Crystalline Banking Network"]
initial_prices = [105.0, 95.0, 87.5, 76.0, 82.0, 132.0, 151.0, 91.0, 87.5, 102.0, 78.0, 113.0, 139.0, 84.0, 62.0, 144.0, 193.0, 119.0, 221.0, 68.0]
volatility = [0.04, 0.035, 0.015, 0.02, 0.025, 0.03, 0.06, 0.018, 0.025, 0.02, 0.015, 0.045, 0.03, 0.017, 0.023, 0.014, 0.055, 0.027, 0.06, 0.018]


# --- Initialize Stocks Table ---
def initialize_stocks():
    """Initializes the stocks table with data if it's empty."""
    conn = get_connection()
    if conn is None:
        return

    cursor = get_cursor(conn)
    if cursor is None:
        conn.close()
        return

    try:
        cursor.execute("SELECT COUNT(*) FROM stocks")
        if cursor.fetchone()[0] == 0:
            for i in range(len(base_tickers)):
                cursor.execute("""
                    INSERT INTO stocks (Ticker, Name, Price, Volatility, InitialPrice)
                    VALUES (%s, %s, %s, %s, %s)
                    """, (base_tickers[i], names[i], initial_prices[i], volatility[i], initial_prices[i]))
            tmf_price = np.average(initial_prices, weights=initial_prices)
            tmf_vol = np.average(volatility, weights=initial_prices)
            cursor.execute("""
                    INSERT INTO stocks (Ticker, Name, Price, Volatility, InitialPrice)
                    VALUES (%s, %s, %s, %s, %s)
                    """, ("TMF", "Total Market Fund", tmf_price, tmf_vol, tmf_price))
            conn.commit()
            st.success("Stocks table initialized.")  # Add success message
        else:
            st.info("Stocks table already contains data.")
    except Exception as e:
        st.error(f"Error initializing stocks table: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


initialize_stocks()


# --- Header and Market Status ---
st.title("\U0001F30C Algalteria Galactic Exchange (AGE)")

col_status, col_admin = st.columns([3, 1])
with col_status:
    st.markdown(
        f"### \U0001F4C8 Market Status: "
        f"{'<span style=\"color: green;\">🟢 RUNNING</span>' if st.session_state.market_running else '<span style=\"color: red;\">🔴 PAUSED</span>'}",
        unsafe_allow_html=True
    )

with col_admin:
    admin_password = st.text_input("Enter admin password", type="password")
    is_admin = admin_password == st.secrets.get("ADMIN_PASSWORD", "secret123")
    if is_admin:
        st.success("\U0001F9D1‍\U0001F680 Admin Mode")

        if st.button("⏯ Pause / Resume Market"):
            # Toggle the market running state
            st.session_state.market_running = not st.session_state.market_running
            
            # Save to database
            conn = get_connection()
            if conn:
                cursor = get_cursor(conn)
                if cursor:
                    try:
                        cursor.execute("""
                            INSERT INTO market_status (`key`, `value`)
                            VALUES (%s, %s)
                            ON DUPLICATE KEY UPDATE `value` = VALUES(`value`)
                        """, ("running", str(st.session_state.market_running)))

                        conn.commit()
                        st.success(f"Market {'resumed' if st.session_state.market_running else 'paused'}")
                    except Exception as e:
                        st.error(f"Error updating market status: {e}")
                        conn.rollback()
                    finally:
                        cursor.close()
                conn.close()
            st.rerun()  # Force a rerun to update the display immediately
    else:
        st.info("\U0001F6F8 Viewer Mode — Live Market Feed Only")


# --- Set the TIME ---
SIM_START_DATE = pd.Timestamp("2200-01-01")


# --- Final update_prices function ---
def update_prices(ticks=1):
    """Updates stock prices for a given number of ticks."""
    conn = get_connection()
    if conn is None:
        return  # Exit if no connection

    cursor = get_cursor(conn)
    if cursor is None:
        conn.close()
        return

    try:
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
                drift = np.clip(drift_rate * tick_scale * row["Price"], -0.002 * row["Price"],
                               0.002 * row["Price"])

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
                new_price = base_price + noise * base_price + drift + mean_reversion  # Added mean reversion
                limit = 0.01  # 1% per tick
                new_price = float(np.clip(new_price, row["Price"] * (1 - limit), row["Price"] * (1 + limit)))
                new_price = max(new_price, 0.01)

                # Update InitialPrice once per real day
                if st.session_state.sim_time % (24 / (24 / TICKS_PER_DAY)) == 0:  # Update once per 24 simulation hours
                    new_initial_price = row["InitialPrice"] * 1.00005  # Slightly reduced growth
                    cursor.execute("UPDATE stocks SET InitialPrice = %s WHERE Ticker = %s",
                                   (new_initial_price, row["Ticker"]))

                df.at[idx, "Price"] = new_price
                sim_timestamp = SIM_START_DATE + timedelta(
                    hours=(st.session_state.sim_time * (24 / TICKS_PER_DAY)))  # Corrected timedelta
                cursor.execute("INSERT INTO price_history (Timestamp, Ticker, Price) VALUES (%s, %s, %s)",
                               (sim_timestamp.isoformat(), row["Ticker"], new_price))

            # Update TMF based on weighted average (using volatility as weight)
            tmf_data = df[df["Ticker"] != "TMF"]
            tmf_price = np.average(tmf_data["Price"], weights=tmf_data["Volatility"])
            df.loc[df["Ticker"] == "TMF", "Price"] = tmf_price

            for _, row in df.iterrows():
                cursor.execute("UPDATE stocks SET Price = %s WHERE Ticker = %s", (row["Price"], row["Ticker"]))

            conn.commit()
            st.session_state.sim_time += 1
    except Exception as e:
        st.error(f"Error updating prices: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


# --- Auto-refresh and Price Updates ---
# Only run auto-refresh if market is running
if st.session_state.market_running:
    count = st_autorefresh(interval=st.session_state.tick_interval_sec * 1000, key="market_tick")
else:
    count = 0  # Set count to 0 when market is paused

if "last_refresh_count" not in st.session_state:
    st.session_state.last_refresh_count = -1

# Only update prices if admin is logged in, market is running, AND we have a new refresh count
if is_admin and st.session_state.market_running and count != st.session_state.last_refresh_count and count > 0:
    update_prices()
    st.session_state.last_update_time = time.time()
    st.session_state.last_refresh_count = count

# Display timing information
if "last_update_time" in st.session_state:
    elapsed = int(time.time() - st.session_state.last_update_time)
    if st.session_state.market_running:
        st.caption(f"⏱️ Last update: {elapsed}s ago — Next in: {max(0, st.session_state.tick_interval_sec - elapsed)}s")
    else:
        st.caption(f"⏸️ Market paused — Last update: {elapsed}s ago")
else:
    if st.session_state.market_running:
        st.caption("⏱️ Waiting for first update...")
    else:
        st.caption("⏸️ Market paused")

# --- Display Stock Data ---
st.markdown("### 📈 Current Stock Prices")
def display_stock_data():
    """Displays the current stock prices in a DataFrame."""
    conn = get_connection()
    if conn is None:
        return

    try:
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
    except Exception as e:
        st.error(f"Error displaying stock data: {e}")
    finally:
        conn.close()

display_stock_data()


# --- Stock Price History ---
st.markdown("### 📊 Stock Price History")
selected_ticker = st.selectbox("Choose a stock", base_tickers + ["TMF"])


def display_stock_history(ticker):
    """Displays the price history of a selected stock."""
    conn = get_connection()
    if conn is None:
        return

    try:
        hist = pd.read_sql(
            "SELECT * FROM price_history WHERE Ticker = %s ORDER BY Timestamp",
            conn,
            params=(ticker,)
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
                    title=f"{ticker} Price History",
                    width="container",
                    height=300
                ).interactive()  # Added interactivity for zoom/pan

                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No price history available for the selected timeframe.")
        else:
            st.info("No price history available yet for this stock.")
    except Exception as e:
        st.error(f"Error displaying stock history: {e}")
    finally:
        conn.close()


if selected_ticker:
    display_stock_history(selected_ticker)


# --- Admin Controls ---
if is_admin:
    st.sidebar.header("⚙️ Admin Tools")

    with st.sidebar.expander("Database Upload"):
        uploaded_file = st.file_uploader("Upload Database File", type=["db"])
        if uploaded_file is not None:
            try:
                with open(DATABASE_PATH, "wb") as f:
                    f.write(uploaded_file.read())
                st.success("Database file uploaded successfully! Please refresh the page to load the data.")

                # Clear session state to force reload
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun() # Use st.rerun()
            except Exception as e:
                st.error(f"Error handling uploaded database: {e}")

    st.sidebar.divider()
    with st.sidebar.expander("🎯 Manual Stock Controls"):
        st.markdown("##### Modify Stock Price")
        col_manual1, col_manual2 = st.columns(2)
        with col_manual1:
            ticker_to_change = st.selectbox("Stock to modify", base_tickers + ["TMF"])
        with col_manual2:
            price_change = st.number_input("New Price", min_value=0.01)
        if st.button("✅ Apply Price Change"):
            conn = get_connection()
            if conn:
                cursor = get_cursor(conn)
                if cursor:
                    try:
                        cursor.execute("UPDATE stocks SET Price = %s WHERE Ticker = %s", (price_change, ticker_to_change))
                        sim_timestamp_now = SIM_START_DATE + timedelta(
                            hours=(st.session_state.sim_time * (24 / TICKS_PER_DAY)))
                        cursor.execute("INSERT INTO price_history (Timestamp, Ticker, Price) VALUES (%s, %s, %s)",
                                       (sim_timestamp_now.isoformat(), ticker_to_change, price_change))
                        conn.commit()
                        st.success(f"Updated {ticker_to_change} to {price_change:.2f} credits.")
                    except Exception as e:
                        st.error(f"Error applying price change: {e}")
                        conn.rollback()
                    finally:
                        cursor.close()
                conn.close()

    st.sidebar.divider()
    with st.sidebar.expander("⏩ Advance Simulation Time"):
        st.markdown("##### Step Forward")
        col_advance1, col_advance2 = st.columns(2)
        with col_advance1:
            if st.button("Advance 1 Hour"):
                update_prices(ticks=1)
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
        conn = get_connection()
        if conn:
            cursor = get_cursor(conn)
            if cursor:
                try:
                    tickers = pd.read_sql("SELECT Ticker FROM stocks", conn)["Ticker"].tolist()
                    selected_vol_ticker = st.selectbox("Select Stock", tickers)
                    current_vol = pd.read_sql("SELECT Volatility FROM stocks WHERE Ticker = %s", conn,
                                            params=(selected_vol_ticker,)).iloc[0, 0]
                    new_vol = st.number_input("New Volatility", value=current_vol, step=0.001, format="%.3f",
                                            key=f"vol_{selected_vol_ticker}")
                    if st.button("📈 Apply Volatility Change"):
                        cursor.execute("UPDATE stocks SET Volatility = %s WHERE Ticker = %s", (new_vol, selected_vol_ticker))
                        conn.commit()
                        st.success(f"Updated volatility of {selected_vol_ticker} to {new_vol:.3f}")
                except Exception as e:
                    st.error(f"Error adjusting volatility: {e}")
                    conn.rollback()
                finally:
                    cursor.close()
            conn.close()

    st.sidebar.divider()
    with st.sidebar.expander("⚖️ Adjust Stock Drift"):  # New section for drift adjustment
        st.markdown("##### Change Drift Influence")
        conn = get_connection()
        if conn:
            cursor = get_cursor(conn)
            if cursor:
                try:
                    drift_tickers = pd.read_sql("SELECT Ticker FROM stocks", conn)["Ticker"].tolist()
                    selected_drift_ticker = st.selectbox("Select Stock to Adjust Drift", drift_tickers)
                    current_drift = pd.read_sql("SELECT DriftMultiplier FROM stocks WHERE Ticker = %s", conn,
                                                params=(selected_drift_ticker,)).iloc[0, 0]
                    new_drift = st.number_input("Drift Multiplier (Dependant on Market Sentiment)", value=current_drift,
                                                step=0.01, format="%.2f", key=f"drift_{selected_drift_ticker}")
                    if st.button("✅ Apply Drift Change"):
                        cursor.execute("UPDATE stocks SET DriftMultiplier = %s WHERE Ticker = %s",
                                        (new_drift, selected_drift_ticker))
                        conn.commit()
                        st.success(f"Updated drift multiplier of {selected_drift_ticker} to {new_drift:.2f}")
                except Exception as e:
                    st.error(f"Error adjusting drift: {e}")
                    conn.rollback()
                finally:
                    cursor.close()
            conn.close()

    st.sidebar.divider()
    with st.sidebar.expander("🏦 Market Parameters"):
        st.markdown("##### Set Global Parameters")
        new_rfr = st.number_input("Risk-Free Rate", value=st.session_state.risk_free_rate, step=0.0001, format="%.4f")
        st.session_state.risk_free_rate = new_rfr
        new_erp = st.number_input("Equity Risk Premium", value=st.session_state.equity_risk_premium, step=0.0001,
                                    format="%.4f")
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
        selected_sentiment_key = st.selectbox("Set Sentiment", list(market_sentiment_options.keys()),
                                            index=list(market_sentiment_options.keys()).index(
                                                st.session_state.market_sentiment))
        st.session_state.market_sentiment = selected_sentiment_key
