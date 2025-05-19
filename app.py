import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from streamlit_autorefresh import st_autorefresh
import altair as alt
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

TICKS_PER_DAY = 3  # Used for faster simulation during Advance mode

st.set_page_config(page_title="Algalteria Galactic Exchange (AGE)", layout="wide")

# --- Database Connection ---
@st.cache_resource
def get_sqlalchemy_engine():
    return create_engine(
        f"mysql+pymysql://{st.secrets['DB_USER']}:{st.secrets['DB_PASSWORD']}@{st.secrets['DB_HOST']}/{st.secrets['DB_NAME']}"
    )

engine = get_sqlalchemy_engine()

# --- Initialize Database Tables ---
def initialize_database():
    """Initializes the database tables if they don't exist."""
    try:
        with engine.connect() as connection:
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS stocks (
                    Ticker VARCHAR(10) PRIMARY KEY,
                    Name VARCHAR(255),
                    Price DOUBLE,
                    Volatility DOUBLE,
                    InitialPrice DOUBLE,
                    DriftMultiplier DOUBLE DEFAULT 1.0
                )
            """))
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS price_history (
                    Timestamp DATETIME,
                    Ticker VARCHAR(10),
                    Price DOUBLE
                )
            """))
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS market_status (
                    `key` VARCHAR(255) PRIMARY KEY,
                    `value` VARCHAR(255)
                )
            """))
            connection.commit()
    except SQLAlchemyError as e:
        st.error(f"Error initializing database tables: {e}")


initialize_database()  # Call at the beginning

# --- Load market running state from database ---
def load_market_status():
    """Load market running status from database."""
    try:
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT value FROM market_status WHERE `key` = 'running'")
            ).fetchone()
            if result:
                return result[0].lower() == 'true'
            else:
                return True  # Default to running if no record exists
    except SQLAlchemyError as e:
        st.error(f"Error loading market status: {e}")
        return True
    return True


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
    try:
        with engine.connect() as connection:
            count = connection.execute(text("SELECT COUNT(*) FROM stocks")).scalar_one()
            if count == 0:
                for i in range(len(base_tickers)):
                    connection.execute(text("""
                        INSERT IGNORE INTO stocks (Ticker, Name, Price, Volatility, InitialPrice)
                        VALUES (:ticker, :name, :price, :volatility, :initial_price)
                    """), {
                        "ticker": base_tickers[i],
                        "name": names[i],
                        "price": float(initial_prices[i]),
                        "volatility": float(volatility[i]),
                        "initial_price": float(initial_prices[i])
                    })
                tmf_price = float(np.average(initial_prices, weights=initial_prices))
                tmf_vol = float(np.average(volatility, weights=initial_prices))
                connection.execute(text("""
                    INSERT IGNORE INTO stocks (Ticker, Name, Price, Volatility, InitialPrice)
                    VALUES (:ticker, :name, :price, :volatility, :initial_price)
                """), {
                    "ticker": "TMF",
                    "name": "Total Market Fund",
                    "price": tmf_price,
                    "volatility": tmf_vol,
                    "initial_price": tmf_price
                })
                connection.commit()
                st.success("Stocks table initialized.")  # Add success message
            else:
                st.info("Stocks table already contains data.")
    except SQLAlchemyError as e:
        st.error(f"Error initializing stocks table: {e}")


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
            try:
                with engine.connect() as connection:
                    connection.execute(text("""
                        INSERT INTO market_status (`key`, `value`)
                        VALUES (:key, :value)
                        ON DUPLICATE KEY UPDATE `value` = VALUES(`value`)
                    """), {"key": "running", "value": str(st.session_state.market_running)})
                    connection.commit()
                    st.success(f"Market {'resumed' if st.session_state.market_running else 'paused'}")
            except SQLAlchemyError as e:
                st.error(f"Error updating market status: {e}")
            st.rerun()  # Force a rerun to update the display immediately
    else:
        st.info("\U0001F6F8 Viewer Mode — Live Market Feed Only")


# --- Set the TIME ---
SIM_START_DATE = pd.Timestamp("2200-01-01")


# --- Final update_prices function ---
def update_prices(ticks=1):
    """Optimized: Updates stock prices using batch inserts/updates."""
    try:
        with engine.connect() as connection:
            SIM_START_DATE = pd.Timestamp("2200-01-01")
            price_history_batch = []
            update_price_batch = []
            update_initial_price_batch = []

            for i in range(ticks):  # Iterate with an index
                df = pd.read_sql(text("SELECT * FROM stocks"), connection)
                tick_scale = 24 / TICKS_PER_DAY

                for idx, row in df.iterrows():
                    if row["Ticker"] == "TMF":
                        continue

                    regime_multiplier = np.random.choice([1, 1.5], p=[0.95, 0.05])
                    scaled_vol = row["Volatility"] * np.sqrt(tick_scale / 24)
                    momentum = np.random.choice([1, -1])
                    noise = np.random.normal(0, scaled_vol * regime_multiplier) * momentum

                    financial_drift = st.session_state.risk_free_rate + st.session_state.equity_risk_premium
                    sentiment_multiplier = {
                        "Bubbling": 0.03,
                        "Booming": 0.01,
                        "Stagnant": 0.005,
                        "Receding": -0.02,
                        "Depression": -0.05
                    }
                    selected_sentiment = st.session_state.get("market_sentiment", "Booming")
                    mult = sentiment_multiplier.get(selected_sentiment, 1.0)
                    drift_rate = (financial_drift * mult * row["DriftMultiplier"]) / 24
                    drift = np.clip(drift_rate * tick_scale * row["Price"], -0.002 * row["Price"], 0.002 * row["Price"])

                    mean_reversion = 0.01 * (row["InitialPrice"] - row["Price"])

                    daily_shock_chance = 0.001
                    shock_chance = 1 - (1 - daily_shock_chance) ** (tick_scale / 24)
                    shock_factor = np.random.choice([0.95, 1.05], p=[0.5, 0.5]) if np.random.rand() < shock_chance else 1.0

                    base_price = row["Price"] * shock_factor
                    new_price = base_price + noise * base_price + drift + mean_reversion
                    new_price = float(np.clip(new_price, row["Price"] * 0.99, row["Price"] * 1.01))
                    new_price = max(new_price, 0.01)

                    if st.session_state.sim_time % (24 / (24 / TICKS_PER_DAY)) == 0:
                        new_initial_price = row["InitialPrice"] * 1.00005
                        update_initial_price_batch.append((new_initial_price, row["Ticker"]))

                    df.at[idx, "Price"] = new_price
                    # Corrected timestamp calculation:
                    current_sim_ticks = st.session_state.sim_time + i
                    sim_timestamp = SIM_START_DATE + timedelta(hours=(current_sim_ticks * (24 / TICKS_PER_DAY)))
                    price_history_batch.append((sim_timestamp, row["Ticker"], new_price))

                tmf_data = df[df["Ticker"] != "TMF"]
                tmf_price = float(np.average(tmf_data["Price"], weights=tmf_data["Volatility"]))
                df.loc[df["Ticker"] == "TMF", "Price"] = tmf_price

                for _, row in df.iterrows():
                    update_price_batch.append((row["Price"], row["Ticker"]))

            connection.execute(
                text("INSERT INTO price_history (Timestamp, Ticker, Price) VALUES (:timestamp, :ticker, :price)"),
                [{"timestamp": ts, "ticker": tick, "price": p} for ts, tick, p in price_history_batch]
            )
            connection.execute(
                text("UPDATE stocks SET Price = :price WHERE Ticker = :ticker"),
                [{"price": price, "ticker": ticker} for price, ticker in update_price_batch]
            )
            if update_initial_price_batch:
                connection.execute(
                    text("UPDATE stocks SET InitialPrice = :initial_price WHERE Ticker = :ticker"),
                    [{"initial_price": ip, "ticker": tick} for ip, tick in update_initial_price_batch]
                )
            connection.commit()
            st.session_state.sim_time += ticks
    except SQLAlchemyError as e:
        st.error(f"Error updating prices: {e}")

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
    try:
        stocks_df = pd.read_sql(text("SELECT * FROM stocks"), engine)
        if stocks_df.empty:
            st.info("No stock data available. Initialize stocks to begin simulation.")
            return

        stocks_df["$ Change"] = stocks_df["Price"] - stocks_df["InitialPrice"]
        stocks_df["% Change"] = (stocks_df["$ Change"] / stocks_df["InitialPrice"]) * 100
        stocks_df = stocks_df.sort_values("Ticker").reset_index(drop=True)

        # Use a safe default format
        style_format = {
            "Price": "{:.2f}",
            "Volatility": "{:.3f}",
            "$ Change": "{:+.2f}",
            "% Change": "{:+.2f}%"
        }

        st.dataframe(
            stocks_df[["Ticker", "Name", "Price", "Volatility", "$ Change", "% Change"]]
            .style.format(style_format),
            use_container_width=True,
            height=400
        )
    except SQLAlchemyError as e:
        st.error(f"Error displaying stock data: {e}")

display_stock_data()


# --- Stock Price History ---
st.markdown("### 📊 Stock Price History")
selected_ticker = st.selectbox("Choose a stock", base_tickers + ["TMF"])


def display_stock_history(ticker):
    """Displays the price history of a selected stock."""
    try:
        hist = pd.read_sql(
            text("SELECT * FROM price_history WHERE Ticker = :ticker ORDER BY Timestamp"),
            engine,
            params={"ticker": ticker}
        )

        if hist.empty:
            st.info("No history found for this stock.")
            return

        hist["Date"] = pd.to_datetime(hist["Timestamp"], errors="coerce")
        hist["SimTime"] = (hist["Date"] - SIM_START_DATE).dt.total_seconds() // 3600

        view_range = st.radio(
            "Select timeframe:",
            ["1 Day", "1 Week", "1 Month", "3 Months", "Year to Date", "1Y", "Alltime"],
            horizontal=True
        )

        now_sim_hours = st.session_state.sim_time * (24 / TICKS_PER_DAY)

        # Apply view range logic
        if view_range == "1 Day":
            hist_filtered = hist[hist["SimTime"] >= now_sim_hours - 24].copy()
            x_field = alt.X("Date:T", title="Hour", axis=alt.Axis(format="%H:%M"))

        elif view_range == "1 Week":
            hist_filtered = hist[hist["SimTime"] >= now_sim_hours - 168].copy()
            hist_filtered["Date"] = hist_filtered["Date"].dt.floor("D")
            hist_filtered = hist_filtered.groupby("Date", as_index=False).agg({"Price": "mean"})
            x_field = alt.X("Date:T", title="Date", axis=alt.Axis(format="%b %d"))

        elif view_range == "1 Month":
            hist_filtered = hist[hist["SimTime"] >= now_sim_hours - 720].copy()
            hist_filtered["Date"] = hist_filtered["Date"].dt.floor("D")
            hist_filtered = hist_filtered.groupby("Date", as_index=False).agg({"Price": "mean"})
            x_field = alt.X("Date:T", title="Date", axis=alt.Axis(format="%b %d"))

        elif view_range == "3 Months":
            hist_filtered = hist[hist["SimTime"] >= now_sim_hours - 2160].copy()
            hist_filtered["Date"] = hist_filtered["Date"].dt.floor("D")
            hist_filtered = hist_filtered.groupby("Date", as_index=False).agg({"Price": "mean"})
            x_field = alt.X("Date:T", title="Week", axis=alt.Axis(format="%b %d"))

        elif view_range == "Year to Date":
            hist_filtered = hist[hist["Date"].dt.year == SIM_START_DATE.year].copy()
            hist_filtered["Date"] = hist_filtered["Date"].dt.floor("D")
            hist_filtered = hist_filtered.groupby("Date", as_index=False).agg({"Price": "mean"})
            x_field = alt.X("Date:T", title="Month", axis=alt.Axis(format="%b"))

        elif view_range == "1Y":
            hist_filtered = hist[hist["SimTime"] >= now_sim_hours - 8760].copy()
            hist_filtered["Date"] = hist_filtered["Date"].dt.floor("D")
            hist_filtered = hist_filtered.groupby("Date", as_index=False).agg({"Price": "mean"})
            x_field = alt.X("Date:T", title="Month", axis=alt.Axis(format="%b"))

        else:  # Alltime
            hist_filtered = hist.copy()
            hist_filtered["Date"] = hist_filtered["Date"].dt.floor("D")
            hist_filtered = hist_filtered.groupby("Date", as_index=False).agg({"Price": "mean"})
            x_field = alt.X("Date:T", title="Year", axis=alt.Axis(format="%Y"))

        # Only plot if data exists
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
            ).interactive()

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No price history available for the selected timeframe.")

    except SQLAlchemyError as e:
        st.error(f"Error displaying stock history: {e}")

if selected_ticker:
    display_stock_history(selected_ticker)
    
st.dataframe(pd.read_sql("SELECT * FROM price_history", engine))


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
            try:
                with engine.connect() as connection:
                    sim_timestamp_now = (SIM_START_DATE + timedelta(
                        hours=(st.session_state.sim_time * (24 / TICKS_PER_DAY)))).to_pydatetime()
                    connection.execute(
                        text("UPDATE stocks SET Price = :price WHERE Ticker = :ticker"),
                        {"price": price_change, "ticker": ticker_to_change}
                    )
                    connection.execute(
                        text("INSERT INTO price_history (Timestamp, Ticker, Price) VALUES (:timestamp, :ticker, :price)"),
                        {"timestamp": sim_timestamp_now, "ticker": ticker_to_change, "price": price_change}
                    )
                    connection.commit()
                    st.success(f"Updated {ticker_to_change} to {price_change:.2f} credits.")
            except SQLAlchemyError as e:
                st.error(f"Error applying price change: {e}")

    st.sidebar.divider()
    with st.sidebar.expander("⏩ Advance Simulation Time"):
        st.markdown("##### Step Forward")
        col_advance1, col_advance2 = st.columns(2)
        with col_advance1:
            if st.button("Advance 1 Hour"):
                update_prices(ticks=1)
                st.success("Advanced 1 hour")
                st.rerun()  # Ensure the UI reloads with new sim_time and updated graph
            if st.button("Advance 1 Day"):
                update_prices(ticks=TICKS_PER_DAY)
                st.success("Advanced 1 day")
                st.rerun()  # Ensure the UI reloads with new sim_time and updated graph
        with col_advance2:
            if st.button("Advance 1 Week"):
                update_prices(ticks=7 * TICKS_PER_DAY)
                st.success("Advanced 1 week")
                st.rerun()  # Ensure the UI reloads with new sim_time and updated graph
            if st.button("Advance 1 Month"):
                update_prices(ticks=30 * TICKS_PER_DAY)
                st.success("Advanced 1 month")
                st.rerun()  # Ensure the UI reloads with new sim_time and updated graph
        if st.button("Advance 1 Year"):
            for _ in range(12):  # Loop 12 times for 12 months
                update_prices(ticks=30 * TICKS_PER_DAY)  # Advance by 30 days (approximately a month)
                st.session_state.sim_time += 30 * TICKS_PER_DAY #Manually update the sim_time
                st.write(f"Advanced 1 month. Current sim_time: {st.session_state.sim_time}") #feedback
        #st.rerun()  # Removed st.rerun() from inside the loop
            st.success("Advanced 1 year (12 months)")  # Message after the loop completes
            st.rerun() #Rerun after the loop


    st.sidebar.divider()
    with st.sidebar.expander("📉 Adjust Stock Volatility"):
        st.markdown("##### Change Volatility")
        try:
            with engine.connect() as connection:
                tickers = pd.read_sql(text("SELECT Ticker FROM stocks"), connection)["Ticker"].tolist()
                selected_vol_ticker = st.selectbox("Select Stock", tickers)
                current_vol = pd.read_sql(
                    text("SELECT Volatility FROM stocks WHERE Ticker = :ticker"),
                    connection,
                    params={"ticker": selected_vol_ticker}
                ).iloc[0, 0]
                new_vol = st.number_input("New Volatility", value=current_vol, step=0.001, format="%.3f",
                                        key=f"vol_{selected_vol_ticker}")
                if st.button("📈 Apply Volatility Change"):
                    connection.execute(
                        text("UPDATE stocks SET Volatility = :volatility WHERE Ticker = :ticker"),
                        {"volatility": new_vol, "ticker": selected_vol_ticker}
                    )
                    connection.commit()
                    st.success(f"Updated volatility of {selected_vol_ticker} to {new_vol:.3f}")
        except SQLAlchemyError as e:
            st.error(f"Error adjusting volatility: {e}")

    st.sidebar.divider()
    with st.sidebar.expander("⚖️ Adjust Stock Drift"):  # New section for drift adjustment
        st.markdown("##### Change Drift Influence")
        try:
            with engine.connect() as connection:
                drift_tickers = pd.read_sql(text("SELECT Ticker FROM stocks"), connection)["Ticker"].tolist()
                selected_drift_ticker = st.selectbox("Select Stock to Adjust Drift", drift_tickers)
                current_drift = pd.read_sql(
                    text("SELECT DriftMultiplier FROM stocks WHERE Ticker = :ticker"),
                    connection,
                    params={"ticker": selected_drift_ticker}
                ).iloc[0, 0]
                new_drift = st.number_input("Drift Multiplier (Dependant on Market Sentiment)", value=current_drift,
                                            step=0.01, format="%.2f", key=f"drift_{selected_drift_ticker}")
                if st.button("✅ Apply Drift Change"):
                    connection.execute(
                        text("UPDATE stocks SET DriftMultiplier = :drift WHERE Ticker = :ticker"),
                        {"drift": new_drift, "ticker": selected_drift_ticker}
                    )
                    connection.commit()
                    st.success(f"Updated drift multiplier of {selected_drift_ticker} to {new_drift:.2f}")
        except SQLAlchemyError as e:
            st.error(f"Error adjusting drift: {e}")

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
    st.sidebar.divider()
    with st.sidebar.expander("💥 **Database Reset**"):
        st.markdown("##### **Danger: Erase All Data and Recreate Tables**")
        admin_password_reset = st.sidebar.text_input("Enter admin password to confirm reset", type="password", key="reset_password")
        is_admin_reset = admin_password_reset == st.secrets.get("ADMIN_PASSWORD", "secret123")
        if is_admin_reset:
            if st.sidebar.button("🔥 **CONFIRM DELETE ALL DATA AND RECREATE TABLES** 🔥", key="reset_button"):
                try:
                    with engine.connect() as connection:
                        connection.execute(text("DROP TABLE IF EXISTS price_history"))
                        connection.execute(text("DROP TABLE IF EXISTS stocks"))
                        connection.execute(text("DROP TABLE IF EXISTS market_status"))
                        connection.commit()
                        st.success("All tables dropped successfully.")
                        # Re-initialize the database structure and initial data
                        initialize_database()
                        initialize_stocks()
                        st.info("Database tables recreated and initial stocks added.")
                except SQLAlchemyError as e:
                    st.error(f"Error dropping tables: {e}")
                st.rerun() # Force a refresh to reflect the changes
        elif admin_password_reset:
            st.sidebar.warning("Incorrect admin password.")

else:
    st.info("\U0001F6F8 Viewer Mode — Live Market Feed Only")
