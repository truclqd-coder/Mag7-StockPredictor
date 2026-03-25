import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Mag7 Forecaster",
    page_icon="📈",
    layout="wide"
)

st.title("🚀 Magnificent 7 Stock Forecaster")

# --- Sidebar Controls ---
st.sidebar.header("Navigation & Timeframe")
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
selected_stock = st.sidebar.selectbox("Select Equity", tickers)

# New Optional Timeframes
timeframe_options = {
    "1 Day": 1,
    "5 Days": 5,
    "1 Week": 7,
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "Max (5Y)": 1825
}

selected_label = st.sidebar.selectbox("Historical Lookback", list(timeframe_options.keys()), index=4) # Default to 3 Months
lookback_days = timeframe_options[selected_label]

# --- Data Ingestion with Caching ---
@st.cache_data(ttl=3600) 
def get_stock_data(ticker):
    try:
        # We fetch 5Y of data so calculations like MA200 work even on short view windows
        df = yf.download(ticker, period="5y", multi_level_index=False)
        if df.empty:
            return None
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# --- Execution ---
with st.spinner(f'Fetching live market data...'):
    df = get_stock_data(selected_stock)

if df is not None:
    # Filter data based on selected timeframe
    display_df = df.tail(lookback_days).copy()
    
    # --- Metrics Row ---
    col1, col2, col3 = st.columns(3)
    current_price = float(df['Close'].iloc[-1])
    prev_price = float(df['Close'].iloc[-2])
    price_change = current_price - prev_price
    pct_change = (price_change / prev_price) * 100

    col1.metric("Current Price", f"${current_price:.2f}", f"{pct_change:.2f}%")
    col2.metric(f"High ({selected_label})", f"${float(display_df['High'].max()):.2f}")
    col3.metric(f"Low ({selected_label})", f"${float(display_df['Low'].min()):.2f}")

    # --- Interactive Chart ---
    st.subheader(f"{selected_stock} Performance - {selected_label}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=display_df['Date'], 
        y=display_df['Close'],
        name="Close Price", 
        line=dict(color='#00d4ff', width=2.5)
    ))

    # Add a simple trendline (Moving Average) only for longer timeframes
    if lookback_days > 20:
        display_df['MA20'] = display_df['Close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=display_df['Date'], y=display_df['MA20'],
            name="20-Day MA", line=dict(color='#ff9900', dash='dot')
        ))

    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Data currently unavailable.")
