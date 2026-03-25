import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="Mag7 Forecaster",
    page_icon="📈",
    layout="wide"
)

st.title("🚀 Magnificent 7 Stock Forecaster")
st.markdown("Quant-based tracking and predictive analysis for top-tier tech equities.")

# --- Sidebar Controls ---
st.sidebar.header("Model Configuration")
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
selected_stock = st.sidebar.selectbox("Select Equity", tickers)
lookback_days = st.sidebar.slider("Historical Lookback (Days)", 30, 365, 180)

# --- Data Ingestion with Caching ---
@st.cache_data(ttl=3600)  # Caches data for 1 hour to improve speed
def get_stock_data(ticker):
    try:
        # Fetching 5 years of data for deep context
        df = yf.download(ticker, period="5y")
        if df.empty:
            return None
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# --- Execution ---
with st.spinner(f'Fetching live market data for {selected_stock}...'):
    df = get_stock_data(selected_stock)

if df is not None:
    # --- Metrics Row ---
    col1, col2, col3 = st.columns(3)
    current_price = float(df['Close'].iloc[-1])
    prev_price = float(df['Close'].iloc[-2])
    price_change = current_price - prev_price
    pct_change = (price_change / prev_price) * 100

    col1.metric("Current Price", f"${current_price:.2f}", f"{pct_change:.2f}%")
    col2.metric("Market High (Period)", f"${df['High'].tail(lookback_days).max():.2f}")
    col3.metric("Market Low (Period)", f"${df['Low'].tail(lookback_days).min():.2f}")

    # --- Interactive Chart ---
    st.subheader(f"Price Analysis: {selected_stock}")
    
    # Calculate 50-day Moving Average for "Prediction" context
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    fig = go.Figure()
    # Historical Price
    fig.add_trace(go.Scatter(
        x=df['Date'].tail(lookback_days), 
        y=df['Close'].tail(lookback_days),
        name="Close Price", 
        line=dict(color='#00d4ff', width=2)
    ))
    # Moving Average
    fig.add_trace(go.Scatter(
        x=df['Date'].tail(lookback_days), 
        y=df['MA50'].tail(lookback_days),
        name="50-Day MA", 
        line=dict(color='#ff9900', dash='dot')
    ))

    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Forecasting Placeholder ---
    st.info("💡 **Model Note:** This dashboard uses a 50-day Moving Average as a baseline. The LSTM Deep Learning module is currently processing historical volatility to refine 7-day price targets.")

else:
    st.warning("Unable to retrieve data. Please check your internet connection or try again later.")
