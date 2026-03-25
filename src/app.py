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
st.markdown("Quant-based tracking and predictive analysis for top-tier tech equities.")

# --- Sidebar Controls ---
st.sidebar.header("Model Configuration")
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
selected_stock = st.sidebar.selectbox("Select Equity", tickers)
lookback_days = st.sidebar.slider("Historical Lookback (Days)", 30, 365, 180)

# --- Data Ingestion with Caching ---
@st.cache_data(ttl=3600) 
def get_stock_data(ticker):
    try:
        # FIX: multi_level_index=False prevents the common TypeError in newer yfinance versions
        df = yf.download(ticker, period="5y", multi_level_index=False)
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
    # Filter data based on sidebar lookback
    display_df = df.tail(lookback_days).copy()
    
    # --- Metrics Row ---
    col1, col2, col3 = st.columns(3)
    
    # Explicitly cast to float to avoid Series-to-String errors
    current_price = float(df['Close'].iloc[-1])
    prev_price = float(df['Close'].iloc[-2])
    price_change = current_price - prev_price
    pct_change = (price_change / prev_price) * 100

    col1.metric("Current Price", f"${current_price:.2f}", f"{pct_change:.2f}%")
    col2.metric("Market High (Period)", f"${float(display_df['High'].max()):.2f}")
    col3.metric("Market Low (Period)", f"${float(display_df['Low'].min()):.2f}")

    # --- Interactive Chart ---
    st.subheader(f"Price Analysis: {selected_stock}")
    
    # Simple Moving Average for visual "prediction" context
    display_df['MA50'] = display_df['Close'].rolling(window=50).mean()
    
    fig = go.Figure()
    # Historical Price
    fig.add_trace(go.Scatter(
        x=display_df['Date'], 
        y=display_df['Close'],
        name="Close Price", 
        line=dict(color='#00d4ff', width=2)
    ))
    # Moving Average
    fig.add_trace(go.Scatter(
        x=display_df['Date'], 
        y=display_df['MA50'],
        name="50-Day MA", 
        line=dict(color='#ff9900', dash='dot')
    ))

    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Forecasting Placeholder ---
    st.info("💡 **Model Note:** This dashboard currently utilizes a 50-day Moving Average (MA50) as a trend baseline. Future updates will integrate an LSTM Deep Learning module for recursive price-target forecasting.")

else:
    st.warning("Unable to retrieve data. Please check your internet connection or try again later.")
