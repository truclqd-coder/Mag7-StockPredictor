import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page Config
st.set_page_config(page_title="Mag7 Predictor", layout="wide")

st.title("🚀 Magnificent 7 Stock Predictor")
st.markdown("Real-time tracking and predictive analysis of the world's tech giants.")

# Sidebar - Settings
st.sidebar.header("Control Panel")
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
selected_stock = st.sidebar.selectbox("Select a Stock", tickers)
days_to_predict = st.sidebar.slider("Prediction Horizon (Days)", 1, 30, 7)

# 1. Fetch Data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start="2020-01-01", end=datetime.now().strftime('%Y-%m-%d'))
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
df = load_data(selected_stock)
data_load_state.text('Data Loaded!')

# 2. Layout - Top Metrics
col1, col2, col3 = st.columns(3)
current_price = df['Close'].iloc[-1]
prev_price = df['Close'].iloc[-2]
delta = ((current_price - prev_price) / prev_price) * 100

col1.metric("Current Price", f"${current_price:.2f}", f"{delta:.2f}%")
col2.metric("52-Week High", f"${df['High'].max():.2f}")
col3.metric("52-Week Low", f"${df['Low'].min():.2f}")

# 3. Main Chart
st.subheader(f"{selected_stock} Historical Performance")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Actual Price", line=dict(color='#00ffcc')))
fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=True)
st.plotly_chart(fig, use_container_width=True)

# 4. Prediction Logic (Simplified Placeholder for LSTM)
st.subheader("🔮 Future Return Forecast")
# Note: In a full project, you'd load your saved Keras (.h5) model here.
last_60_days = df['Close'].tail(60).mean()
prediction = last_60_days * (1 + (delta/100)) # Simple trend projection for demo

st.write(f"Based on recent volatility, the estimated price in {days_to_predict} days is approximately:")
st.header(f"${prediction:.2f}")

# 5. Technical Indicators Table
st.subheader("Key Technicals")
st.dataframe(df.tail(10), use_container_width=True)
