import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pypfopt import EfficientFrontier, risk_models, expected_returns

# --- Page Setup ---
st.set_page_config(page_title="Mag7 Quant Terminal", layout="wide")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.6rem; color: #ff4b4b; }
    .stRadio > div { flex-direction: row; justify-content: flex-start; gap: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Mag7 Research Terminal & Strategic Optimizer")

# --- Global Sidebar ---
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
selected_ticker = st.sidebar.selectbox("Primary Focus Ticker", tickers)

# --- HARDENED EARNINGS ENGINE ---
@st.cache_data(ttl=3600)
def fetch_terminal_data(ticker_list, focus_ticker):
    all_data = yf.download(ticker_list, period="5y", multi_level_index=False)['Close']
    focus_obj = yf.Ticker(focus_ticker)
    info = focus_obj.info
    
    next_e = "N/A"
    try:
        raw_ts = info.get('earningsTimestamp') or info.get('nextEarningsDate')
        if raw_ts:
            next_e = pd.to_datetime(raw_ts, unit='s').strftime('%Y-%m-%d')
        if next_e == "N/A":
            e_df = focus_obj.get_earnings_dates(limit=1)
            if e_df is not None and not e_df.empty:
                next_e = e_df.index[0].strftime('%Y-%m-%d')
    except:
        next_e = "Check IR Calendar"
        
    return all_data, info, next_e

all_close, main_info, e_date = fetch_terminal_data(tickers, selected_ticker)
tf_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "5Y": 1825}

# --- App Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Research", "🔄 Performance", "⚖️ Optimizer"])

# --- TAB 1: Equity Research ---
with tab1:
    col_l, col_r = st.columns([2, 1])
    with col_l:
        choice1 = st.radio("Chart Window:", ["1M", "3M", "6M", "1Y", "5Y"], index=1, horizontal=True, key="r_chart")
        plot_df = all_close[selected_ticker].tail(tf_map[choice1])
        fig_price = px.line(plot_df, template="plotly_dark", title=f"{selected_ticker} ({choice1})")
        fig_price.update_layout(xaxis_title=None, yaxis_title="Price ($)")
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col_r:
        st.subheader("Key Statistics")
        st.error(f"⚠️ Next Earnings: **{e_date}**")
        st.divider()
        st.metric("Market Cap", f"${main_info.get('marketCap', 0):,.0f}")
        st.metric("P/E Ratio", main_info.get('trailingPE', 'N/A'))
        st.metric("Beta (5Y)", main_info.get('beta', 'N/A'))
        st.metric("1Y Target", f"${main_info.get('targetMeanPrice', 'N/A')}")

# --- TAB 2: Performance Comparison
