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

# --- TRIPLE-LAYER EARNINGS ENGINE ---
@st.cache_data(ttl=3600)
def fetch_terminal_data(ticker_list, focus_ticker):
    all_data = yf.download(ticker_list, period="5y", multi_level_index=False)['Close']
    focus_obj = yf.Ticker(focus_ticker)
    info = focus_obj.info
    
    next_e = "N/A"
    try:
        # Layer 1: Earnings Dates Method
        earning_df = focus_obj.get_earnings_dates(limit=1)
        if earning_df is not None and not earning_df.empty:
            next_e = earning_df.index[0].strftime('%Y-%m-%d')
        
        # Layer 2: Calendar Fallback
        if next_e == "N/A":
            cal = focus_obj.calendar
            if isinstance(cal, pd.DataFrame) and not cal.empty:
                next_e = cal.iloc[0, 0].strftime('%Y-%m-%d')
                
        # Layer 3: Info Metadata Fallback
        if next_e == "N/A":
            raw_e = info.get('nextEarningsDate')
            if raw_e:
                next_e = pd.to_datetime(raw_e, unit='s').strftime('%Y-%m-%d')
    except:
        next_e = "Check IR Calendar"
        
    return all_data, info, next_e

all_close, main_info, e_date = fetch_terminal_data(tickers, selected_ticker)
tf_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "5Y": 1825}

# --- App Tabs (Heatmap Removed) ---
tab1, tab2, tab3 = st.tabs(["📊 Research", "🔄 Performance", "⚖️ Optimizer"])

# --- TAB 1: Equity Research ---
with tab1:
    col_l, col_r = st.columns([2, 1])
    with col_l:
        # Timeframe Selector directly above the chart
        tf_choice
