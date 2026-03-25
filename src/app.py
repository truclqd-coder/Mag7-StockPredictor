import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pypfopt import EfficientFrontier, risk_models, expected_returns

# --- Page Config & Styling ---
st.set_page_config(page_title="Mag7 Quant Terminal", layout="wide")

# Custom CSS for high-visibility alerts and layout
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.6rem; color: #ff4b4b; }
    .stRadio > div { flex-direction: row; justify-content: flex-start; gap: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Mag7 Research Terminal & Strategic Optimizer")

# --- Global Sidebar ---
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
selected_ticker = st.sidebar.selectbox("Primary Focus Ticker", tickers)

# --- Enhanced Data Engine ---
@st.cache_data(ttl=3600)
def fetch_terminal_data(ticker_list, focus_ticker):
    # Historical Close Prices
    all_data = yf.download(ticker_list, period="5y", multi_level_index=False)['Close']
    
    # Metadata for focus stock
    focus_obj = yf.Ticker(focus_ticker)
    info = focus_obj.info
    
    # Robust Earnings Logic
    try:
        cal = focus_obj.calendar
        next_e = cal.iloc[0, 0].strftime('%Y-%m-%d') if isinstance(cal, pd.DataFrame) else "Check IR"
    except:
        next_e = info.get('nextEarningsDate', "TBD (Post-Reporting)")
        
    return all_data, info, next_e

all_close, main_info, e_date = fetch_terminal_data(tickers, selected_ticker)

# --- Global Components ---
def get_timeframe_selector(key):
    tf_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "5Y": 1825}
    choice = st.radio("Analysis Window:", list(tf_map.keys()), index=1, horizontal=True, key=key)
    return tf_map[choice]

# --- App Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Research", "🔄 Performance", "⚖️ Optimizer", "🔥 Risk Heatmap"])

# --- TAB 1: Equity Research ---
with tab1:
    col_l, col_r = st.columns([2, 1])
    with col_l:
        window = get_timeframe_selector("tf_research")
        fig_price = px.line(all_close[selected_ticker].tail(window), template="plotly_dark", title=f"{selected_ticker} Price Trend")
        fig_price.update_layout(xaxis_title=None, yaxis_title="Price ($)")
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col_r:
        st.subheader("Key Statistics")
        st.metric(label="⚠️ Next Earnings (Volatility Alert)", value=e_date, help="High volatility is expected on reporting dates.")
        st.divider()
        
        # Tooltips added for Financial Engineering definitions
        st.metric("Market Cap", f"${main_info.get('marketCap', 0):,.0f}", 
                  help="Total dollar value of all outstanding shares. P = Price x Shares.")
        
        st.metric("P/E Ratio (TTM)", main_info.get('trailingPE', 'N/A'), 
                  help="Price-to-Earnings ratio. Measures the price paid for $1 of profit.")
        
        st.metric("Beta (5Y)", main_info.get('beta', 'N/A'), 
                  help="Volatility relative to S&P 500. >1.0 means higher risk than the market.")
        
        st.metric("1Y Target Est", f"${main_info.get('targetMeanPrice', 'N/A')}", 
                  help="Average analyst price forecast for the next 12 months.")

# --- TAB 2: Performance Comparison ---
with tab2:
    window = get_timeframe_selector("tf_perf")
    norm_df = (all_close.tail(window) / all_close.tail(window).iloc[0]) * 100
    
    fig_comp = go.Figure()
    for t in tickers:
        is_focus = (t == selected_ticker)
        fig_comp.add_trace(go.Scatter(
            x=norm_df.index, y=norm_df[t], name=t,
            line=dict(width=4 if is_focus else 1, color='white' if is_focus else None),
            opacity=1 if is
