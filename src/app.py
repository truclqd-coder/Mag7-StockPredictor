import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Attempt to import optimization library
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False

# --- Page Setup & THEME ---
st.set_page_config(page_title="Mag7 Quant Terminal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .terminal-title {
        color: #58a6ff; font-family: 'Courier New', monospace; font-size: 28px;
        font-weight: bold; border-bottom: 2px solid #30363d; padding-bottom: 10px; margin-bottom: 20px;
    }
    section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    div[data-testid="stMetric"] { background-color: #1c2128; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    .rating-box { padding: 18px; border-radius: 8px; text-align: center; border: 2px solid; font-weight: bold; margin-bottom: 15px; }
    .earnings-highlight { background-color: #1c2128; border-left: 5px solid #58a6ff; padding: 15px; border-radius: 4px; margin-bottom: 15px; }
    .event-risk-tag { color: #d29922; font-weight: bold; font-size: 0.75rem; border: 1px solid #d29922; padding: 2px 6px; border-radius: 3px; display: inline-block; margin-top: 5px; }
    .stTooltipIcon { color: #58a6ff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Data Engine ---
@st.cache_data(ttl=300)
def fetch_multi_performance(tickers, horizon):
    params = {
        "1D": {"period": "1d", "interval": "1m"}, "5D": {"period": "5d", "interval": "5m"},
        "1M": {"period": "1mo", "interval": "1h"}, "3M": {"period": "3mo", "interval": "1d"},
        "6M": {"period": "6mo", "interval": "1d"}, "1Y": {"period": "1y", "interval": "1d"},
        "5Y": {"period": "5y", "interval": "1d"}
    }
    config = params.get(horizon, {"period": "1y", "interval": "1d"})
    data = yf.download(tickers, **config, multi_level_index=False)['Close']
    return data

@st.cache_data(ttl=3600)
def fetch_global_meta(ticker_list):
    prices = yf.download(ticker_list, period="5y", multi_level_index=False)['Close']
    vol_series = prices.pct_change().std() * np.sqrt(252)
    meta_store = {}
    for t in ticker_list:
        obj = yf.Ticker(t)
        info = obj.info
        def clean_date(ts): return pd.to_datetime(ts, unit='s').strftime('%Y-%m-%d') if ts else "N/A"
        meta_store[t] = {
            "Current": info.get('currentPrice') or info.get('regularMarketPrice'),
            "Target": info.get('targetMeanPrice', 0),
            "Beta": info.get('beta', 1.0),
            "Volatility": vol_series[t],
            "PE": info.get('trailingPE', 'N/A'),
            "MarketCap": info.get('marketCap', 0),
            "EPS": info.get('trailingEps', 'N/A'),
            "NextEarnings": clean_date(info.get('nextEarningsDate') or info.get('earningsTimestamp')),
            "ExDiv": clean_date(info.get('exDividendDate')),
            "DivYield": info.get('dividendYield', 0),
            "Rating": info.get('recommendationKey', 'N/A').replace('_', ' ').upper(),
            "Analysts": info.get('numberOfAnalystOpinions', 'N/A')
        }
    return prices, meta_store

# --- Initialization ---
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
horizons = ["1D", "5D", "1M", "3M", "6M", "1Y", "5Y"]
selected_ticker = st.sidebar.selectbox("Active Security", tickers)

with st.spinner("Processing Terminal Data..."):
    all_prices_5y, all_meta = fetch_global_meta(tickers)
    m = all_meta[selected_ticker]

st.markdown('<div class="terminal-title">MAG7 QUANTITATIVE TERMINAL v2.0</div>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📈 PERFORMANCE", "📊 COMPARISON", "⚖️ OPTIMIZER"])

# --- TAB 1: Performance ---
with tab1:
    col_l, col_r = st.columns([2.5, 1])
    with col_l:
        h1 = st.radio("TIME HORIZON", horizons, index=3, horizontal=True, key="t1_horizon")
        t1_data = fetch_multi_performance(selected_ticker, h1)
        fig1 = px.line(t1_data, template="plotly_dark", title=f"{selected_ticker} Price History")
        fig1.update_traces(line_color='#58a6ff', line_width=2)
        fig1.update_layout(yaxis_title="Market Price (USD)", xaxis_title="Date/Time", hovermode="x unified")
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("### 📊 Valuation & Dividends")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("EPS (TTM)", f"${m['EPS']:.2f}", help="Earnings Per Share: The portion of a company's profit allocated to each outstanding share of common stock.")
        v2.metric("P/E Ratio", f"{m['PE']:.2f}", help="Price-to-Earnings Ratio: Measures a company's current share price relative to its earnings per share.")
        v3.metric("Div Yield", f"{m['DivYield']:.2%}", help="Dividend Yield: A financial ratio that tells you the percentage of a company's share price that it pays out in dividends each year.")
        v4.metric("Ex-Div Date", m['ExDiv'], help="Ex-Dividend Date: You must own the stock before this date to receive the next scheduled dividend payment.")

    with col_r:
        r_color = "#3fb950" if "BUY" in m['Rating'] else "#d29922" if "HOLD" in m['Rating'] else "#f85149"
        st.markdown(f'<div class="rating-box" style="border-color: {r_color}; color: {r_color}; background-color: {r_color}11;">{m["Rating"]}<br><small>({m["Analysts"]} Analysts)</small></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="earnings-highlight"><small>NEXT EARNINGS</small><br><span style="font-size: 22px; font-weight: bold; color: #58a6ff;">{m["NextEarnings"]}</span>{"<br><span class=event-risk-tag>⚠️ HIGH EVENT RISK</span>" if m["Volatility"] > 0.35 else ""}</div>', unsafe_allow_html=True)
        
        st.metric("Current Price", f"${m['Current']:.2f}")
        upside = ((m['Target'] / m['Current']) - 1) * 100 if m['Target'] > 0 else 0
        st.metric("1Y Price Target", f"${m['Target']:.2f}", delta=f"{upside:.1f}% Upside", help="Average 12-month analyst price estimate.")
        st.metric("Beta (β)", f"{m['Beta']:.2f}", help="Beta: A measure of a stock's volatility in relation to the overall market (S&P 500).")
        st.metric("Ann. Volatility (σ)", f"{m['Volatility']:.1%}", help="Annualized Volatility: The standard deviation of the stock's returns over a year, representing price risk.")

# --- TAB 2 & 3 logic remains exactly as per previous stable version ---
# ... (Tab 2 Comparison & Tab 3 Optimizer code)
