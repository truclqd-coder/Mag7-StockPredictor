import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# Attempt to import optimization library
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False

# --- Page Setup ---
st.set_page_config(page_title="Mag7 Quant Terminal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .terminal-title {
        color: #58a6ff; font-family: 'Courier New', monospace; font-size: 28px;
        font-weight: bold; border-bottom: 2px solid #30363d; padding-bottom: 10px; margin-bottom: 20px;
    }
    div[data-testid="stMetric"] { background-color: #1c2128; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    .target-card { background: linear-gradient(135deg, #1c2128 0%, #0d1117 100%); border: 1px solid #30363d; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- Data Engine ---
@st.cache_data(ttl=300)
def fetch_multi_performance(tickers, horizon):
    # Mapping "YTD" to yfinance native period
    params = {
        "1D": {"period": "1d", "interval": "1m"}, "5D": {"period": "5d", "interval": "5m"},
        "1M": {"period": "1mo", "interval": "1h"}, "3M": {"period": "3mo", "interval": "1d"},
        "6M": {"period": "6mo", "interval": "1d"}, 
        "YTD": {"period": "ytd", "interval": "1d"}, # <--- YTD Logic
        "1Y": {"period": "1y", "interval": "1d"}, "5Y": {"period": "5y", "interval": "1d"}
    }
    config = params.get(horizon, {"period": "1y", "interval": "1d"})
    return yf.download(tickers, **config, multi_level_index=False)['Close']

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
            "Current": info.get('currentPrice', 0),
            "PrevClose": info.get('previousClose', 0),
            "Target": info.get('targetMeanPrice', 0),
            "Beta": info.get('beta', 1.0),
            "Volatility": vol_series[t],
            "PE": info.get('trailingPE', 0),
            "EPS": info.get('trailingEps', 0),
            "NextEarnings": clean_date(info.get('nextEarningsDate') or info.get('earningsTimestamp')),
            "Rating": info.get('recommendationKey', 'N/A').replace('_', ' ').upper()
        }
    return prices, meta_store

# --- Initialization ---
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
horizons = ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y"] # <--- YTD in List
selected_ticker = st.sidebar.selectbox("Active Security", tickers)

with st.spinner("Loading Terminal v1.0..."):
    all_prices_5y, all_meta = fetch_global_meta(tickers)
    m = all_meta[selected_ticker]

st.markdown('<div class="terminal-title">MAG7 QUANTITATIVE TERMINAL v1.0</div>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📈 PERFORMANCE", "📊 COMPARISON", "⚖️ OPTIMIZER"])

# --- TAB 1 ---
with tab1:
    h1 = st.radio("TIME HORIZON", horizons, index=5, horizontal=True) # Default to YTD (index 5)
    t1_data = fetch_multi_performance(selected_ticker, h1)
    st.plotly_chart(px.line(t1_data, template="plotly_dark").update_layout(yaxis_title="Price (USD)"), use_container_width=True)
    
    st.markdown("### 📊 Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prev Close", f"${m['PrevClose']:.2f}")
    c2.metric("EPS (TTM)", f"${m['EPS']:.2f}", help="Earnings Per Share (Net Income/Shares)")
    c3.metric("P/E Ratio", f"{m['PE']:.2f}", help="Valuation multiple (Price/Earnings)")
    c4.metric("Beta (β)", f"{m['Beta']:.2f}", help="Market Sensitivity")

# --- TAB 2 ---
with tab2:
    h2 = st.radio("COMPARISON HORIZON", horizons, index=5, horizontal=True)
    comp_data = fetch_multi_performance(tickers, h2)
    norm_df = (comp_data / comp_data.iloc[0]) * 100
    st.plotly_chart(px.line(norm_df, template="plotly_dark").update_layout(yaxis_title="Normalized %"), use_container_width=True)
    
    st.dataframe(pd.DataFrame([{
        "Ticker": t, "Next Earnings": all_meta[t]['NextEarnings'], 
        "Rating": all_meta[t]['Rating'], "Beta": round(all_meta[t]['Beta'], 2)
    } for t in tickers]), use_container_width=True)

# --- TAB 3 ---
with tab3:
    if OPTIMIZER_AVAILABLE:
        st.subheader("Allocation Strategy")
        o_in, o_out = st.columns([1, 2.2])
        with o_in:
            mu_hist = expected_returns.mean_historical_return(all_prices_5y)
            user_views = {t: st.number_input(f"{t} Exp. Return %", value=float(mu_hist[t]*100)) for t in tickers}
            target_p = st.slider("Target Return %", 5, 100, 25)
        with o_out:
            try:
                mu, S = pd.Series({t: v/100 for t, v in user_views.items()}), risk_models.sample_cov(all_prices_5y)
                ef = EfficientFrontier(mu, S)
                weights = ef.efficient_return(target_return=target_p/100)
                ret, vol, sha = ef.portfolio_performance()
                
                ret_color = "#3fb950" if ret >= 0 else "#f85149"
                st.markdown(f'<div class="target-card" style="border-color: {ret_color};"><small>TARGET RETURN</small><div style="color: {ret_color}; font-size: 38px; font-weight: bold;">{ret:.1% }</div></div>', unsafe_allow_html=True)
                st.plotly_chart(px.pie(names=list(ef.clean_weights().keys()), values=list(ef.clean_weights().values()), hole=0.5, template="plotly_dark"), use_container_width=True)
            except: st.warning("Infeasible target.")
