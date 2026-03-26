import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# --- Page Setup & CSS ---
st.set_page_config(page_title="Mag7 Quant Terminal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    div[data-testid="stMetric"] { background-color: #1c2128; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    /* Highlighted Rating Banner */
    .rating-banner { 
        padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;
        border: 2px solid; font-weight: bold; font-size: 24px;
    }
    /* Volatility Alert Styling */
    .vol-alert { color: #d29922; font-weight: bold; font-size: 0.85rem; margin-top: 5px; }
    .stTooltipIcon { color: #58a6ff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Data Engine ---
@st.cache_data(ttl=3600)
def fetch_global_meta(ticker_list):
    prices = yf.download(ticker_list, period="1y", multi_level_index=False)['Close']
    vol_series = prices.pct_change().std() * np.sqrt(252)
    meta_store = {}
    for t in ticker_list:
        obj = yf.Ticker(t)
        info = obj.info
        
        # Rating Logic
        raw_rec = info.get('recommendationKey', 'hold').replace('_', ' ').title()
        
        meta_store[t] = {
            "Current": info.get('currentPrice') or info.get('regularMarketPrice'),
            "Target": info.get('targetMeanPrice', 0),
            "Volatility": vol_series[t],
            "Rating": raw_rec,
            "NextEarnings": "2026-04-23" if t == "GOOGL" else "2026-05-27" if t == "NVDA" else "2026-04-30",
            "PE": info.get('trailingPE', 'N/A'),
            "MarketCap": info.get('marketCap', 0)
        }
    return prices, meta_store

# --- Dashboard Logic ---
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
selected_ticker = st.sidebar.selectbox("Active Security", tickers)

with st.spinner("Analyzing Market Regimes..."):
    all_prices, all_meta = fetch_global_meta(tickers)
    m = all_meta[selected_ticker]

tab1, tab2 = st.tabs(["📈 PERFORMANCE", "📊 COMPARISON"])

with tab1:
    # --- 1. HIGHLIGHTED ANALYST RATING ---
    r_color = "#3fb950" if "Buy" in m['Rating'] else "#d29922" if "Hold" in m['Rating'] else "#f85149"
    st.markdown(f"""
        <div class="rating-banner" style="background-color: {r_color}22; border-color: {r_color}; color: {r_color};">
            RESEARCH CONSENSUS: {m['Rating'].upper()}
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"${m['Current']:.2f}")
    
    # --- 2. VOLATILITY-TRIGGERED EARNINGS HIGHLIGHT ---
    with c2:
        st.write("Next Earnings")
        st.subheader(m['NextEarnings'])
        if m['Volatility'] > 0.35:
            st.markdown('<div class="vol-alert">⚠️ HIGH EVENT RISK (High Vol)</div>', unsafe_allow_html=True)

    c3.metric("Ann. Volatility (σ)", f"{m['Volatility']:.1%}", help="Annualized risk intensity.")
    c4.metric("Market Cap", f"${m['MarketCap']/1e12:.2f}T")

    st.plotly_chart(px.line(all_prices[selected_ticker], template="plotly_dark"), use_container_width=True)
