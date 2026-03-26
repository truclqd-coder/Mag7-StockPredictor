import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# --- Page Setup & Bloomberg Theme ---
st.set_page_config(page_title="Mag7 Quant Terminal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    div[data-testid="stMetric"] { background-color: #1c2128; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    /* Analyst Rating Highlight */
    .rating-header { 
        padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 20px;
        border: 2px solid; font-weight: bold; font-size: 22px; letter-spacing: 1px;
    }
    /* Earnings Risk Alert */
    .event-risk { 
        background-color: #d2992222; border: 1px solid #d29922; color: #d29922;
        padding: 5px 10px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; margin-top: 5px;
    }
    .stTooltipIcon { color: #58a6ff !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def fetch_global_meta(ticker_list):
    prices = yf.download(ticker_list, period="1y", multi_level_index=False)['Close']
    vol_series = prices.pct_change().std() * np.sqrt(252)
    meta_store = {}
    for t in ticker_list:
        obj = yf.Ticker(t)
        info = obj.info
        
        # Date Clean-up
        raw_e = info.get('nextEarningsDate') or info.get('earningsTimestamp')
        e_date = pd.to_datetime(raw_e, unit='s').strftime('%Y-%m-%d') if raw_e else "N/A"
        raw_div = info.get('exDividendDate')
        d_date = pd.to_datetime(raw_div, unit='s').strftime('%Y-%m-%d') if raw_div else "N/A"

        meta_store[t] = {
            "Current": info.get('currentPrice') or info.get('regularMarketPrice'),
            "Target": info.get('targetMeanPrice', 0),
            "Rating": info.get('recommendationKey', 'N/A').replace('_', ' ').upper(),
            "NextEarnings": e_date,
            "Volatility": vol_series[t],
            "PE": info.get('trailingPE', 'N/A'),
            "EPS": info.get('trailingEps', 'N/A'),
            "MarketCap": info.get('marketCap', 0),
            "ExDiv": d_date,
            "DivYield": info.get('dividendYield', 0)
        }
    return prices, meta_store

# --- Initialization ---
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
selected_ticker = st.sidebar.selectbox("Active Security", tickers)

with st.spinner("Processing Terminal Data..."):
    all_prices, all_meta = fetch_global_meta(tickers)
    m = all_meta[selected_ticker]

tab1, tab2 = st.tabs(["📈 PERFORMANCE", "📊 COMPARISON"])

with tab1:
    # 1. OUTSTANDING ANALYST RATING BANNER
    r_color = "#3fb950" if "BUY" in m['Rating'] else "#d29922" if "HOLD" in m['Rating'] else "#f85149"
    st.markdown(f"""
        <div class="rating-header" style="border-color: {r_color}; color: {r_color}; background-color: {r_color}11;">
            ANALYST CONSENSUS: {m['Rating']}
        </div>
    """, unsafe_allow_html=True)

    # 2. KEY METRICS ROW (Next Earnings Featured)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"${m['Current']:.2f}")
    
    with c2:
        st.write("Next Earnings Date")
        st.subheader(m['NextEarnings'])
        if m['Volatility'] > 0.35: # High Volatility Threshold
            st.markdown('<div class="event-risk">⚠️ HIGH EVENT RISK</div>', unsafe_allow_html=True)
            
    c3.metric("P/E Ratio", f"{m['PE']:.2f}" if isinstance(m['PE'], (int, float)) else "N/A", help="Price-to-Earnings Ratio.")
    c4.metric("Ann. Volatility (σ)", f"{m['Volatility']:.1%}", help="Annualized risk intensity.")

    # 3. CHART & FUNDAMENTALS
    st.plotly_chart(px.line(all_prices[selected_ticker], template="plotly_dark"), use_container_width=True)
    
    f1, f2, f3 = st.columns(3)
    f1.metric("EPS (TTM)", f"${m['EPS']:.2f}", help="Earnings per share.")
    f2.metric("Market Cap", f"${m['MarketCap']/1e12:.2f}T")
    f3.metric("Ex-Dividend Date", m['ExDiv'], help="Last date to buy for dividend eligibility.")
