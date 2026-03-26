import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- Page Setup ---
st.set_page_config(page_title="Mag7 Quant Terminal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    /* Terminal Header Title */
    .terminal-title {
        color: #58a6ff; font-family: 'Courier New', monospace; font-size: 32px;
        font-weight: bold; border-bottom: 2px solid #30363d; padding-bottom: 10px; margin-bottom: 25px;
    }
    section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    div[data-testid="stMetric"] { background-color: #1c2128; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    
    /* High Visibility Rating Card */
    .rating-box { 
        padding: 20px; border-radius: 10px; text-align: center; 
        border: 2px solid; font-weight: bold; margin-bottom: 15px;
    }
    /* Earnings Highlight */
    .earnings-highlight {
        background-color: #1c2128; border-left: 5px solid #58a6ff;
        padding: 15px; border-radius: 4px; margin-bottom: 15px;
    }
    .event-risk-tag { 
        color: #d29922; font-weight: bold; font-size: 0.75rem; border: 1px solid #d29922;
        padding: 2px 6px; border-radius: 3px; display: inline-block; margin-top: 5px;
    }
    .stTooltipIcon { color: #58a6ff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Data Engine ---
@st.cache_data(ttl=3600)
def fetch_global_meta(ticker_list):
    prices = yf.download(ticker_list, period="5y", multi_level_index=False)['Close']
    vol_series = prices.pct_change().std() * np.sqrt(252)
    ann_return = (prices.iloc[-1] / prices.iloc[-252] - 1) 
    
    meta_store = {}
    for t in ticker_list:
        obj = yf.Ticker(t)
        info = obj.info
        def clean_date(ts):
            return pd.to_datetime(ts, unit='s').strftime('%Y-%m-%d') if ts else "N/A"

        meta_store[t] = {
            "Current": info.get('currentPrice') or info.get('regularMarketPrice'),
            "Target": info.get('targetMeanPrice', 0),
            "Beta": info.get('beta', 1.0),
            "Volatility": vol_series[t],
            "AnnReturn": ann_return[t],
            "PE": info.get('trailingPE', 'N/A'),
            "MarketCap": info.get('marketCap', 0),
            "EPS": info.get('trailingEps', 'N/A'),
            "NextEarnings": clean_date(info.get('nextEarningsDate') or info.get('earningsTimestamp')),
            "ExDiv": clean_date(info.get('exDividendDate')),
            "DivYield": info.get('dividendYield', 0),
            "DivRate": info.get('dividendRate', 0),
            "Rating": info.get('recommendationKey', 'N/A').replace('_', ' ').upper(),
            "Analysts": info.get('numberOfAnalystOpinions', 'N/A')
        }
    return prices, meta_store

# --- Logic ---
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
selected_ticker = st.sidebar.selectbox("Active Security", tickers)

with st.spinner("Streaming Quant Data..."):
    all_prices_5y, all_meta = fetch_global_meta(tickers)
    m = all_meta[selected_ticker]

# --- APP TITLE ---
st.markdown('<div class="terminal-title">MAG7 QUANTITATIVE TERMINAL v2.0</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📈 PERFORMANCE", "📊 COMPARISON"])

with tab1:
    col_l, col_r = st.columns([2.5, 1])

    with col_l:
        # Chart
        fig_price = px.line(all_prices_5y[selected_ticker].tail(252), template="plotly_dark", title=f"{selected_ticker} Trailing 12M")
        fig_price.update_traces(line_color='#58a6ff', line_width=2)
        fig_price.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=450)
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Bottom Fundamentals Row
        st.markdown("### 📊 Valuation & Dividends")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("EPS (TTM)", f"${m['EPS']:.2f}", help="Earnings Per Share (TTM)")
        v2.metric("P/E Ratio", f"{m['PE']:.2f}", help="Price-to-Earnings Ratio")
        v3.metric("Div Yield", f"{m['DivYield']:.2%}", help="Annual Dividend Yield")
        v4.metric("Ex-Div Date", m['ExDiv'], help="Next Ex-Dividend Date")

    with col_r:
        # 1. OUTSTANDING ANALYST RATING
        r_color = "#3fb950" if "BUY" in m['Rating'] else "#d29922" if "HOLD" in m['Rating'] else "#f85149"
        st.markdown(f"""
            <div class="rating-box" style="border-color: {r_color}; color: {r_color}; background-color: {r_color}11;">
                <small style="color: #8b949e; letter-spacing: 1px;">CONSENSUS RATING</small><br>
                {m['Rating']}<br>
                <small style="font-size: 12px;">({m['Analysts']} Analysts)</small>
            </div>
        """, unsafe_allow_html=True)

        # 2. OUTSTANDING EARNINGS DATE
        st.markdown(f"""
            <div class="earnings-highlight">
                <small style="color: #8b949e;">NEXT EARNINGS DATE</small><br>
                <span style="font-size: 24px; font-weight: bold; color: #58a6ff;">{m['NextEarnings']}</span>
                {'<br><span class="event-risk-tag">⚠️ HIGH EVENT RISK</span>' if m['Volatility'] > 0.35 else ''}
            </div>
        """, unsafe_allow_html=True)

        st.metric("Current Price", f"${m['Current']:.2f}")
        
        upside = ((m['Target'] / m['Current']) - 1) * 100 if m['Target'] > 0 else 0
        st.metric("1Y Price Target", f"${m['Target']:.2f}", delta=f"{upside:.1f}% Upside")
        
        st.metric("Beta (β)", f"{m['Beta']:.2f}", help="Sensitivity to S&P 500")
        st.metric("Ann. Volatility (σ)", f"{m['Volatility']:.1%}", help="Total annualized risk")

# --- TAB 2 ---
with tab2:
    st.subheader("Sector Comparison Matrix")
    df_disp = pd.DataFrame([{
        "Ticker": t, "Rating": all_meta[t]['Rating'], "1Y Return": f"{all_meta[t]['AnnReturn']:.1%}", 
        "P/E": all_meta[t]["PE"], "EPS": all_meta[t]["EPS"], "Beta": round(all_meta[t]['Beta'], 2)
    } for t in tickers])
    st.dataframe(df_disp, use_container_width=True)
