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

# --- Page Setup & THEME OVERRIDE ---
st.set_page_config(page_title="Mag7 Quant Terminal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: #0e1117; padding: 10px 0; }
    .stTabs [data-baseweb="tab"] { 
        height: 45px; background-color: #1c2128; border-radius: 4px 4px 0 0; 
        color: #8b949e; font-weight: 600; border: 1px solid #30363d; padding: 0 20px;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #21262d !important; color: #58a6ff !important; border-bottom: 3px solid #58a6ff !important;
    }
    div[data-testid="stMetric"] { background-color: #1c2128; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    div[data-testid="stMetricValue"] { color: #3fb950; font-family: 'Courier New', monospace; }
    .rating-card { background-color: #1c2128; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin-top: 10px; }
    .stTooltipIcon { color: #58a6ff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Data Engine ---
@st.cache_data(ttl=3600)
def fetch_global_meta(ticker_list):
    prices = yf.download(ticker_list, period="5y", multi_level_index=False)['Close']
    vol_series = prices.pct_change().std() * np.sqrt(252)
    
    meta_store = {}
    for t in ticker_list:
        obj = yf.Ticker(t)
        info = obj.info
        
        # Date Formatting
        def fmt_date(ts):
            return pd.to_datetime(ts, unit='s').strftime('%Y-%m-%d') if ts else "N/A"

        meta_store[t] = {
            "Current": info.get('currentPrice') or info.get('regularMarketPrice'),
            "Target": info.get('targetMeanPrice', 0),
            "Beta": info.get('beta', 1.0),
            "Volatility": vol_series[t],
            "PE": info.get('trailingPE', 'N/A'),
            "MarketCap": info.get('marketCap', 0),
            "EPS": info.get('trailingEps', 'N/A'),
            "NextEarnings": fmt_date(info.get('nextEarningsDate') or info.get('earningsTimestamp')),
            "ExDiv": fmt_date(info.get('exDividendDate')),
            "DivYield": info.get('dividendYield', 0),
            "DivRate": info.get('dividendRate', 0),
            "Rating": info.get('recommendationKey', 'N/A').replace('_', ' ').title(),
            "Analysts": info.get('numberOfAnalystOpinions', 'N/A')
        }
    return prices, meta_store

# --- Initialization ---
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
selected_ticker = st.sidebar.selectbox("Active Security", tickers)

with st.spinner("Synchronizing Market Intelligence..."):
    all_prices_5y, all_meta = fetch_global_meta(tickers)

tab1, tab2, tab3 = st.tabs(["📈 PERFORMANCE", "📊 SECTOR COMPARISON", "⚖️ STRATEGIC OPTIMIZER"])

# --- TAB 1: Performance Analysis ---
with tab1:
    m = all_meta[selected_ticker]
    
    # 1. PRIMARY METRICS ROW
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"${m['Current']:.2f}", help="Last traded market price.")
    
    upside = ((m['Target'] / m['Current']) - 1) * 100 if m['Target'] > 0 else 0
    c2.metric("1Y Price Target", f"${m['Target']:.2f}", delta=f"{upside:.1f}% Upside", 
              help="Average 12-month analyst estimate.")
    
    c3.metric("Next Earnings", m['NextEarnings'], 
              help="The next scheduled date for the company to report quarterly financial results.")
    
    c4.metric("Market Cap", f"${m['MarketCap']/1e12:.2f}T", 
              help="Total market value of all outstanding shares (Market Price x Total Shares).")

    # 2. CHART SECTION
    fig = px.line(all_prices_5y[selected_ticker].tail(252), template="plotly_dark", title=f"{selected_ticker} 12-Month Trend")
    fig.update_traces(line_color='#58a6ff', line_width=2)
    st.plotly_chart(fig, use_container_width=True)

    # 3. VALUATION & DIVIDENDS ROW
    st.markdown("### 📊 Fundamental Deep Dive")
    v1, v2, v3, v4 = st.columns(4)
    
    v1.metric("P/E Ratio", f"{m['PE']:.2f}" if isinstance(m['PE'], (int, float)) else "N/A", 
              help="Price-to-Earnings Ratio. Shows how much investors pay for $1 of profit.")
    
    v2.metric("EPS (TTM)", f"${m['EPS']:.2f}" if isinstance(m['EPS'], (int, float)) else "N/A", 
              help="Earnings Per Share. The portion of profit allocated to each outstanding share.")
    
    div_display = f"{m['DivRate']} ({m['DivYield']:.2%})" if m['DivYield'] > 0 else "0.00 (0%)"
    v3.metric("Forward Div & Yield", div_display, 
              help="The estimated annual dividend payment and the percentage return based on current price.")
    
    v4.metric("Ex-Dividend Date", m['ExDiv'], 
              help="The date by which you must own the stock to receive the next dividend payment.")

    # 4. RATINGS SECTION
    st.divider()
    st.markdown("### 🏛️ Analyst Rating (Consensus)")
    rating_color = "#3fb950" if "Buy" in m['Rating'] else "#d29922"
    st.markdown(f"""
        <div class="rating-card">
            <span style="color: {rating_color}; font-size: 22px; font-weight: bold;">{m['Rating']}</span>
            <span style="color: #8b949e; margin-left: 20px;">Based on {m['Analysts']} Analysts</span>
        </div>
    """, unsafe_allow_html=True)

# --- TAB 2: Mag7 Comparison ---
with tab2:
    st.subheader("Sector Benchmark Comparison")
    df_disp = pd.DataFrame([{
        "Ticker": t, 
        "Price": f"${all_meta[t]['Current']:.2f}",
        "P/E": all_meta[t]["PE"],
        "EPS": all_meta[t]["EPS"],
        "Div Yield": f"{all_meta[t]['DivYield']:.2%}",
        "Beta": round(all_meta[t]['Beta'], 2),
        "Vol %": f"{all_meta[t]['Volatility']:.1%}"
    } for t in tickers])
    st.dataframe(df_disp, use_container_width=True)
