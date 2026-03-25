import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Attempt to import optimization library
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False

# --- Page Setup ---
st.set_page_config(page_title="Mag7 Research Terminal", layout="wide")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #00ffc8; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #1e1e1e; border-radius: 5px; color: white; }
    .earnings-highlight { padding: 10px; border-radius: 5px; background-color: #ff4b4b; color: white; font-weight: bold; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- Session & Data Engine ---
def get_session():
    session = Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

custom_session = get_session()

@st.cache_data(ttl=3600)
def fetch_data(ticker_list):
    # Historical Prices
    prices = yf.download(ticker_list, period="5y", multi_level_index=False, session=custom_session)['Close']
    
    # Metadata for all 7
    meta = {}
    for t in ticker_list:
        obj = yf.Ticker(t, session=custom_session)
        info = obj.info
        
        # Earnings Date Fix
        e_date = "N/A"
        raw_ts = info.get('earningsTimestamp') or info.get('nextEarningsDate')
        if raw_ts:
            e_date = pd.to_datetime(raw_ts, unit='s').strftime('%Y-%m-%d')
        
        meta[t] = {
            "Market Cap": info.get('marketCap', 0),
            "P/E": info.get('trailingPE', 'N/A'),
            "Earnings": e_date,
            "Target": info.get('targetMeanPrice', 'N/A')
        }
    return prices, meta

# --- Main App Logic ---
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
st.title("🚀 Mag7 Quant Research & Allocation Terminal")

with st.spinner("Fetching market data..."):
    all_close, all_meta = fetch_data(tickers)

tab1, tab2, tab3 = st.tabs(["📈 Stock Performance", "📊 Mag7 Comparison", "⚖️ Portfolio Optimizer"])

# --- TAB 1: Stock Performance ---
with tab1:
    sel_stock = st.selectbox("Select Ticker for Analysis", tickers)
    period = st.radio("Time Horizon", ["1M", "3M", "6M", "1Y", "5Y"], index=3, horizontal=True)
    
    days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "5Y": 1825}
    data = all_close[sel_stock].tail(days[period])
    
    fig = px.line(data, template="plotly_dark", title=f"{sel_stock} Price Action")
    fig.update_traces(line_color='#00ffc8')
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: Mag7 Comparison ---
with tab2:
    st.subheader("Key Statistics & Next Earnings")
    
    # Create Table Data
    rows = []
    for t in tickers:
        m = all_meta[t]
        rows.append({
            "Ticker": t,
            "Next Earnings": m["Earnings"],
            "Market Cap ($B)": f"{m['Market Cap']/1e9:,.1f}B",
            "P/E Ratio": m["P/E"],
            "1Y Analyst Target": f"${m['Target']}"
        })
    df_compare = pd.DataFrame(rows)
    
    # Custom display with highlighted Earnings
    st.dataframe(df_compare.style.highlight_between(subset=["Next Earnings"], color="#442222"), use_container_width=True)
    
    st.divider()
    st.subheader("Relative Growth (Rebased to 100)")
    norm_df = (all_close / all_close.iloc[0]) * 100
    fig_comp = px.line(norm_df, template="plotly_dark")
    st.plotly_chart(fig_comp, use_container_width=True)

# --- TAB 3: Optimizer ---
with tab3:
    if not OPTIMIZER_AVAILABLE:
        st.error("Please add 'PyPortfolioOpt' to your requirements.txt")
    else:
        st.subheader("Mean-Variance Asset Allocation")
        
        col_in, col_out = st.columns([1, 2])
        
        with col_in:
            st.write("### 1. Your Stock Forecasts")
            # Calculate historical as a baseline
            mu_hist = expected_returns.mean_historical_return(all_close)
            custom_views = {}
            for t in tickers:
                custom_views[t] = st.number_input(f"{t} Exp. Return %", value=float(mu_hist[t]*100), step=1.0)
            
            st.write("### 2. Portfolio Goal")
            target_p_ret = st.slider("Target Portfolio Return %", 5, 100, 20)
            mode = st.toggle("Switch to Max Sharpe (ignores Target %)", value=False)

        with col_out:
            try:
                # Setup Optimizer
                mu = pd.Series({t: v/100 for t, v in custom_views.items()})
                S = risk_models.sample_cov(all_close)
                ef = EfficientFrontier(mu, S)
                ef.add_constraint(lambda w: w <= 0.40) # 40% diversification cap
                
                if mode:
                    weights = ef.max_sharpe()
                    title = "Optimal Portfolio (Max Sharpe)"
                else:
                    weights = ef.efficient_return(target_return=target_p_ret/100)
                    title = f"Lowest Risk for {target_p_ret}% Return"
                
                cleaned_weights = ef.clean_weights()
                ret, vol, sha = ef.portfolio_performance()
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Exp. Return", f"{ret:.1%}")
                m2.metric("Volatility", f"{vol:.1%}")
                m3.metric("Sharpe Ratio", f"{sha:.2f}")
                
                # Chart
                fig_pie = px.pie(
                    names=list(cleaned_weights.keys()), 
                    values=list(cleaned_weights.values()), 
                    hole=0.5, 
                    title=title,
                    template="plotly_dark"
                )
                st.plotly_chart(fig_pie)
                
                st.write("#### Detailed Allocation")
                st.table(pd.Series(cleaned_weights, name="Weight %").apply(lambda x: f"{x:.1%}"))
                
            except Exception as e:
                st.warning("Could not find a valid portfolio for that target. Try lowering your target return.")
