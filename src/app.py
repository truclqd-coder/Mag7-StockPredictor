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

# --- Page Setup ---
st.set_page_config(page_title="Mag7 Quant Terminal", layout="wide")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #00ffc8; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #1e1e1e; border-radius: 5px; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- Data Engine (Let yfinance handle the session internally) ---
@st.cache_data(ttl=3600)
def fetch_global_data(ticker_list):
    # DOWNLOAD: No manual session passed here anymore
    prices = yf.download(ticker_list, period="5y", multi_level_index=False)['Close']
    
    meta_store = {}
    for t in ticker_list:
        obj = yf.Ticker(t)
        info = obj.info
        
        # Earnings Date Fix
        e_date = "N/A"
        raw_ts = info.get('earningsTimestamp') or info.get('nextEarningsDate')
        if raw_ts:
            e_date = pd.to_datetime(raw_ts, unit='s').strftime('%Y-%m-%d')
        
        meta_store[t] = {
            "Market Cap": info.get('marketCap', 0),
            "P/E": info.get('trailingPE', 'N/A'),
            "Next Earnings": e_date,
            "Target": info.get('targetMeanPrice', 'N/A')
        }
    return prices, meta_store

# --- Data Initialization ---
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
tf_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "5Y": 1825}

st.title("🚀 Mag7 Quant Research & Allocation Terminal")
selected_ticker = st.sidebar.selectbox("Primary Focus Ticker", tickers)

with st.spinner("Synchronizing Market Data..."):
    all_close, all_meta = fetch_global_data(tickers)

# --- App Tabs ---
tab1, tab2, tab3 = st.tabs(["📈 Performance Analysis", "📊 Mag7 Comparison", "⚖️ Portfolio Optimizer"])

# --- TAB 1: Performance Analysis ---
with tab1:
    col_l, col_r = st.columns([2, 1])
    with col_l:
        choice1 = st.radio("Chart Horizon:", list(tf_map.keys()), index=1, horizontal=True, key="p_tf")
        plot_data = all_close[selected_ticker].tail(tf_map[choice1])
        
        fig_price = px.line(plot_data, template="plotly_dark", title=f"{selected_ticker} Price Action ({choice1})")
        fig_price.update_traces(line_color='#00ffc8')
        fig_price.update_layout(xaxis_title=None, yaxis_title="Price ($)")
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col_r:
        st.subheader("Asset Intelligence")
        m = all_meta[selected_ticker]
        st.error(f"⚠️ Next Earnings: **{m['Next Earnings']}**")
        st.divider()
        st.metric("Market Capitalization", f"${m['Market Cap']/1e9:,.1f}B")
        st.metric("P/E Ratio (TTM)", m['P/E'])
        st.metric("Analyst 1Y Target", f"${m['Target']}")

# --- TAB 2: Mag7 Comparison ---
with tab2:
    st.subheader("Sector-Wide Benchmarking")
    choice2 = st.radio("Comparison Window:", list(tf_map.keys()), index=1, horizontal=True, key="c_tf")
    comp_df = all_close.tail(tf_map[choice2])
    norm_df = (comp_df / comp_df.iloc[0]) * 100
    
    fig_comp = px.line(norm_df, template="plotly_dark", title=f"Relative Growth Comparison (Base 100)")
    fig_comp.update_layout(yaxis_title="Rebased Price")
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.divider()
    compare_rows = []
    for t in tickers:
        m = all_meta[t]
        compare_rows.append({
            "Ticker": t,
            "Next Earnings": m["Next Earnings"],
            "Market Cap ($B)": round(m['Market Cap']/1e9, 1),
            "P/E Ratio": m["P/E"],
            "1Y Target": m["Target"]
        })
    st.dataframe(pd.DataFrame(compare_rows), use_container_width=True)

# --- TAB 3: Strategic Optimizer ---
with tab3:
    if not OPTIMIZER_AVAILABLE:
        st.error("PyPortfolioOpt is not installed.")
    else:
        st.subheader("Mean-Variance Portfolio Strategy")
        c_in, c_out = st.columns([1, 2])
        with c_in:
            st.write("### 1. Market Convictions")
            mu_hist = expected_returns.mean_historical_return(all_close)
            user_views = {}
            for t in tickers:
                user_views[t] = st.number_input(f"{t} Exp. Return %", value=float(mu_hist[t]*100), step=1.0)
            
            st.write("### 2. Allocation Logic")
            target_p = st.slider("Target Portfolio Return %", 5, 100, 25)
            mode = st.toggle("Maximize Sharpe Ratio", value=False)

        with c_out:
            try:
                mu = pd.Series({t: v/100 for t, v in user_views.items()})
                S = risk_models.sample_cov(all_close)
                ef = EfficientFrontier(mu, S)
                ef.add_constraint(lambda w: w <= 0.40)
                
                weights = ef.max_sharpe() if mode else ef.efficient_return(target_return=target_p/100)
                cleaned = ef.clean_weights()
                ret, vol, sha = ef.portfolio_performance()
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Exp. Return", f"{ret:.1%}")
                m2.metric("Volatility", f"{vol:.1%}")
                m3.metric("Sharpe Ratio", f"{sha:.2f}")
                
                fig_pie = px.pie(names=list(cleaned.keys()), values=list(cleaned.values()), 
                                 hole=0.5, template="plotly_dark", title="Optimal Allocation")
                st.plotly_chart(fig_pie)
                st.table(pd.Series(cleaned, name="Weight").apply(lambda x: f"{x:.1%}" if x > 0 else "0%"))
            except Exception as e:
                st.warning("No valid portfolio for that target.")
