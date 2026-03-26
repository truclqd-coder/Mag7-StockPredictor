import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Attempt to import optimization library
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False

# --- Page Setup & CSS Styling ---
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
    
    .target-card {
        background: linear-gradient(135deg, #1c2128 0%, #0d1117 100%);
        border: 1px solid #30363d; border-radius: 12px; padding: 20px;
        text-align: center; margin-bottom: 20px;
    }
    .rating-box { padding: 18px; border-radius: 8px; text-align: center; border: 2px solid; font-weight: bold; margin-bottom: 15px; }
    .earnings-highlight { background-color: #1c2128; border-left: 5px solid #58a6ff; padding: 15px; border-radius: 4px; margin-bottom: 15px; }
    .stTooltipIcon { color: #58a6ff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Data Engine ---
@st.cache_data(ttl=300)
def fetch_multi_performance(tickers, horizon):
    params = {
        "1D": {"period": "1d", "interval": "1m"}, "5D": {"period": "5d", "interval": "5m"},
        "1M": {"period": "1mo", "interval": "1h"}, "3M": {"period": "3mo", "interval": "1d"},
        "6M": {"period": "6mo", "interval": "1d"}, 
        "YTD": {"period": "ytd", "interval": "1d"},
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
            "ExDiv": clean_date(info.get('exDividendDate')),
            "DivYield": info.get('dividendYield', 0),
            "Rating": info.get('recommendationKey', 'N/A').replace('_', ' ').upper(),
            "Analysts": info.get('numberOfAnalystOpinions', 'N/A')
        }
    return prices, meta_store

# --- Initialization ---
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
horizons = ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y"]
selected_ticker = st.sidebar.selectbox("Active Security", tickers)

with st.spinner("Re-syncing Quant Data..."):
    all_prices_5y, all_meta = fetch_global_meta(tickers)
    m = all_meta[selected_ticker]

st.markdown('<div class="terminal-title">MAG7 QUANTITATIVE TERMINAL v1.0</div>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📈 PERFORMANCE", "📊 COMPARISON", "⚖️ OPTIMIZER"])

# --- TAB 1: ALL METRICS RESTORED ---
with tab1:
    col_l, col_r = st.columns([2.5, 1])
    with col_l:
        h1 = st.radio("TIME HORIZON", horizons, index=5, horizontal=True, key="t1_horizon")
        t1_data = fetch_multi_performance(selected_ticker, h1)
        st.plotly_chart(px.line(t1_data, template="plotly_dark", title=f"{selected_ticker} Price Path").update_layout(yaxis_title="Price (USD)", hovermode="x unified"), use_container_width=True)
        
        st.markdown("### 📊 Market Depth & Fundamentals")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Prev Close", f"${m['PrevClose']:.2f}", help="Last session closing price.")
        v2.metric("EPS (TTM)", f"${m['EPS']:.2f}", help="Earnings Per Share (Last 12 Mo).")
        v3.metric("P/E Ratio", f"{m['PE']:.2f}", help="Price-to-Earnings Ratio.")
        v4.metric("Div Yield", f"{m['DivYield']:.2%}", help="Annual Dividend Yield.")

    with col_r:
        r_color = "#3fb950" if "BUY" in m['Rating'] else "#d29922" if "HOLD" in m['Rating'] else "#f85149"
        st.markdown(f'<div class="rating-box" style="border-color: {r_color}; color: {r_color};">{m["Rating"]}<br><small>({m["Analysts"]} Analysts)</small></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="earnings-highlight"><small>NEXT EARNINGS</small><br><span style="color: #58a6ff; font-weight: bold;">{m["NextEarnings"]}</span></div>', unsafe_allow_html=True)
        
        st.metric("Current Price", f"${m['Current']:.2f}")
        
        # Upside logic with color
        upside = ((m['Target'] / m['Current']) - 1) * 100 if m['Current'] > 0 else 0
        u_color = "#3fb950" if upside >= 0 else "#f85149"
        st.markdown(f'<div style="background:#1c2128; padding:15px; border-radius:8px; border:1px solid #30363d;"><small style="color:#8b949e;">1Y TARGET UPSIDE</small><br><span style="color:{u_color}; font-size:20px; font-weight:bold;">{upside:.1f}%</span></div>', unsafe_allow_html=True)
        
        st.metric("Beta (β)", f"{m['Beta']:.2f}", help="Market sensitivity (1.0 = S&P 500).")
        st.metric("Ann. Volatility (σ)", f"{m['Volatility']:.1%}", help="Annualized risk (Standard Deviation).")

# --- TAB 2: ALL COMPARISONS RESTORED ---
with tab2:
    st.subheader("Relative Sector Benchmarking")
    h2 = st.radio("COMPARISON HORIZON", horizons, index=5, horizontal=True, key="t2_horizon")
    comp_data = fetch_multi_performance(tickers, h2)
    norm_df = (comp_data / comp_data.iloc[0]) * 100
    st.plotly_chart(px.line(norm_df, template="plotly_dark", title="Mag7 Normalized (Base 100)").update_layout(yaxis_title="Performance %"), use_container_width=True)
    
    # Summary Table with Earnings Dates
    summary_data = [{
        "Ticker": t,
        "Next Earnings": all_meta[t]['NextEarnings'],
        "Rating": all_meta[t]['Rating'],
        "P/E Ratio": all_meta[t]['PE'],
        "Beta": round(all_meta[t]['Beta'], 2),
        "Volatility": f"{all_meta[t]['Volatility']:.1%}"
    } for t in tickers]
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# --- TAB 3: STRATEGIC OPTIMIZER RESTORED ---
with tab3:
    if OPTIMIZER_AVAILABLE:
        st.subheader("Mean-Variance Portfolio Allocation")
        o_in, o_out = st.columns([1, 2.2])
        with o_in:
            mu_hist = expected_returns.mean_historical_return(all_prices_5y)
            user_views = {t: st.number_input(f"{t} Exp. Return %", value=float(mu_hist[t]*100), step=1.0) for t in tickers}
            target_p = st.slider("Target Portfolio Return %", 5, 100, 25)
            mode = st.toggle("Maximize Sharpe Ratio")
        with o_out:
            try:
                mu, S = pd.Series({t: v/100 for t, v in user_views.items()}), risk_models.sample_cov(all_prices_5y)
                ef = EfficientFrontier(mu, S)
                weights = ef.max_sharpe() if mode else ef.efficient_return(target_return=target_p/100)
                ret, vol, sha = ef.portfolio_performance()
                
                # Portfolio Return logic with color
                ret_color = "#3fb950" if ret >= 0 else "#f85149"
                st.markdown(f'<div class="target-card" style="border-color: {ret_color};"><small style="color: #8b949e;">OPTIMIZED PORTFOLIO RETURN</small><div style="color: {ret_color}; font-size: 38px; font-weight: bold;">{ret:.1% }</div></div>', unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                c1.metric("Risk (Volatility)", f"{vol:.1%}")
                c2.metric("Sharpe Ratio", f"{sha:.2f}")
                st.plotly_chart(px.pie(names=list(ef.clean_weights().keys()), values=list(ef.clean_weights().values()), hole=0.5, template="plotly_dark", title="Optimal Weights"), use_container_width=True)
            except: st.warning("Target return is infeasible—lower your expectations.")
