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
    
    /* Next Earnings Banner */
    .earnings-banner {
        background: linear-gradient(180deg, rgba(88, 166, 255, 0.15) 0%, rgba(13, 17, 23, 0) 100%);
        border: 2px solid #58a6ff; border-radius: 12px; padding: 25px; text-align: center; margin-bottom: 20px;
    }
    .earnings-date { color: #58a6ff; font-size: 32px; font-weight: 800; font-family: 'Courier New', monospace; }

    /* Target & Rating Cards */
    .target-card {
        background: #1c2128; border: 1px solid #30363d; border-radius: 12px; padding: 18px;
        text-align: center; margin-bottom: 15px;
    }
    .rating-box { padding: 15px; border-radius: 8px; text-align: center; border: 2px solid; font-weight: bold; margin-bottom: 5px; font-size: 20px; }

    /* Tab 3 Hero Return Card */
    .hero-return-card {
        background: linear-gradient(135deg, #1c2128 0%, #0d1117 100%);
        border: 2px solid #58a6ff; border-radius: 15px; padding: 30px;
        text-align: center; margin-bottom: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Data Engine ---
@st.cache_data(ttl=300)
def fetch_multi_performance(tickers, horizon):
    params = {
        "1D": {"period": "1d", "interval": "1m"}, "5D": {"period": "5d", "interval": "5m"},
        "1M": {"period": "1mo", "interval": "1h"}, "3M": {"period": "3mo", "interval": "1d"},
        "6M": {"period": "6mo", "interval": "1d"}, "YTD": {"period": "ytd", "interval": "1d"},
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
            "Rating": info.get('recommendationKey', 'N/A').replace('_', ' ').upper(),
            "Analysts": info.get('numberOfAnalystOpinions', 'N/A'),
            "DivYield": info.get('dividendYield', 0)
        }
    return prices, meta_store

# --- Initialization ---
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
horizons = ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y"]
selected_ticker = st.sidebar.selectbox("Active Security", tickers)

with st.spinner("Processing Quant Data..."):
    all_prices_5y, all_meta = fetch_global_meta(tickers)
    m = all_meta[selected_ticker]

st.markdown('<div class="terminal-title">MAG7 QUANTITATIVE TERMINAL v1.0</div>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📈 PERFORMANCE", "📊 COMPARISON", "⚖️ OPTIMIZER"])

# --- Tab 1 & Tab 2 (Logic fully intact from previous stable build) ---
with tab1:
    col_l, col_r = st.columns([2.5, 1])
    with col_l:
        h1 = st.radio("TIME HORIZON", horizons, index=5, horizontal=True)
        t1_data = fetch_multi_performance(selected_ticker, h1)
        st.plotly_chart(px.line(t1_data, template="plotly_dark").update_layout(yaxis_title="USD", hovermode="x unified"), use_container_width=True)
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Prev Close", f"${m['PrevClose']:.2f}", help="Previous close price."); v2.metric("EPS (TTM)", f"${m['EPS']:.2f}", help="Earnings per share."); v3.metric("P/E Ratio", f"{m['PE']:.2f}", help="Price to earnings ratio."); v4.metric("Div Yield", f"{m['DivYield']:.2%}", help="Dividend yield.")
    with col_r:
        st.markdown(f'<div class="earnings-banner"><small>NEXT EARNINGS</small><div class="earnings-date">{m["NextEarnings"]}</div></div>', unsafe_allow_html=True)
        r_color = "#3fb950" if "BUY" in m['Rating'] else "#f85149"
        st.markdown(f'<div class="rating-box" style="border-color:{r_color}; color:{r_color};">{m["Rating"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<span style="color:#8b949e; font-size:12px;">Based on {m["Analysts"]} Analysts</span>', unsafe_allow_html=True)
        st.metric("Current Price", f"${m['Current']:.2f}")
        upside = ((m['Target'] / m['Current']) - 1) * 100 if m['Current'] > 0 else 0
        u_color = "#3fb950" if upside >= 0 else "#f85149"
        st.markdown(f'<div class="target-card"><small>1Y TARGET PRICE</small><br><span style="font-size:24px; font-weight:bold;">${m["Target"]:.2f}</span><br><span style="color:{u_color}; font-weight:bold;">{upside:+.1f}% Upside</span></div>', unsafe_allow_html=True)
        st.metric("Beta (β)", f"{m['Beta']:.2f}", help="Market volatility sensitivity."); st.metric("Ann. Volatility (σ)", f"{m['Volatility']:.1%}", help="Annualized risk.")

with tab2:
    h2 = st.radio("COMPARISON HORIZON", horizons, index=5, horizontal=True, key="t2_h")
    comp_data = fetch_multi_performance(tickers, h2)
    norm_df = (comp_data / comp_data.iloc[0]) * 100
    st.plotly_chart(px.line(norm_df, template="plotly_dark", title="Sector Benchmarking").update_layout(yaxis_title="Relative %"), use_container_width=True)
    st.dataframe(pd.DataFrame([{"Ticker": t, "Next Earnings": all_meta[t]['NextEarnings'], "Rating": all_meta[t]['Rating'], "Beta": round(all_meta[t]['Beta'], 2)} for t in tickers]), use_container_width=True)

# --- TAB 3: RESTORED MATH & HERO LAYOUT ---
with tab3:
    if OPTIMIZER_AVAILABLE:
        st.subheader("Modern Portfolio Strategy (MPT)")
        o_in, o_out = st.columns([1, 2.2])
        
        with o_in:
            st.markdown("### 🛠️ Configuration")
            mu_hist = expected_returns.mean_historical_return(all_prices_5y)
            user_views = {t: st.number_input(f"{t} Exp. Return %", value=float(mu_hist[t]*100), step=1.0) for t in tickers}
            target_p = st.slider("Target Portfolio Return %", 5, 100, 25)
            mode = st.toggle("Maximize Sharpe Ratio (Auto-Optimization)")
        
        with o_out:
            try:
                # Calculations
                mu, S = pd.Series({t: v/100 for t, v in user_views.items()}), risk_models.sample_cov(all_prices_5y)
                ef = EfficientFrontier(mu, S)
                weights = ef.max_sharpe() if mode else ef.efficient_return(target_return=target_p/100)
                ret, vol, sha = ef.portfolio_performance()
                
                # Hero Return Card
                ret_color = "#58a6ff" if ret >= 0 else "#f85149"
                st.markdown(f"""
                    <div class="hero-return-card">
                        <small style="color:#8b949e; letter-spacing:2px; text-transform:uppercase; font-weight:bold;">Target Portfolio Return</small>
                        <div style="color:{ret_color}; font-size:48px; font-weight:900; margin-top:10px;">{ret:.1%}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Math Metrics Column
                m1, m2 = st.columns(2)
                m1.metric("Portfolio Volatility (Risk)", f"{vol:.1%}", help="Annualized risk of this allocation.")
                m2.metric("Sharpe Ratio", f"{sha:.2f}", help="Risk-adjusted return (Higher is better).")
                
                # Pie Chart Weights
                st.plotly_chart(px.pie(names=list(ef.clean_weights().keys()), values=list(ef.clean_weights().values()), hole=0.5, template="plotly_dark", title="Optimal Allocation Strategy"), use_container_width=True)
            except: st.warning("Target return is mathematically infeasible based on current inputs.")
