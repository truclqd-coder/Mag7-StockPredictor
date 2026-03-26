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

# --- Page Setup & Bloomberg Theme ---
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
    
    .rating-box { 
        padding: 18px; border-radius: 8px; text-align: center; 
        border: 2px solid; font-weight: bold; margin-bottom: 15px;
    }
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
@st.cache_data(ttl=300)
def fetch_performance_data(ticker, horizon):
    params = {
        "1D": {"period": "1d", "interval": "1m"}, "5D": {"period": "5d", "interval": "5m"},
        "1M": {"period": "1mo", "interval": "1h"}, "3M": {"period": "3mo", "interval": "1d"},
        "6M": {"period": "6mo", "interval": "1d"}, "1Y": {"period": "1y", "interval": "1d"},
        "5Y": {"period": "5y", "interval": "1d"}
    }
    config = params.get(horizon, {"period": "1y", "interval": "1d"})
    return yf.download(ticker, **config, multi_level_index=False)['Close']

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
horizons = ["1D", "5D", "1M", "3M", "6M", "1Y", "5Y"]
selected_ticker = st.sidebar.selectbox("Active Security", tickers)

with st.spinner("Streaming Market Intelligence..."):
    all_prices_5y, all_meta = fetch_global_meta(tickers)
    m = all_meta[selected_ticker]

st.markdown('<div class="terminal-title">MAG7 QUANTITATIVE TERMINAL v2.0</div>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📈 PERFORMANCE", "📊 COMPARISON", "⚖️ OPTIMIZER"])

# --- TAB 1: Performance ---
with tab1:
    col_l, col_r = st.columns([2.5, 1])
    with col_l:
        # Restore Time Horizon Selector
        choice1 = st.radio("TIME HORIZON", horizons, index=3, horizontal=True, key="p_tf")
        plot_data = fetch_performance_data(selected_ticker, choice1)
        fig_price = px.line(plot_data, template="plotly_dark", title=f"{selected_ticker} Price Action ({choice1})")
        fig_price.update_traces(line_color='#58a6ff', line_width=2)
        fig_price.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
        st.plotly_chart(fig_price, use_container_width=True)
        
        st.markdown("### 📊 Valuation & Dividends")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("EPS (TTM)", f"${m['EPS']:.2f}", help="Earnings Per Share (TTM)")
        v2.metric("P/E Ratio", f"{m['PE']:.2f}", help="Price-to-Earnings Ratio")
        v3.metric("Div Yield", f"{m['DivYield']:.2%}", help="Annual Yield")
        v4.metric("Ex-Div Date", m['ExDiv'], help="Next Ex-Dividend Date")

    with col_r:
        # High Visibility Rating & Earnings
        r_color = "#3fb950" if "BUY" in m['Rating'] else "#d29922" if "HOLD" in m['Rating'] else "#f85149"
        st.markdown(f'<div class="rating-box" style="border-color: {r_color}; color: {r_color}; background-color: {r_color}11;">{m["Rating"]}<br><small>({m["Analysts"]} Analysts)</small></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="earnings-highlight"><small>NEXT EARNINGS</small><br><span style="font-size: 22px; font-weight: bold; color: #58a6ff;">{m["NextEarnings"]}</span>{"<br><span class=event-risk-tag>⚠️ HIGH EVENT RISK</span>" if m["Volatility"] > 0.35 else ""}</div>', unsafe_allow_html=True)
        
        st.metric("Current Price", f"${m['Current']:.2f}")
        st.metric("1Y Target", f"${m['Target']:.2f}", delta=f"{((m['Target']/m['Current'])-1)*100:.1f}% Upside")
        st.metric("Market Cap", f"${m['MarketCap']/1e12:.2f}T")
        st.metric("Ann. Volatility (σ)", f"{m['Volatility']:.1%}")

# --- TAB 2: Comparison ---
with tab2:
    st.subheader("Relative Performance (Normalized to 100)")
    norm_df = (all_prices_5y.tail(252) / all_prices_5y.tail(252).iloc[0]) * 100
    st.plotly_chart(px.line(norm_df, template="plotly_dark"), use_container_width=True)
    st.dataframe(pd.DataFrame([{ "Ticker": t, "Rating": all_meta[t]['Rating'], "1Y Return": f"{all_meta[t]['AnnReturn']:.1%}", "P/E": all_meta[t]["PE"], "Beta": round(all_meta[t]['Beta'], 2) } for t in tickers]), use_container_width=True)

# --- TAB 3: Optimizer ---
with tab3:
    if not OPTIMIZER_AVAILABLE:
        st.error("Optimization Library Missing.")
    else:
        st.subheader("Asset Allocation Model")
        c_in, c_out = st.columns([1, 2])
        with c_in:
            mu = expected_returns.mean_historical_return(all_prices_5y)
            S = risk_models.sample_cov(all_prices_5y)
            target_p = st.slider("Target Annual Return %", 5, 100, 25)
        with c_out:
            try:
                ef = EfficientFrontier(mu, S)
                weights = ef.efficient_return(target_return=target_p/100)
                cleaned = ef.clean_weights()
                ret, vol, sha = ef.portfolio_performance()
                col1, col2, col3 = st.columns(3)
                col1.metric("Exp. Return", f"{ret:.1%}"); col2.metric("Volatility", f"{vol:.1%}"); col3.metric("Sharpe", f"{sha:.2f}")
                st.table(pd.DataFrame([{"Ticker": t, "Weight": f"{w*100:.1f}%"} for t, w in cleaned.items() if w > 0]))
            except: st.warning("Target return infeasible.")
