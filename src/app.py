import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

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
    div[data-testid="stMetric"] { background-color: #1c2128; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    .earnings-banner { background: linear-gradient(180deg, rgba(88, 166, 255, 0.15) 0%, rgba(13, 17, 23, 0) 100%); border: 2px solid #58a6ff; border-radius: 12px; padding: 25px; text-align: center; margin-bottom: 20px; }
    .rating-card { padding: 20px; border-radius: 12px; text-align: center; border: 2px solid; background: rgba(28, 33, 40, 0.5); margin-bottom: 15px; }
    .hero-return-card { background: linear-gradient(135deg, #1c2128 0%, #0d1117 100%); border: 2px solid #58a6ff; border-radius: 15px; padding: 30px; text-align: center; margin-bottom: 25px; }
    </style>
    """, unsafe_allow_html=True)

# --- AI Prediction Engine ---
def predict_7day_trend(df):
    """Simple Linear Regression to project the next 7 days."""
    df = df.reset_index()
    df['Days'] = np.arange(len(df))
    X = df[['Days']].values
    y = df['Close'].values
    
    model = LinearRegression().fit(X, y)
    
    # Project 7 days forward
    future_days = np.array([len(df) + i for i in range(1, 8)]).reshape(-1, 1)
    predictions = model.predict(future_days)
    
    last_date = df['Date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
    return pd.DataFrame({'Date': future_dates, 'Predicted': predictions})

# --- Data Engine ---
@st.cache_data(ttl=300)
def fetch_multi_performance(ticker_list, horizon):
    period_map = {"1D":"1d", "5D":"5d", "1M":"1mo", "3M":"3mo", "6M":"6mo", "YTD":"ytd", "1Y":"1y", "5Y":"5y"}
    p = period_map.get(horizon, "1y")
    return yf.download(ticker_list, period=p, multi_level_index=False)['Close']

@st.cache_data(ttl=3600)
def fetch_global_meta(ticker_list):
    prices = yf.download(ticker_list, period="5y", multi_level_index=False)['Close']
    vol_series = prices.pct_change().std() * np.sqrt(252)
    def get_ret(days):
        t_date = prices.index[-1] - timedelta(days=days)
        idx = prices.index.get_indexer([t_date], method='backfill')[0]
        return ((prices.iloc[-1] / prices.iloc[idx]) - 1)
    
    y_idx = prices.index.get_indexer([datetime(datetime.now().year, 1, 1)], method='backfill')[0]
    
    meta_store = {}
    for t in ticker_list:
        obj = yf.Ticker(t)
        info = obj.info
        meta_store[t] = {
            "Current": info.get('currentPrice', 0),
            "PrevClose": info.get('previousClose', 0),
            "Target": info.get('targetMeanPrice', 0),
            "Beta": info.get('beta', 1.0),
            "Volatility": vol_series[t],
            "PE": info.get('trailingPE', 0),
            "EPS": info.get('trailingEps', 0),
            "NextEarnings": pd.to_datetime(info.get('nextEarningsDate'), unit='s').strftime('%Y-%m-%d') if info.get('nextEarningsDate') else "N/A",
            "Rating": info.get('recommendationKey', 'N/A').replace('_', ' ').upper(),
            "Analysts": info.get('numberOfAnalystOpinions', 'N/A'),
            "DivYield": info.get('dividendYield', 0),
            "Return1Y": get_ret(365)[t],
            "ReturnYTD": ((prices.iloc[-1][t] / prices.iloc[y_idx][t]) - 1)
        }
    return prices, meta_store

# --- Initialization ---
mag7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
others = ["PYPL", "CX", "GNW"]
all_tickers = list(set(mag7 + others))
selected_ticker = st.sidebar.selectbox("Active Security", mag7)

all_prices_5y, all_meta = fetch_global_meta(all_tickers)
m = all_meta[selected_ticker]

st.markdown('<div class="terminal-title">MAG7 QUANTITATIVE TERMINAL v1.0</div>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["📈 PERFORMANCE", "📊 COMPARISON", "⚖️ OPTIMIZER", "📁 PORTFOLIO"])

# --- TAB 1: ADDED TREND PREDICTION ---
with tab1:
    col_l, col_r = st.columns([2.5, 1])
    with col_l:
        h1 = st.radio("HORIZON", ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y"], index=5, horizontal=True)
        t1_data = fetch_multi_performance([selected_ticker], h1)
        
        # Base Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t1_data.index, y=t1_data.values, name="Historical", line=dict(color='#58a6ff')))
        
        # Add Prediction Layer (using 30D lookback for model)
        if h1 in ["1M", "3M", "6M", "YTD", "1Y", "5Y"]:
            lookback_data = fetch_multi_performance([selected_ticker], "1M")
            pred_df = predict_7day_trend(lookback_data)
            fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted'], name="AI 7D Trend", line=dict(color='#ff7b72', dash='dot')))
            st.caption("🔴 Dashed line indicates a 7-day linear trend projection based on recent volatility.")
            
        fig.update_layout(template="plotly_dark", yaxis_title="USD", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Prev Close", f"${m['PrevClose']:.2f}"); v2.metric("EPS", f"${m['EPS']:.2f}"); v3.metric("P/E Ratio", f"{m['PE']:.2f}"); v4.metric("Div Yield", f"{m['DivYield']:.2%}")
    with col_r:
        r_color = "#3fb950" if "BUY" in m['Rating'] else "#f85149"
        st.markdown(f'<div class="rating-card" style="border-color:{r_color}; color:{r_color};"><div style="font-size:26px; font-weight:900;">{m["Rating"]}</div><div style="color:#8b949e; font-size:13px;">Consensus of {m["Analysts"]} Analysts</div></div>', unsafe_allow_html=True)
        st.metric("Current Price", f"${m['Current']:.2f}")
        st.metric("Beta (β)", f"{m['Beta']:.2f}", help="Sensitivity to market moves."); st.metric("Ann. Volatility (σ)", f"{m['Volatility']:.1%}", help="Annualized risk.")

# --- TAB 2: COMPARISON ---
with tab2:
    h2 = st.radio("COMPARISON HORIZON", ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y"], index=5, horizontal=True, key="t2_h")
    comp_data = fetch_multi_performance(mag7, h2)
    st.plotly_chart(px.line((comp_data / comp_data.iloc[0]) * 100, template="plotly_dark", title=f"Relative Growth Index ({h2})"), use_container_width=True)
    def color_ret(val): return f'color: {"#f85149" if val < 0 else "#3fb950"}; font-weight: bold'
    summary_df = pd.DataFrame([{"Ticker": t, "1Y Return (%)": all_meta[t]['Return1Y'] * 100, "Rating": all_meta[t]['Rating']} for t in mag7])
    st.dataframe(summary_df.style.format({"1Y Return (%)": "{:.1f}%"}).applymap(color_ret, subset=['1Y Return (%)']), use_container_width=True)

# --- TAB 3: OPTIMIZER ---
with tab3:
    if OPTIMIZER_AVAILABLE:
        st.subheader("Modern Portfolio Strategy")
        o_in, o_out = st.columns([1, 2.2])
        with o_in:
            target_p = st.slider("Target Portfolio Return %", 5, 100, 25, help="Glossary: Solve for specific return.")
            mode = st.toggle("Maximize Sharpe Ratio", help="Glossary: Best risk-adjusted return.")
        with o_out:
            st.markdown(f'<div class="hero-return-card"><small>CALCULATED PORTFOLIO RETURN</small><div style="color:#58a6ff; font-size:48px; font-weight:900;">{target_p}%</div></div>', unsafe_allow_html=True)

# --- TAB 4: PORTFOLIO ---
with tab4:
    st.subheader("📁 Live Asset Portfolio")
    my_holdings = [{"Ticker": "MSFT", "Shares": 34, "BuyPrice": 411.64}, {"Ticker": "GOOGL", "Shares": 16.69014, "BuyPrice": 168.86}, {"Ticker": "AMZN", "Shares": 18, "BuyPrice": 206.36}, {"Ticker": "NVDA", "Shares": 61, "BuyPrice": 128.61}, {"Ticker": "PYPL", "Shares": 25, "BuyPrice": 71.67}, {"Ticker": "CX", "Shares": 5, "BuyPrice": 6.53}, {"Ticker": "GNW", "Shares": 11, "BuyPrice": 5.81}]
    cash_balance = 13.67 
    equity_value = sum([all_meta[a['Ticker']]['Current'] * a['Shares'] for a in my_holdings])
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Equity", f"${equity_value:
