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
    
    .earnings-banner {
        background: linear-gradient(180deg, rgba(88, 166, 255, 0.15) 0%, rgba(13, 17, 23, 0) 100%);
        border: 2px solid #58a6ff; border-radius: 12px; padding: 25px; text-align: center; margin-bottom: 20px;
    }
    .earnings-date { color: #58a6ff; font-size: 32px; font-weight: 800; font-family: 'Courier New', monospace; }

    .rating-card {
        padding: 20px; border-radius: 12px; text-align: center; border: 2px solid; margin-bottom: 15px;
        background: rgba(28, 33, 40, 0.5);
    }
    .rating-text { font-size: 26px; font-weight: 900; letter-spacing: 1px; margin-bottom: 2px; }
    .analyst-subtext { color: #8b949e; font-size: 13px; font-weight: 500; }

    .target-card {
        background: #1c2128; border: 1px solid #30363d; border-radius: 12px; padding: 18px;
        text-align: center; margin-bottom: 15px;
    }

    .hero-return-card {
        background: linear-gradient(135deg, #1c2128 0%, #0d1117 100%);
        border: 2px solid #58a6ff; border-radius: 15px; padding: 30px;
        text-align: center; margin-bottom: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ML Engine: 7-Day Trend Prediction ---
def get_ml_prediction(ticker):
    # Fetch 30 days of data to train the model
    df = yf.download(ticker, period="1mo", multi_level_index=False)['Close']
    df = df.reset_index()
    df['DayNum'] = np.arange(len(df))
    
    # Train Linear Regression Model
    X = df[['DayNum']].values
    y = df['Close'].values
    model = LinearRegression().fit(X, y)
    
    # Predict next 7 days
    future_days = np.array([len(df) + i for i in range(1, 8)]).reshape(-1, 1)
    preds = model.predict(future_days)
    
    future_dates = [df['Date'].max() + timedelta(days=i) for i in range(1, 8)]
    return pd.DataFrame({'Date': future_dates, 'Predicted': preds})

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

# --- TAB 1: PERFORMANCE + NEW ML TREND ---
with tab1:
    col_l, col_r = st.columns([2.5, 1])
    with col_l:
        h1 = st.radio("HORIZON", ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y"], index=5, horizontal=True)
        t1_data = fetch_multi_performance([selected_ticker], h1)
        
        # Plotly Graph with Prediction Layer
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t1_data.index, y=t1_data.values, name="Historical", line=dict(color='#58a6ff', width=2)))
        
        # Add ML Prediction Line (Visible on 1M+ views)
        if h1 not in ["1D", "5D"]:
            pred_df = get_ml_prediction(selected_ticker)
            fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted'], name="ML 7D Trend", line=dict(color='#ff7b72', dash='dash')))
            st.caption("🤖 Red dashed line shows the 7-day trend predicted by a Linear Regression model.")

        fig.update_layout(template="plotly_dark", yaxis_title="USD", margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Prev Close", f"${m['PrevClose']:.2f}"); v2.metric("EPS", f"${m['EPS']:.2f}"); v3.metric("P/E Ratio", f"{m['PE']:.2f}"); v4.metric("Div Yield", f"{m['DivYield']:.2%}")
    with col_r:
        st.markdown(f'<div class="earnings-banner"><small>NEXT EARNINGS</small><div class="earnings-date">{m["NextEarnings"]}</div></div>', unsafe_allow_html=True)
        r_color = "#3fb950" if "BUY" in m['Rating'] else "#f85149"
        st.markdown(f'<div class="rating-card" style="border-color:{r_color}; color:{r_color};"><div class="rating-text">{m["Rating"]}</div><div class="analyst-subtext">Consensus of {m["Analysts"]} Analysts</div></div>', unsafe_allow_html=True)
        st.metric("Current Price", f"${m['Current']:.2f}")
        st.metric("Beta (β)", f"{m['Beta']:.2f}", help="Sensitivity to market moves."); st.metric("Ann. Volatility (σ)", f"{m['Volatility']:.1%}", help="Annualized risk.")

# --- TAB 2: COMPARISON (RESTORED) ---
with tab2:
    h2 = st.radio("COMPARISON HORIZON", ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y"], index=5, horizontal=True, key="t2_h")
    comp_data = fetch_multi_performance(mag7, h2)
    st.plotly_chart(px.line((comp_data / comp_data.iloc[0]) * 100, template="plotly_dark", title=f"Relative Growth Index ({h2})"), use_container_width=True)
    def color_ret(val): return f'color: {"#f85149" if val < 0 else "#3fb950"}; font-weight: bold'
    summary_df = pd.DataFrame([{"Ticker": t, "1Y Return (%)": all_meta[t]['Return1Y'] * 100, "Rating": all_meta[t]['Rating']} for t in mag7])
    st.dataframe(summary_df.style.format({"1Y Return (%)": "{:.1f}%"}).applymap(color_ret, subset=['1Y Return (%)']), use_container_width=True)

# --- TAB 3: OPTIMIZER + GLOSSARY (RESTORED) ---
with tab3:
    if OPTIMIZER_AVAILABLE:
        st.subheader("Modern Portfolio Strategy (MPT)")
        o_in, o_out = st.columns([1, 2.2])
        with o_in:
            mu_hist = expected_returns.mean_historical_return(all_prices_5y[mag7])
            user_views = {t: st.number_input(f"{t} Exp. Return %", value=float(mu_hist[t]*100), help="Adjust expected return.") for t in mag7}
            target_p = st.slider("Target Portfolio Return %", 5, 100, 25, help="Solve for specific return goal.")
            mode = st.toggle("Maximize Sharpe Ratio", help="Most risk-efficient portfolio.")
        with o_out:
            try:
                mu, S = pd.Series({t: v/100 for t, v in user_views.items()}), risk_models.sample_cov(all_prices_5y[mag7])
                ef = EfficientFrontier(mu, S)
                weights = ef.max_sharpe() if mode else ef.efficient_return(target_return=target_p/100)
                ret, vol, sha = ef.portfolio_performance()
                st.markdown(f'<div class="hero-return-card"><small>CALCULATED PORTFOLIO RETURN</small><div style="color:#58a6ff; font-size:48px; font-weight:900;">{ret:.1%}</div></div>', unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                m1.metric("Risk (Volatility)", f"{vol:.1%}", help="Annualized portfolio deviation."); m2.metric("Sharpe Ratio", f"{sha:.2f}", help="Excess return per unit of risk.")
                st.plotly_chart(px.pie(names=list(ef.clean_weights().keys()), values=list(ef.clean_weights().values()), hole=0.5, template="plotly_dark"), use_container_width=True)
            except: st.warning("Target return is mathematically infeasible.")

# --- TAB 4: PORTFOLIO (SYNCED TO YOUR IMAGE + $13.67 CASH) ---
with tab4:
    st.subheader("📁 Live Asset Portfolio")
    my_holdings = [
        {"Ticker": "MSFT", "Shares": 34, "BuyPrice": 411.64},
        {"Ticker": "GOOGL", "Shares": 16.69014, "BuyPrice": 168.86},
        {"Ticker": "AMZN", "Shares": 18, "BuyPrice": 206.36},
        {"Ticker": "NVDA", "Shares": 61, "BuyPrice": 128.61},
        {"Ticker": "PYPL", "Shares": 25, "BuyPrice": 71.67},
        {"Ticker": "CX", "Shares": 5, "BuyPrice": 6.53},
        {"Ticker": "GNW", "Shares": 11, "BuyPrice": 5.81}
    ]
    cash_balance = 13.67 
    equity_value = sum([all_meta[a['Ticker']]['Current'] * a['Shares'] for a in my_holdings])
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Equity", f"${equity_value:,.2f}"); m2.metric("Cash Balance", f"${cash_balance:,.2f}"); m3.metric("Portfolio Value", f"${(equity_value + cash_balance):,.2f}")
    p_df = pd.DataFrame([{"Ticker": a['Ticker'], "Shares": round(a['Shares'], 2), "Open P&L (%)": ((all_meta[a['Ticker']]['Current']/a['BuyPrice'])-1)*100} for a in my_holdings])
    st.dataframe(p_df.style.format({"Open P&L (%)": "{:.1f}%"}).applymap(color_ret, subset=['Open P&L (%)']), use_container_width=True)
