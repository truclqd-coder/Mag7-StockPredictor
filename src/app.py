import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pypfopt import EfficientFrontier, risk_models, expected_returns

# --- Page Setup ---
st.set_page_config(page_title="Mag7 Quant Terminal", layout="wide")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.6rem; color: #ff4b4b; }
    .stRadio > div { flex-direction: row; justify-content: flex-start; gap: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Mag7 Research Terminal & Strategic Optimizer")

# --- Global Sidebar ---
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
selected_ticker = st.sidebar.selectbox("Primary Focus Ticker", tickers)

# --- HARDENED EARNINGS ENGINE ---
@st.cache_data(ttl=3600)
def fetch_terminal_data(ticker_list, focus_ticker):
    all_data = yf.download(ticker_list, period="5y", multi_level_index=False)['Close']
    focus_obj = yf.Ticker(focus_ticker)
    info = focus_obj.info
    
    next_e = "N/A"
    try:
        raw_ts = info.get('earningsTimestamp') or info.get('nextEarningsDate')
        if raw_ts:
            next_e = pd.to_datetime(raw_ts, unit='s').strftime('%Y-%m-%d')
        if next_e == "N/A":
            e_df = focus_obj.get_earnings_dates(limit=1)
            if e_df is not None and not e_df.empty:
                next_e = e_df.index[0].strftime('%Y-%m-%d')
    except:
        next_e = "Check IR Calendar"
        
    return all_data, info, next_e

all_close, main_info, e_date = fetch_terminal_data(tickers, selected_ticker)
tf_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "5Y": 1825}

# --- App Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Research", "🔄 Performance", "⚖️ Optimizer"])

# --- TAB 1: Equity Research ---
with tab1:
    col_l, col_r = st.columns([2, 1])
    with col_l:
        choice1 = st.radio("Chart Window:", ["1M", "3M", "6M", "1Y", "5Y"], index=1, horizontal=True, key="r_chart")
        plot_df = all_close[selected_ticker].tail(tf_map[choice1])
        fig_price = px.line(plot_df, template="plotly_dark", title=f"{selected_ticker} ({choice1})")
        fig_price.update_layout(xaxis_title=None, yaxis_title="Price ($)")
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col_r:
        st.subheader("Key Statistics")
        st.error(f"⚠️ Next Earnings: **{e_date}**")
        st.divider()
        st.metric("Market Cap", f"${main_info.get('marketCap', 0):,.0f}")
        st.metric("P/E Ratio", main_info.get('trailingPE', 'N/A'))
        st.metric("Beta (5Y)", main_info.get('beta', 'N/A'))
        st.metric("1Y Target", f"${main_info.get('targetMeanPrice', 'N/A')}")

# --- TAB 2: Performance Comparison ---
with tab2:
    choice2 = st.radio("Compare Window:", ["1M", "3M", "6M", "1Y", "5Y"], index=1, horizontal=True, key="p_chart")
    comp_df = all_close.tail(tf_map[choice2])
    norm_df = (comp_df / comp_df.iloc[0]) * 100
    
    fig_comp = go.Figure()
    for t in tickers:
        fig_comp.add_trace(go.Scatter(
            x=norm_df.index, y=norm_df[t], name=t,
            line=dict(width=4 if t == selected_ticker else 1, color='white' if t == selected_ticker else None),
            opacity=1 if t == selected_ticker else 0.5
        ))
    fig_comp.update_layout(template="plotly_dark", title=f"Relative Growth ({choice2})", yaxis_title="Rebased Price")
    st.plotly_chart(fig_comp, use_container_width=True)

# --- TAB 3: Strategic Optimizer ---
with tab3:
    st.subheader("Custom Mean-Variance Strategy")
    st.markdown("Enter your forecasts to calculate the **Minimum Volatility** weights.")
    
    # Risk Engine (Covariance Matrix)
    S = risk_models.sample_cov(all_close)
    mu_hist = expected_returns.mean_historical_return(all_close)

    col_opt1, col_opt2 = st.columns([1, 2])
    
    with col_opt1:
        st.write("### 1. Forecast Stock Returns")
        views = {}
        for t in tickers:
            views[t] = st.number_input(f"{t} Expected Return %", value=float(mu_hist[t]*100), step=1.0)
        
        st.divider()
        st.write("### 2. Set Portfolio Target")
        target_ret = st.slider("Target Portfolio Return (%)", 0, 100, 20)
        use_target = st.checkbox("Optimize for Target Return", value=False)

    with col_opt2:
        try:
            custom_mu = pd.Series({t: v/100 for t, v in views.items()})
            ef = EfficientFrontier(custom_mu, S)
            ef.add_constraint(lambda w: w <= 0.40) # Max 40% cap
            
            if use_target:
                weights = ef.efficient_return(target_return=target_ret/100)
                label = f"Min. Volatility for {target_ret}% Return"
            else:
                weights = ef.max_sharpe()
                label = "Max Sharpe Ratio (Best Risk/Reward)"

            cleaned = ef.clean_weights()
            ret, vol, sha = ef.portfolio_performance()
            
            st.write(f"### Results: {label}")
            res_c1, res_c2 = st.columns(2)
            res_c1.metric("Expected Return", f"{ret:.1%}")
            res_c1.metric("Portfolio Volatility (Risk)", f"{vol:.1%}")
            res_c2.metric("Sharpe Ratio", f"{sha:.2f}")
            
            st.write("#### Optimized Allocation")
            st.table(pd.Series({k: v for k, v in cleaned.items() if v > 0}, name="Weight").apply(lambda x: f"{x:.1%}"))
            
            fig_pie = px.pie(names=list(cleaned.keys()), values=list(cleaned.values()), hole=0.4, 
                             template="plotly_dark", title="Portfolio Mix")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        except Exception as e:
            st.error("Scenario impossible. Try lowering your Target Return or increasing stock expectations.")
