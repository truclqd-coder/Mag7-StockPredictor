import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

# --- Page Configuration ---
st.set_page_config(page_title="Mag7 Research & Optimizer", layout="wide")
st.title("🚀 Magnificent 7 Research Terminal & Strategic Optimizer")

# --- Sidebar Configuration ---
st.sidebar.header("Global Settings")
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
selected_ticker = st.sidebar.selectbox("Primary Focus Ticker", tickers)

timeframe_map = {
    "1 Month": 30, "3 Months": 90, "6 Months": 180, 
    "1 Year": 365, "Max (5Y)": 1825
}
time_label = st.sidebar.selectbox("Analysis Window", list(timeframe_map.keys()), index=1)
lookback = timeframe_map[time_label]

# --- Global Data Engine (Cached) ---
@st.cache_data(ttl=3600)
def fetch_comprehensive_data(ticker_list, focus_ticker):
    # Fetch all tickers for comparison/optimization
    df = yf.download(ticker_list, period="5y", multi_level_index=False)
    close_data = df['Close']
    
    # Fetch specific info and calendar for the focus ticker
    stock_obj = yf.Ticker(focus_ticker)
    info = stock_obj.info
    calendar = stock_obj.calendar
    return close_data, info, calendar

with st.spinner('Syncing with global market servers...'):
    all_close, main_info, main_cal = fetch_comprehensive_data(tickers, selected_ticker)

# --- App Layout with Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Equity Research", "🔄 Performance Comparison", "🔮 Strategic Optimizer"])

# --- TAB 1: Equity Research ---
with tab1:
    col_l, col_r = st.columns([2, 1])
    
    with col_l:
        st.subheader(f"{selected_ticker} Price Action")
        chart_df = all_close[selected_ticker].tail(lookback)
        fig = px.line(chart_df, template="plotly_dark", color_discrete_sequence=['#00d4ff'])
        fig.update_layout(xaxis_title=None, yaxis_title="Price ($)", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col_r:
        st.subheader("Key Statistics")
        # Earnings Logic
        e_date = "N/A"
        if main_cal is not None:
            e_date = main_cal.get('Earnings Date', ['N/A'])[0] if isinstance(main_cal, dict) else main_cal.iloc[0,0]

        stats = {
            "Market Cap": f"${main_info.get('marketCap', 0):,.0f}",
            "Beta (5Y)": main_info.get('beta', 'N/A'),
            "PE Ratio": main_info.get('trailingPE', 'N/A'),
            "Next Earnings": e_date,
            "1Y Target": f"${main_info.get('targetMeanPrice', 'N/A')}",
            "Div Yield": f"{main_info.get('dividendYield', 0)*100:.2f}%" if main_info.get('dividendYield') else "0.00%"
        }
        st.table(pd.Series(stats, name="Value"))

# --- TAB 2: Performance Comparison ---
with tab2:
    st.subheader(f"Normalized Growth: Mag7 vs {selected_ticker}")
    st.markdown("Rebased to **100** at start of period to show relative % returns.")
    
    comp_df = all_close.tail(lookback).copy()
    norm_df = (comp_df / comp_df.iloc[0]) * 100
    
    fig_comp = go.Figure()
    for t in tickers:
        is_focus = (t == selected_ticker)
        fig_comp.add_trace(go.Scatter(
            x=norm_df.index, y=norm_df[t], name=t,
            line=dict(width=4 if is_focus else 1, color='white' if is_focus else None),
            opacity=1 if is_focus else 0.5
        ))
    fig_comp.update_layout(template="plotly_dark", yaxis_title="Growth (Base 100)")
    st.plotly_chart(fig_comp, use_container_width=True)

# --- TAB 3: Strategic Optimizer (Scenario Simulator) ---
with tab3:
    st.subheader("⚖️ Mean-Variance Scenario Simulator")
    st.markdown("Adjust the **Expected Annual Returns (%)** below. The model will recommend optimal weights to maximize the Sharpe Ratio.")
    
    # 1. Inputs for Expected Returns
    hist_mu = expected_returns.mean_historical_return(all_close)
    S = risk_models.sample_cov(all_close)
    
    st.write("### 1. Define Your Forecasts")
    input_cols = st.columns(len(tickers))
    user_views = {}
    for i, t in enumerate(tickers):
        # Default to historical mean
        user_views[t] = input_cols[i].number_input(f"{t} %", value=float(hist_mu[t]*100), step=1.0)

    # 2. Optimization Engine
    custom_mu = pd.Series({t: v/100 for t, v in user_views.items()})
    
    try:
        ef = EfficientFrontier(custom_mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        ret, vol, sharpe = ef.portfolio_performance()

        # 3. Display Outputs
        res_l, res_r = st.columns([1, 2])
        with res_l:
            st.write("### Recommended Weights")
            # Only show assets with an allocation > 0
            active_w = {k: v for k, v in cleaned_weights.items() if v > 0}
            st.table(pd.Series(active_w, name="Allocation").apply(lambda x: f"{x:.1%}"))
            
            st.metric("Simulated Portfolio Return", f"{ret:.1%}")
            st.metric("Portfolio Volatility (Risk)", f"{vol:.1%}")
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        with res_r:
            fig_pie = px.pie(
                names=list(cleaned_weights.keys()), 
                values=list(cleaned_weights.values()),
                hole=0.4, title="Optimized Asset Mix",
                color_discrete_sequence=px.colors.qualitative.T10
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    except Exception as e:
        st.error("Scenario is mathematically inconsistent. Please ensure your return expectations are realistic.")

    st.divider()
    st.caption("⚠️ **Disclaimer:** This tool uses Mean-Variance Optimization. Asset weights are highly sensitive to input return estimates (the 'Instability Problem'). This is for educational portfolio analysis and not financial advice.")
