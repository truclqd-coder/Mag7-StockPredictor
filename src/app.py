import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

# --- Page Setup ---
st.set_page_config(page_title="Mag7 Research & Optimizer", layout="wide")
st.title("🚀 Magnificent 7 Research Terminal & Optimizer")

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
def fetch_all_data(ticker_list):
    # Fetch all tickers at once to save API calls
    df = yf.download(ticker_list, period="5y", multi_level_index=False)
    # yfinance returns a MultiIndex when downloading multiple tickers. 
    # We extract Close prices for comparison and optimization.
    close_data = df['Close']
    
    # Also fetch specific fundamental info for the selected ticker
    info = yf.Ticker(selected_ticker).info
    return close_data, info

with st.spinner('Syncing with market servers...'):
    all_close, main_info = fetch_all_data(tickers)

# --- App Layout with Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Equity Research", "🔄 Relative Performance", "⚖️ Portfolio Optimizer"])

# --- TAB 1: Equity Research (Single Ticker) ---
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
        stats = {
            "Market Cap": f"${main_info.get('marketCap', 0):,.0f}",
            "Beta (5Y)": main_info.get('beta', 'N/A'),
            "PE Ratio": main_info.get('trailingPE', 'N/A'),
            "EPS (TTM)": main_info.get('trailingEps', 'N/A'),
            "1Y Target": f"${main_info.get('targetMeanPrice', 'N/A')}",
            "Div Yield": f"{main_info.get('dividendYield', 0)*100:.2f}%" if main_info.get('dividendYield') else "N/A"
        }
        st.table(pd.Series(stats, name="Value"))

# --- TAB 2: Performance Comparison ---
with tab2:
    st.subheader(f"Normalized Growth: Mag7 vs {selected_ticker}")
    # Normalize data to Base 100
    comp_df = all_close.tail(lookback).copy()
    norm_df = (comp_df / comp_df.iloc[0]) * 100
    
    fig_comp = go.Figure()
    for t in tickers:
        is_main = (t == selected_ticker)
        fig_comp.add_trace(go.Scatter(
            x=norm_df.index, y=norm_df[t], name=t,
            line=dict(width=4 if is_main else 1, color=None if not is_main else 'white'),
            opacity=1 if is_main else 0.5
        ))
    fig_comp.update_layout(template="plotly_dark", yaxis_title="Growth (Base 100)")
    st.plotly_chart(fig_comp, use_container_width=True)

# --- TAB 3: Portfolio Optimizer (Mean-Variance) ---
with tab3:
    st.subheader("Mean-Variance Optimization")
    st.caption("Calculating the Max Sharpe Ratio based on the last 5 years of covariance.")
    
    # 1. Math Models
    mu = expected_returns.mean_historical_return(all_close)
    S = risk_models.sample_cov(all_close)
    
    # 2. Optimization
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False)
    
    res_l, res_r = st.columns(2)
    with res_l:
        st.write("### Optimal Weights")
        w_df = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Allocation'])
        st.dataframe(w_df.style.format("{:.2%}") )
        
        st.metric("Expected Annual Return", f"{perf[0]:.1%}")
        st.metric("Annual Volatility", f"{perf[1]:.1%}")
        st.metric("Sharpe Ratio", f"{perf[2]:.2f}")
        
    with res_r:
        fig_pie = px.pie(
            names=list(cleaned_weights.keys()), 
            values=list(cleaned_weights.values()),
            hole=0.5, title="Recommended Asset Mix",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_pie, use_container_width=True)
