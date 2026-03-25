import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns

# --- Page Setup & Custom CSS ---
st.set_page_config(page_title="Mag7 Quant Terminal", layout="wide")

# Custom CSS to highlight the Volatility Alert
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #ff4b4b; }
    .main { background-color: #0e1117; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Mag7 Research Terminal & Risk Optimizer")

# --- Global Sidebar Settings ---
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
selected_ticker = st.sidebar.selectbox("Primary Focus Ticker", tickers)

timeframe_map = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365, "Max (5Y)": 1825}
time_label = st.sidebar.selectbox("Analysis Window", list(timeframe_map.keys()), index=1)
lookback = timeframe_map[time_label]

# --- Enhanced Data Engine ---
@st.cache_data(ttl=3600)
def fetch_terminal_data(ticker_list, focus_ticker):
    # Prices for all (used in Tabs 2, 3, and 4)
    all_data = yf.download(ticker_list, period="5y", multi_level_index=False)['Close']
    
    # Fundamentals for focus stock (Tab 1)
    focus_obj = yf.Ticker(focus_ticker)
    info = focus_obj.info
    
    # Robust Earnings Retrieval
    try:
        # get_earnings_dates provides the most accurate forward-looking schedule
        cal = focus_obj.get_earnings_dates(limit=1)
        next_e = cal.index[0].strftime('%Y-%m-%d') if not cal.empty else "Check IR"
    except:
        next_e = "N/A"
        
    return all_data, info, next_e

with st.spinner('Syncing with market servers...'):
    all_close, main_info, e_date = fetch_terminal_data(tickers, selected_ticker)

# --- App Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Research", "🔄 Performance", "🔮 Optimizer", "🔥 Risk Heatmap"])

# --- TAB 1: Equity Research ---
with tab1:
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.subheader(f"{selected_ticker} Price Action")
        fig_price = px.line(all_close[selected_ticker].tail(lookback), template="plotly_dark")
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col_r:
        st.subheader("Key Statistics")
        st.metric(label="⚠️ Next Earnings (Volatility Alert)", value=e_date, delta="High Impact Event")
        st.divider()
        stats = {
            "Market Cap": f"${main_info.get('marketCap', 0):,.0f}",
            "P/E Ratio": main_info.get('trailingPE', 'N/A'),
            "Beta (5Y)": main_info.get('beta', 'N/A'),
            "1Y Target": f"${main_info.get('targetMeanPrice', 'N/A')}"
        }
        st.table(pd.Series(stats, name="Value"))

# --- TAB 2: Normalized Comparison ---
with tab2:
    st.subheader("Relative Growth (Base 100)")
    norm_df = (all_close.tail(lookback) / all_close.tail(lookback).iloc[0]) * 100
    fig_comp = go.Figure()
    for t in tickers:
        is_f = (t == selected_ticker)
        fig_comp.add_trace(go.Scatter(x=norm_df.index, y=norm_df[t], name=t,
                                     line=dict(width=4 if is_f else 1, color='white' if is_f else None),
                                     opacity=1 if is_f else 0.5))
    fig_comp.update_layout(template="plotly_dark", yaxis_title="Growth of $100")
    st.plotly_chart(fig_comp, use_container_width=True)

# --- TAB 3: Strategic Optimizer ---
with tab3:
    st.subheader("Mean-Variance Scenario Simulator")
    st.markdown("Adjust returns to recalculate the **Max Sharpe Ratio** portfolio.")
    
    # Math Setup
    mu_hist = expected_returns.mean_historical_return(all_close)
    S = risk_models.sample_cov(all_close)
    
    # Input Grid
    in_cols = st.columns(len(tickers))
    views = {t: in_cols[i].number_input(f"{t} %", value=float(mu_hist[t]*100), step=1.0) for i, t in enumerate(tickers)}
    
    try:
        custom_mu = pd.Series({t: v/100 for t, v in views.items()})
        ef = EfficientFrontier(custom_mu, S)
        weights = ef.max_sharpe()
        cleaned = ef.clean_weights()
        ret, vol, sha = ef.portfolio_performance()
        
        res_l, res_r = st.columns([1, 2])
        with res_l:
            st.write("### Optimal Weights")
            st.table(pd.Series({k: v for k, v in cleaned.items() if v > 0}, name="Weight").apply(lambda x: f"{x:.1%}"))
            st.metric("Expected Return", f"{ret:.1%}")
            st.metric("Sharpe Ratio", f"{sha:.2f}")
        with res_r:
            fig_p = px.pie(names=list(cleaned.keys()), values=list(cleaned.values()), hole=0.4, 
                           title="Recommended Mix", color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_p, use_container_width=True)
    except:
        st.error("Invalid return scenario.")

# --- TAB 4: Volatility Heatmap ---
with tab4:
    st.subheader("Systemic Risk Heatmap")
    st.markdown("Correlation of daily returns. Values near **1.0** indicate high similarity.")
    
    returns = all_close.tail(lookback).pct_change().dropna()
    corr = returns.corr()
    
    fig_h, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, cmap='RdYlGn', ax=ax, center=0.5)
    fig_h.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    plt.xticks(color='white')
    plt.yticks(color='white')
    st.pyplot(fig_h)
