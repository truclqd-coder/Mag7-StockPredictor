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

# --- TRIPLE-LAYER EARNINGS ENGINE ---
@st.cache_data(ttl=3600)
def fetch_terminal_data(ticker_list, focus_ticker):
    all_data = yf.download(ticker_list, period="5y", multi_level_index=False)['Close']
    focus_obj = yf.Ticker(focus_ticker)
    info = focus_obj.info
    
    next_e = "N/A"
    try:
        # Layer 1: Earnings Dates Method
        earning_df = focus_obj.get_earnings_dates(limit=1)
        if earning_df is not None and not earning_df.empty:
            next_e = earning_df.index[0].strftime('%Y-%m-%d')
        
        # Layer 2: Calendar Fallback
        if next_e == "N/A":
            cal = focus_obj.calendar
            if isinstance(cal, pd.DataFrame) and not cal.empty:
                next_e = cal.iloc[0, 0].strftime('%Y-%m-%d')
                
        # Layer 3: Info Metadata Fallback
        if next_e == "N/A":
            raw_e = info.get('nextEarningsDate')
            if raw_e:
                next_e = pd.to_datetime(raw_e, unit='s').strftime('%Y-%m-%d')
    except:
        next_e = "Check IR Calendar"
        
    return all_data, info, next_e

all_close, main_info, e_date = fetch_terminal_data(tickers, selected_ticker)
tf_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "5Y": 1825}

# --- App Tabs (Heatmap Removed) ---
tab1, tab2, tab3 = st.tabs(["📊 Research", "🔄 Performance", "⚖️ Optimizer"])

# --- TAB 1: Equity Research ---
with tab1:
    col_l, col_r = st.columns([2, 1])
    with col_l:
        # Timeframe Selector directly above the chart
        tf_choice1 = st.radio("Select Window:", ["1M", "3M", "6M", "1Y", "5Y"], index=1, horizontal=True, key="tf_research")
        window1 = tf_map[tf_choice1]

        plot_data = all_close[selected_ticker].tail(window1)
        fig_price = px.line(plot_data, template="plotly_dark", title=f"{selected_ticker} - Last {tf_choice1}")
        fig_price.update_layout(xaxis_title=None, yaxis_title="Price ($)")
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col_r:
        st.subheader("Key Statistics")
        st.metric(label="⚠️ Next Earnings (Volatility Alert)", value=e_date, help="High volatility is expected on reporting dates.")
        st.divider()
        st.metric("Market Cap", f"${main_info.get('marketCap', 0):,.0f}", help="Total dollar value of all outstanding shares.")
        st.metric("P/E Ratio (TTM)", main_info.get('trailingPE', 'N/A'), help="Price-to-Earnings ratio valuation.")
        st.metric("Beta (5Y)", main_info.get('beta', 'N/A'), help="Volatility relative to S&P 500.")
        st.metric("1Y Target Est", f"${main_info.get('targetMeanPrice', 'N/A')}", help="Consensus analyst price forecast.")

# --- TAB 2: Performance Comparison ---
with tab2:
    tf_choice2 = st.radio("Select Window:", ["1M", "3M", "6M", "1Y", "5Y"], index=1, horizontal=True, key="tf_perf")
    window2 = tf_map[tf_choice2]
    
    comp_df = all_close.tail(window2)
    norm_df = (comp_df / comp_df.iloc[0]) * 100
    
    fig_comp = go.Figure()
    for t in tickers:
        fig_comp.add_trace(go.Scatter(
            x=norm_df.index, y=norm_df[t], name=t,
            line=dict(width=4 if t == selected_ticker else 1, color='white' if t == selected_ticker else None),
            opacity=1 if t == selected_ticker else 0.5
        ))
    fig_comp.update_layout(template="plotly_dark", title=f"Growth of $100 over {tf_choice2}", yaxis_title="Rebased Price")
    st.plotly_chart(fig_comp, use_container_width=True)

# --- TAB 3: Strategic Optimizer ---
with tab3:
    st.subheader("Mean-Variance Scenario Simulator")
    st.info("💡 Diversification Cap: Max 40% per asset to prevent concentration risk.")
    
    # Optimizer doesn't need a timeframe selector as it uses the full historical COV matrix for stability
    mu_hist = expected_returns.mean_historical_return(all_close)
    S = risk_models.sample_cov(all_close)
    
    in_cols = st.columns(len(tickers))
    views = {t: in_cols[i].number_input(f"{t} %", value=float(mu_hist[t]*100), step=1.0) for i, t in enumerate(tickers)}
    
    try:
        custom_mu = pd.Series({t: v/100 for t, v in views.items()})
        ef = EfficientFrontier(custom_mu, S)
        ef.add_constraint(lambda w: w <= 0.40) # Forced Diversification
        
        weights = ef.max_sharpe()
        cleaned = ef.clean_weights()
        ret, vol, sha = ef.portfolio_performance()
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write("### Optimal Weights")
            st.table(pd.Series({k: v for k, v in cleaned.items() if v > 0}, name="Allocation").apply(lambda x: f"{x:.1%}"))
            st.metric("Expected Return", f"{ret:.1%}")
            st.metric("Sharpe Ratio", f"{sha:.2f}")
        with c2:
            fig_pie = px.pie(names=list(cleaned.keys()), values=list(cleaned.values()), hole=0.4, 
                             template="plotly_dark", title="Recommended Portfolio Mix")
            st.plotly_chart(fig_pie, use_container_width=True)
    except:
        st.error("Optimization Error: Ensure at least one stock has a positive return expectation.")
