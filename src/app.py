import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pypfopt import EfficientFrontier, risk_models, expected_returns

# --- Page Config & Styling ---
st.set_page_config(page_title="Mag7 Quant Terminal", layout="wide")

# Custom CSS for high-visibility alerts and layout
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.6rem; color: #ff4b4b; }
    .stRadio > div { flex-direction: row; justify-content: flex-start; gap: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Mag7 Research Terminal & Strategic Optimizer")

# --- Global Sidebar ---
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
selected_ticker = st.sidebar.selectbox("Primary Focus Ticker", tickers)

# --- Enhanced Data Engine ---
@st.cache_data(ttl=3600)
def fetch_terminal_data(ticker_list, focus_ticker):
    # Historical Close Prices
    all_data = yf.download(ticker_list, period="5y", multi_level_index=False)['Close']
    
    # Metadata for focus stock
    focus_obj = yf.Ticker(focus_ticker)
    info = focus_obj.info
    
    # Robust Earnings Logic
    try:
        cal = focus_obj.calendar
        next_e = cal.iloc[0, 0].strftime('%Y-%m-%d') if isinstance(cal, pd.DataFrame) else "Check IR"
    except:
        next_e = info.get('nextEarningsDate', "TBD (Post-Reporting)")
        
    return all_data, info, next_e

all_close, main_info, e_date = fetch_terminal_data(tickers, selected_ticker)

# --- Global Components ---
def get_timeframe_selector(key):
    tf_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "5Y": 1825}
    choice = st.radio("Analysis Window:", list(tf_map.keys()), index=1, horizontal=True, key=key)
    return tf_map[choice]

# --- App Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Research", "🔄 Performance", "⚖️ Optimizer", "🔥 Risk Heatmap"])

# --- TAB 1: Equity Research ---
with tab1:
    col_l, col_r = st.columns([2, 1])
    with col_l:
        window = get_timeframe_selector("tf_research")
        fig_price = px.line(all_close[selected_ticker].tail(window), template="plotly_dark", title=f"{selected_ticker} Price Trend")
        fig_price.update_layout(xaxis_title=None, yaxis_title="Price ($)")
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col_r:
        st.subheader("Key Statistics")
        st.metric(label="⚠️ Next Earnings (Volatility Alert)", value=e_date, help="High volatility is expected on reporting dates.")
        st.divider()
        
        # Tooltips added for Financial Engineering definitions
        st.metric("Market Cap", f"${main_info.get('marketCap', 0):,.0f}", 
                  help="Total dollar value of all outstanding shares. P = Price x Shares.")
        
        st.metric("P/E Ratio (TTM)", main_info.get('trailingPE', 'N/A'), 
                  help="Price-to-Earnings ratio. Measures the price paid for $1 of profit.")
        
        st.metric("Beta (5Y)", main_info.get('beta', 'N/A'), 
                  help="Volatility relative to S&P 500. >1.0 means higher risk than the market.")
        
        st.metric("1Y Target Est", f"${main_info.get('targetMeanPrice', 'N/A')}", 
                  help="Average analyst price forecast for the next 12 months.")

# --- TAB 2: Performance Comparison ---
with tab2:
    window = get_timeframe_selector("tf_perf")
    norm_df = (all_close.tail(window) / all_close.tail(window).iloc[0]) * 100
    
    fig_comp = go.Figure()
    for t in tickers:
        is_focus = (t == selected_ticker)
        fig_comp.add_trace(go.Scatter(
            x=norm_df.index, y=norm_df[t], name=t,
            line=dict(width=4 if is_focus else 1, color='white' if is_focus else None),
            opacity=1 if is_focus else 0.5
        ))
    fig_comp.update_layout(template="plotly_dark", title="Growth of $100 (Rebased)", yaxis_title="Rebased Price")
    st.plotly_chart(fig_comp, use_container_width=True)

# --- TAB 3: Strategic Optimizer ---
with tab3:
    st.subheader("Mean-Variance Scenario Simulator")
    st.info("💡 Constrained Optimization: Max 40% per asset to ensure diversification.")
    
    mu_hist = expected_returns.mean_historical_return(all_close)
    S = risk_models.sample_cov(all_close)
    
    in_cols = st.columns(len(tickers))
    views = {t: in_cols[i].number_input(f"{t} %", value=float(mu_hist[t]*100), step=1.0) for i, t in enumerate(tickers)}
    
    try:
        custom_mu = pd.Series({t: v/100 for t, v in views.items()})
        ef = EfficientFrontier(custom_mu, S)
        
        # Diversification constraint to prevent 100% NVDA bias
        ef.add_constraint(lambda w: w <= 0.40) 
        
        weights = ef.max_sharpe()
        cleaned = ef.clean_weights()
        ret, vol, sha = ef.portfolio_performance()
        
        res_l, res_r = st.columns([1, 2])
        with res_l:
            st.write("### Optimal Weights")
            st.table(pd.Series({k: v for k, v in cleaned.items() if v > 0}, name="Allocation").apply(lambda x: f"{x:.1%}"))
            st.metric("Expected Return", f"{ret:.1%}")
            st.metric("Sharpe Ratio", f"{sha:.2f}")
        with res_r:
            fig_pie = px.pie(names=list(cleaned.keys()), values=list(cleaned.values()), hole=0.4, 
                             template="plotly_dark", title="Recommended Portfolio Mix")
            st.plotly_chart(fig_pie, use_container_width=True)
    except:
        st.error("Scenario is mathematically impossible. Ensure at least one stock has a positive return.")

# --- TAB 4: Risk Heatmap ---
with tab4:
    window = get_timeframe_selector("tf_heat")
    returns = all_close.tail(window).pct_change().dropna()
    corr_matrix = returns.corr().round(2)
    
    st.subheader("🔥 Systemic Risk Heatmap")
    
    fig_heat = px.imshow(
        corr_matrix, text_auto=True, aspect="auto",
        color_continuous_scale='RdYlGn', template="plotly_dark"
    )
    fig_heat.update_layout(height=500, margin=dict(l=0, r=0, b=0, t=30))
    st.plotly_chart(fig_heat, use_container_width=True)
    
    avg_corr = corr_matrix.values[np.triu_indices_len(corr_matrix, k=1)].mean()
    st.metric("Average Group Correlation", f"{avg_corr:.2f}", help="Closer to 1.0 means the stocks move in lockstep.")
