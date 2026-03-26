import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# --- Page Setup ---
st.set_page_config(page_title="Mag7 Quant Terminal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    div[data-testid="stMetric"] { background-color: #1c2128; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    /* Tooltip Icon Styling */
    .stTooltipIcon { color: #58a6ff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Data Engine ---
@st.cache_data(ttl=3600)
def fetch_global_meta(ticker_list):
    prices_df = yf.download(ticker_list, period="5y", multi_level_index=False)['Close']
    meta_store = {}
    for t in ticker_list:
        obj = yf.Ticker(t)
        info = obj.info
        meta_store[t] = {
            "Current": info.get('currentPrice') or info.get('regularMarketPrice'),
            "Target": info.get('targetMeanPrice', 0),
            "Beta": info.get('beta', 1.0),
            "Volatility": (prices_df[t].pct_change().std() * np.sqrt(252)),
            "PE": info.get('trailingPE', 'N/A')
        }
    return prices_df, meta_store

# --- Initialization ---
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
selected_ticker = st.sidebar.selectbox("Active Security", tickers)

with st.spinner("Streaming Market Data..."):
    all_prices, all_meta = fetch_global_meta(tickers)

tab1, tab2, tab3 = st.tabs(["📈 PERFORMANCE", "📊 SECTOR COMPARISON", "⚖️ STRATEGIC OPTIMIZER"])

# --- TAB 1: Performance Analysis ---
with tab1:
    m = all_meta[selected_ticker]
    
    # 1Y Price Target Logic
    target_val = m['Target']
    current_val = m['Current']
    
    # Help Definitions
    help_target = """
    **1Y Price Target:** The average 12-month price estimate issued by Wall Street analysts. 
    It represents the 'fair value' expected based on earnings forecasts and market conditions.
    """
    
    c1, c2, c3, c4 = st.columns(4)
    
    # Current Price
    c1.metric("Current Price", f"${current_val:.2f}", help="The most recent market trading price.")
    
    # 1Y Price Target with Delta (Upside/Downside)
    if target_val and target_val > 0:
        upside_pct = ((target_val / current_val) - 1) * 100
        c2.metric(
            label="1Y Price Target", 
            value=f"${target_val:.2f}", 
            delta=f"{upside_pct:.1f}% Upside",
            help=help_target
        )
    else:
        c2.metric("1Y Price Target", "N/A", help="No analyst consensus data available.")

    # Risk Metrics
    c3.metric("Beta (β)", f"{m['Beta']:.2f}", help="Sensitivity to market moves. >1.0 is aggressive.")
    c4.metric("Volatility (σ)", f"{m['Volatility']:.1%}", help="Annualized intensity of price swings.")

    st.markdown("---")
    
    # Chart with Target Line
    fig = px.line(all_prices[selected_ticker].tail(252), template="plotly_dark", title=f"{selected_ticker} Performance vs. Analyst Target")
    fig.update_traces(line_color='#58a6ff')
    
    # Add a horizontal line for the Target Price if it exists
    if target_val > 0:
        fig.add_hline(y=target_val, line_dash="dash", line_color="#d29922", 
                      annotation_text=f"Analyst Target: ${target_val}", 
                      annotation_position="bottom right")
        
    st.plotly_chart(fig, use_container_width=True)
