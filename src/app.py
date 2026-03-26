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

# --- Page Setup & THEME OVERRIDE ---
st.set_page_config(page_title="Mag7 Quant Terminal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: #0e1117; padding: 10px 0; }
    .stTabs [data-baseweb="tab"] { 
        height: 45px; background-color: #1c2128; border-radius: 4px 4px 0 0; 
        color: #8b949e; font-weight: 600; border: 1px solid #30363d; padding: 0 20px;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #21262d !important; color: #58a6ff !important; border-bottom: 3px solid #58a6ff !important;
    }
    div[data-testid="stMetric"] { background-color: #1c2128; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    div[data-testid="stMetricValue"] { color: #3fb950; font-family: 'Courier New', monospace; }
    .rating-card { background-color: #1c2128; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin-top: 10px; }
    .stTooltipIcon { color: #58a6ff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Data Engine ---
@st.cache_data(ttl=300)
def fetch_performance_data(ticker, horizon):
    params = {
        "1D": {"period": "1d", "interval": "1m"}, "5D": {"period": "5d", "interval": "5m"},
        "7D": {"period": "7d", "interval": "15m"}, "1M": {"period": "1mo", "interval": "1h"},
        "3M": {"period": "3mo", "interval": "1d"}, "6M": {"period": "6mo", "interval": "1d"},
        "1Y": {"period": "1y", "interval": "1d"}, "5Y": {"period": "5y", "interval": "1d"}
    }
    config = params.get(horizon, {"period": "1y", "interval": "1d"})
    return yf.download(ticker, **config, multi_level_index=False)['Close']

@st.cache_data(ttl=3600)
def fetch_global_meta(ticker_list):
    prices = yf.download(ticker_list, period="5y", multi_level_index=False)['Close']
    vol_series = prices.pct_change().std() * np.sqrt(252)
    # 1Y Return calculation
    ann_return = (prices.iloc[-1] / prices.iloc[-252] - 1) 
    
    meta_store = {}
    for t in ticker_list:
        obj = yf.Ticker(t)
        info = obj.info
        
        # Helper for date formatting
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
            "Rating": info.get('recommendationKey', 'N/A').replace('_', ' ').title(),
            "Analysts": info.get('numberOfAnalystOpinions', 'N/A')
        }
    return prices, meta_store

# --- Initialization ---
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
horizons = ["1D", "5D", "7D", "1M", "3M", "6M", "1Y", "5Y"]
selected_ticker = st.sidebar.selectbox("Active Security", tickers)

with st.spinner("Synchronizing Market Intelligence..."):
    all_prices_5y, all_meta = fetch_global_meta(tickers)

tab1, tab2, tab3 = st.tabs(["📈 PERFORMANCE", "📊 SECTOR COMPARISON", "⚖️ STRATEGIC OPTIMIZER"])

# --- TAB 1: Performance Analysis ---
with tab1:
    m = all_meta[selected_ticker]
    col_l, col_r = st.columns([2.5, 1])

    with col_l:
        choice1 = st.radio("SELECT TIMEFRAME", horizons, index=3, horizontal=True, key="p_tf")
        plot_data = fetch_performance_data(selected_ticker, choice1)
        fig_price = px.line(plot_data, template="plotly_dark")
        fig_price.update_traces(line_color='#58a6ff', line_width=2)
        fig_price.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Sub-row for extra fundamentals in Tab 1
        st.markdown("### 📊 Valuation & Dividends")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("EPS (TTM)", f"${m['EPS']:.2f}" if isinstance(m['EPS'], (int, float)) else "N/A", help="Earnings Per Share over the Last 12 Months.")
        v2.metric("P/E Ratio", f"{m['PE']:.2f}" if isinstance(m['PE'], (int, float)) else "N/A", help="Price-to-Earnings Ratio. Valuation relative to profit.")
        
        div_str = f"{m['DivRate']} ({m['DivYield']:.2%})" if m['DivYield'] > 0 else "0.00 (0%)"
        v3.metric("Forward Div & Yield", div_str, help="Annual dividend rate and yield percentage.")
        v4.metric("Ex-Dividend Date", m['ExDiv'], help="Last date to buy to be eligible for the next dividend.")

    with col_r:
        st.metric("Next Earnings", m['NextEarnings'], help="Scheduled date for quarterly financial reporting.")
        st.metric("Current Price", f"${m['Current']:.2f}", help="Last traded market price.")
    
        upside = ((m['Target'] / m['Current']) - 1) * 100 if m['Target'] > 0 else 0
        st.metric("1Y Price Target", f"${m['Target']:.2f}", delta=f"{upside:.1f}% Upside", help="Average 12-month analyst estimate.")
        
        st.metric("Market Cap", f"${m['MarketCap']/1e12:.2f}T", help="Total market value of all outstanding shares.")

        st.markdown("### 🏛️ Analyst Rating")
        rating_color = "#3fb950" if "Buy" in m['Rating'] else "#d29922"
        st.markdown(f"""
            <div class="rating-card">
                <small style="color: #8b949e;">CONSENSUS RECOMMENDATION</small><br>
                <span style="color: {rating_color}; font-size: 22px; font-weight: bold;">{m['Rating']}</span><br>
                <small style="color: #8b949e;">Based on {m['Analysts']} Analysts</small>
            </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        st.metric("Beta (β)", f"{m['Beta']:.2f}", help="Sensitivity to S&P 500 moves.")
        st.metric("Ann. Volatility (σ)", f"{m['Volatility']:.1%}", help="Total risk via price swing intensity.")

# --- TAB 2: Mag7 Comparison ---
with tab2:
    st.subheader("Sector Benchmark Comparison")
    # Table now includes 1Y Return, P/E, EPS, etc.
    df_disp = pd.DataFrame([{
        "Ticker": t, 
        "Rating": all_meta[t]['Rating'],
        "Price": f"${all_meta[t]['Current']:.2f}",
        "1Y Return": f"{all_meta[t]['AnnReturn']:.1%}",
        "P/E": all_meta[t]["PE"],
        "EPS": all_meta[t]["EPS"],
        "Div Yield": f"{all_meta[t]['DivYield']:.2%}",
        "Beta": round(all_meta[t]['Beta'], 2)
    } for t in tickers])
    st.dataframe(df_disp, use_container_width=True)
