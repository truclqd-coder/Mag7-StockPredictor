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
    
    # Quantitative Calculations
    daily_returns = prices.pct_change().dropna()
    vol_series = daily_returns.std() * np.sqrt(252)
    annual_return = (prices.iloc[-1] / prices.iloc[-252] - 1) # 1-Year Trailing Return
    
    meta_store = {}
    for t in ticker_list:
        obj = yf.Ticker(t)
        info = obj.info
        
        # Rating Proxy: Using Consensus Recommendation
        raw_rating = info.get('recommendationKey', 'N/A').replace('_', ' ').title()
        
        meta_store[t] = {
            "Current": info.get('currentPrice') or info.get('regularMarketPrice'),
            "Target": info.get('targetMeanPrice', 0),
            "Beta": info.get('beta', 1.0),
            "Volatility": vol_series[t],
            "AnnReturn": annual_return[t],
            "PE": info.get('trailingPE', 'N/A'),
            "Rating": raw_rating,
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
    col_l, col_r = st.columns([2.5, 1])
    m = all_meta[selected_ticker]

    with col_l:
        choice1 = st.radio("SELECT TIMEFRAME", horizons, index=3, horizontal=True, key="p_tf")
        plot_data = fetch_performance_data(selected_ticker, choice1)
        fig_price = px.line(plot_data, template="plotly_dark")
        fig_price.update_traces(line_color='#58a6ff', line_width=2)
        fig_price.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col_r:
        st.metric("Current Price", f"${m['Current']:.2f}", help="Last traded market price.")
        
        # 1Y Target with Help & Delta
        upside = ((m['Target'] / m['Current']) - 1) * 100 if m['Target'] > 0 else 0
        st.metric("1Y Price Target", f"${m['Target']:.2f}", delta=f"{upside:.1f}% Upside", 
                  help="Average 12-month analyst estimate. (?) Measures predicted growth.")
        
        # Ratings Section (Morningstar/Moody's Proxy)
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
        st.metric("Beta (β)", f"{m['Beta']:.2f}", help="Sensitivity to S&P 500 moves. (?) 1.0 is market average.")
        st.metric("Ann. Volatility (σ)", f"{m['Volatility']:.1%}", help="Total risk via price swing intensity.")

# --- TAB 2: Mag7 Comparison ---
with tab2:
    st.subheader("Sector Benchmark Comparison")
    choice2 = st.radio("BENCHMARK HORIZON", horizons, index=3, horizontal=True, key="c_tf")
    days_map = {"1D": 1, "5D": 5, "7D": 7, "1M": 21, "3M": 63, "6M": 126, "1Y": 252, "5Y": 1260}
    comp_df = all_prices_5y.tail(days_map[choice2])
    norm_df = (comp_df / comp_df.iloc[0]) * 100
    fig_comp = px.line(norm_df, template="plotly_dark")
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.divider()
    # Updated table with Annual Return and Rating
    df_disp = pd.DataFrame([{
        "Ticker": t, 
        "Rating": all_meta[t]['Rating'],
        "Price": f"${all_meta[t]['Current']:.2f}",
        "1Y Return": f"{all_meta[t]['AnnReturn']:.1%}",
        "Beta": round(all_meta[t]['Beta'], 2),
        "Vol %": f"{all_meta[t]['Volatility']:.1%}",
        "P/E": all_meta[t]["PE"]
    } for t in tickers])
    st.dataframe(df_disp, use_container_width=True)

# --- TAB 3: Strategic Optimizer ---
with tab3:
    if not OPTIMIZER_AVAILABLE:
        st.error("Quant Library Missing.")
    else:
        st.subheader("Strategic Portfolio Allocation")
        c_in, c_out = st.columns([1, 2])
        with c_in:
            mu_hist = expected_returns.mean_historical_return(all_prices_5y)
            user_views = {t: st.number_input(f"{t} Return %", value=float(mu_hist[t]*100), step=1.0) for t in tickers}
            target_p = st.slider("Target Portfolio Return %", 5, 100, 25)
            mode = st.toggle("Maximize Sharpe Ratio", value=False)

        with c_out:
            try:
                mu = pd.Series({t: v/100 for t, v in user_views.items()})
                S = risk_models.sample_cov(all_prices_5y)
                ef = EfficientFrontier(mu, S)
                ef.add_constraint(lambda w: w <= 0.40)
                weights = ef.max_sharpe() if mode else ef.efficient_return(target_return=target_p/100)
                cleaned = ef.clean_weights()
                ret, vol, sha = ef.portfolio_performance()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Exp. Return", f"{ret:.1%}")
                col2.metric("Volatility", f"{vol:.1%}")
                col3.metric("Sharpe Ratio", f"{sha:.2f}")
                
                st.table(pd.DataFrame([{"Ticker": t, "Weight": f"{w*100:.1f}%"} for t, w in cleaned.items() if w > 0]))
                st.plotly_chart(px.pie(names=list(cleaned.keys()), values=list(cleaned.values()), hole=0.6, template="plotly_dark"))
            except:
                st.warning("Infeasible target.")
