import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

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

# --- Data Engine ---
@st.cache_data(ttl=300)
def fetch_multi_performance(tickers, horizon):
    params = {
        "1D": {"period": "1d", "interval": "1m"}, "5D": {"period": "5d", "interval": "5m"},
        "1M": {"period": "1mo", "interval": "1h"}, "3M": {"period": "3mo", "interval": "1d"},
        "6M": {"period": "6mo", "interval": "1d"}, "YTD": {"period": "ytd", "interval": "1d"},
        "1Y": {"period": "1y", "interval": "1d"}, "5Y": {"period": "5y", "interval": "1d"}
    }
    config = params.get(horizon, {"period": "1y", "interval": "1d"})
    return yf.download(tickers, **config, multi_level_index=False)['Close']

@st.cache_data(ttl=3600)
def fetch_global_meta(ticker_list):
    prices = yf.download(ticker_list, period="5y", multi_level_index=False)['Close']
    vol_series = prices.pct_change().std() * np.sqrt(252)
    
    def get_ret(days):
        t_date = prices.index[-1] - timedelta(days=days)
        idx = prices.index.get_indexer([t_date], method='backfill')[0]
        return ((prices.iloc[-1] / prices.iloc[idx]) - 1)

    y_start = datetime(datetime.now().year, 1, 1)
    y_idx = prices.index.get_indexer([y_start], method='backfill')[0]
    
    meta_store = {}
    for t in ticker_list:
        obj = yf.Ticker(t)
        info = obj.info
        def clean_date(ts): return pd.to_datetime(ts, unit='s').strftime('%Y-%m-%d') if ts else "N/A"
        meta_store[t] = {
            "Current": info.get('currentPrice', 0),
            "PrevClose": info.get('previousClose', 0),
            "Target": info.get('targetMeanPrice', 0),
            "Beta": info.get('beta', 1.0),
            "Volatility": vol_series[t],
            "PE": info.get('trailingPE', 0),
            "EPS": info.get('trailingEps', 0),
            "NextEarnings": clean_date(info.get('nextEarningsDate') or info.get('earningsTimestamp')),
            "Rating": info.get('recommendationKey', 'N/A').replace('_', ' ').upper(),
            "Analysts": info.get('numberOfAnalystOpinions', 'N/A'),
            "DivYield": info.get('dividendYield', 0),
            "Return1Y": get_ret(365)[t],
            "Return6M": get_ret(180)[t],
            "ReturnYTD": ((prices.iloc[-1][t] / prices.iloc[y_idx][t]) - 1)
        }
    return prices, meta_store

# --- Initialization ---
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
horizons = ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y"]
selected_ticker = st.sidebar.selectbox("Active Security", tickers)

with st.spinner("Processing Quant Data..."):
    all_prices_5y, all_meta = fetch_global_meta(tickers)
    m = all_meta[selected_ticker]

st.markdown('<div class="terminal-title">MAG7 QUANTITATIVE TERMINAL v1.0</div>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["📈 PERFORMANCE", "📊 COMPARISON", "⚖️ OPTIMIZER", "📁 PORTFOLIO"])

# --- TAB 1: RESTORED BETA & VOLATILITY ---
with tab1:
    col_l, col_r = st.columns([2.5, 1])
    with col_l:
        h1 = st.radio("TIME HORIZON", horizons, index=5, horizontal=True)
        t1_data = fetch_multi_performance(selected_ticker, h1)
        st.plotly_chart(px.line(t1_data, template="plotly_dark").update_layout(yaxis_title="USD", hovermode="x unified"), use_container_width=True)
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Prev Close", f"${m['PrevClose']:.2f}"); v2.metric("EPS (TTM)", f"${m['EPS']:.2f}"); v3.metric("P/E Ratio", f"{m['PE']:.2f}"); v4.metric("Div Yield", f"{m['DivYield']:.2%}")
    with col_r:
        st.markdown(f'<div class="earnings-banner"><small>NEXT EARNINGS</small><div class="earnings-date">{m["NextEarnings"]}</div></div>', unsafe_allow_html=True)
        r_color = "#3fb950" if "BUY" in m['Rating'] else "#f85149"
        st.markdown(f'<div class="rating-card" style="border-color:{r_color}; color:{r_color};"><div class="rating-text">{m["Rating"]}</div><div class="analyst-subtext">Consensus of {m["Analysts"]} Analysts</div></div>', unsafe_allow_html=True)
        st.metric("Current Price", f"${m['Current']:.2f}")
        
        upside = ((m['Target'] / m['Current']) - 1) * 100 if m['Current'] > 0 else 0
        u_color = "#3fb950" if upside >= 0 else "#f85149"
        st.markdown(f'<div class="target-card"><small>1Y TARGET PRICE</small><br><span style="font-size:24px; font-weight:bold;">${m["Target"]:.2f}</span><br><span style="color:{u_color}; font-weight:bold;">{upside:+.1f}% Upside</span></div>', unsafe_allow_html=True)
        
        # RESTORED METRICS
        st.metric("Beta (β)", f"{m['Beta']:.2f}", help="Sensitivity relative to S&P 500.")
        st.metric("Ann. Volatility (σ)", f"{m['Volatility']:.1%}", help="Annualized price fluctuation risk.")

# --- TAB 2: COMPARISON ---
with tab2:
    h2 = st.radio("COMPARISON HORIZON", horizons, index=5, horizontal=True, key="t2_h")
    comp_data = fetch_multi_performance(tickers, h2)
    norm_df = (comp_data / comp_data.iloc[0]) * 100
    st.plotly_chart(px.line(norm_df, template="plotly_dark", title=f"Relative Growth Index ({h2})").update_layout(yaxis_title="Performance %"), use_container_width=True)
    summary_df = pd.DataFrame([{"Ticker": t, "1Y Return (%)": all_meta[t]['Return1Y'] * 100, "Next Earnings": all_meta[t]['NextEarnings'], "Rating": all_meta[t]['Rating'], "Beta": round(all_meta[t]['Beta'], 2)} for t in tickers])
    def color_ret(val): return f'color: {"#f85149" if val < 0 else "#3fb950"}; font-weight: bold'
    st.dataframe(summary_df.style.format({"1Y Return (%)": "{:.1f}%"}).applymap(color_ret, subset=['1Y Return (%)']), use_container_width=True)

# --- TAB 3: OPTIMIZER ---
with tab3:
    st.subheader("Modern Portfolio Strategy")
    target_p = st.slider("Target Portfolio Return %", 5, 100, 25)
    st.markdown(f'<div class="hero-return-card"><small>Target Return</small><div style="color:#58a6ff; font-size:48px; font-weight:900;">{target_p}%</div></div>', unsafe_allow_html=True)

# --- TAB 4: PORTFOLIO MANAGER ---
with tab4:
    st.subheader("📁 Asset Performance Tracker")
    my_holdings = [
        {"Ticker": "NVDA", "Shares": 61, "BuyPrice": 110.50},
        {"Ticker": "MSFT", "Shares": 34, "BuyPrice": 395.20},
        {"Ticker": "GOOGL", "Shares": 16, "BuyPrice": 142.00},
        {"Ticker": "AMZN", "Shares": 8, "BuyPrice": 168.00}
    ]
    p_rows = []
    for asset in my_holdings:
        t, curr = asset['Ticker'], all_meta[asset['Ticker']]['Current']
        p_rows.append({
            "Ticker": t, "Shares": asset['Shares'], "Buy Price": f"${asset['BuyPrice']:.2f}",
            "Current Price": f"${curr:.2f}", "Total Gain (%)": ((curr / asset['BuyPrice']) - 1) * 100,
            "1Y Return (%)": all_meta[t]['Return1Y'] * 100,
            "YTD Return (%)": all_meta[t]['ReturnYTD'] * 100,
            "6M Return (%)": all_meta[t]['Return6M'] * 100
        })
    p_df = pd.DataFrame(p_rows)
    cols_to_style = ['Total Gain (%)', '1Y Return (%)', 'YTD Return (%)', '6M Return (%)']
    st.dataframe(p_df.style.format({c: "{:.1f}%" for c in cols_to_style}).applymap(color_ret, subset=cols_to_style), use_container_width=True)
