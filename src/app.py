import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# Attempt to import optimization library
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False

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
selected_ticker = st.sidebar.selectbox("Active Security", tickers)

with st.spinner("Processing Quant Data..."):
    all_prices_5y, all_meta = fetch_global_meta(tickers)
    m = all_meta[selected_ticker]

st.markdown('<div class="terminal-title">MAG7 QUANTITATIVE TERMINAL v1.0</div>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["📈 PERFORMANCE", "📊 COMPARISON", "⚖️ OPTIMIZER", "📁 PORTFOLIO"])

# --- TAB 1 (RESTORED BETA/VOL + TOOLTIPS) ---
with tab1:
    col_l, col_r = st.columns([2.5, 1])
    with col_l:
        h1 = st.radio("HORIZON", ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y"], index=5, horizontal=True)
        t1_data = yf.download(selected_ticker, period=h1.lower() if 'Y' in h1 else h1, multi_level_index=False)['Close']
        st.plotly_chart(px.line(t1_data, template="plotly_dark").update_layout(yaxis_title="USD"), use_container_width=True)
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Prev Close", f"${m['PrevClose']:.2f}"); v2.metric("EPS", f"${m['EPS']:.2f}"); v3.metric("P/E Ratio", f"{m['PE']:.2f}"); v4.metric("Div Yield", f"{m['DivYield']:.2%}")
    with col_r:
        st.markdown(f'<div class="earnings-banner"><small>NEXT EARNINGS</small><div class="earnings-date">{m["NextEarnings"]}</div></div>', unsafe_allow_html=True)
        r_color = "#3fb950" if "BUY" in m['Rating'] else "#f85149"
        st.markdown(f'<div class="rating-card" style="border-color:{r_color}; color:{r_color};"><div class="rating-text">{m["Rating"]}</div><div class="analyst-subtext">Consensus of {m["Analysts"]} Analysts</div></div>', unsafe_allow_html=True)
        st.metric("Current Price", f"${m['Current']:.2f}")
        upside = ((m['Target'] / m['Current']) - 1) * 100 if m['Current'] > 0 else 0
        st.markdown(f'<div class="target-card"><small>1Y TARGET</small><br><span style="font-size:24px; font-weight:bold;">${m["Target"]:.2f}</span><br><span style="color:{"#3fb950" if upside >=0 else "#f85149"}; font-weight:bold;">{upside:+.1f}% Upside</span></div>', unsafe_allow_html=True)
        st.metric("Beta (β)", f"{m['Beta']:.2f}", help="Measures asset sensitivity to market moves."); st.metric("Ann. Volatility (σ)", f"{m['Volatility']:.1%}", help="Annualized risk based on historical standard deviation.")

# --- TAB 2 (STAYED THE SAME) ---
with tab2:
    def color_ret(val): return f'color: {"#f85149" if val < 0 else "#3fb950"}; font-weight: bold'
    summary_df = pd.DataFrame([{"Ticker": t, "1Y Return (%)": all_meta[t]['Return1Y'] * 100, "Rating": all_meta[t]['Rating'], "Beta": round(all_meta[t]['Beta'], 2)} for t in tickers])
    st.dataframe(summary_df.style.format({"1Y Return (%)": "{:.1f}%"}).applymap(color_ret, subset=['1Y Return (%)']), use_container_width=True)

# --- TAB 3 (RESTORED CODES + GLOSSARY) ---
with tab3:
    if OPTIMIZER_AVAILABLE:
        st.subheader("Modern Portfolio Strategy (MPT)")
        o_in, o_out = st.columns([1, 2.2])
        with o_in:
            mu_hist = expected_returns.mean_historical_return(all_prices_5y)
            user_views = {t: st.number_input(f"{t} Exp. Return %", value=float(mu_hist[t]*100), help=f"Adjust the expected annual return for {t}.") for t in tickers}
            target_p = st.slider("Target Portfolio Return %", 5, 100, 25, help="Select the annual return goal for the optimizer to solve for.")
            mode = st.toggle("Maximize Sharpe Ratio", help="Ignores target return to find the most risk-efficient portfolio.")
        with o_out:
            try:
                mu, S = pd.Series({t: v/100 for t, v in user_views.items()}), risk_models.sample_cov(all_prices_5y)
                ef = EfficientFrontier(mu, S)
                weights = ef.max_sharpe() if mode else ef.efficient_return(target_return=target_p/100)
                ret, vol, sha = ef.portfolio_performance()
                
                # Hero Card with help text
                st.markdown(f'<div class="hero-return-card"><small title="The weighted average return of the optimized assets.">CALCULATED PORTFOLIO RETURN</small><div style="color:#58a6ff; font-size:48px; font-weight:900;">{ret:.1%}</div></div>', unsafe_allow_html=True)
                
                m1, m2 = st.columns(2)
                m1.metric("Risk (Volatility)", f"{vol:.1%}", help="Annualized portfolio standard deviation. Lower is typically safer.")
                m2.metric("Sharpe Ratio", f"{sha:.2f}", help="The excess return per unit of risk. Higher is better (generally > 1.0 is good).")
                st.plotly_chart(px.pie(names=list(ef.clean_weights().keys()), values=list(ef.clean_weights().values()), hole=0.5, template="plotly_dark", title="Optimal Asset Allocation"), use_container_width=True)
            except: st.warning("Target return is mathematically infeasible based on these inputs.")

# --- TAB 4 (STAYED THE SAME) ---
with tab4:
    st.subheader("📁 Personal Asset Tracker")
    my_holdings = [{"Ticker": "NVDA", "Shares": 61, "BuyPrice": 110.50}, {"Ticker": "MSFT", "Shares": 34, "BuyPrice": 395.20}, {"Ticker": "GOOGL", "Shares": 16, "BuyPrice": 142.00}, {"Ticker": "AMZN", "Shares": 8, "BuyPrice": 168.00}]
    p_rows = [{"Ticker": a['Ticker'], "Total Gain (%)": ((all_meta[a['Ticker']]['Current']/a['BuyPrice'])-1)*100, "1Y Return (%)": all_meta[a['Ticker']]['Return1Y']*100, "YTD Return (%)": all_meta[a['Ticker']]['ReturnYTD']*100} for a in my_holdings]
    p_df = pd.DataFrame(p_rows)
    cols = ['Total Gain (%)', '1Y Return (%)', 'YTD Return (%)']
    st.dataframe(p_df.style.format({c: "{:.1f}%" for c in cols}).applymap(color_ret, subset=cols), use_container_width=True)
