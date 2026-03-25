import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="Mag7 Research Terminal", layout="wide")

st.title("🚀 Magnificent 7 Research Terminal")

# --- Sidebar ---
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
selected_ticker = st.sidebar.selectbox("Select Primary Ticker", tickers)

timeframe_options = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365, "Max (5Y)": 1825}
selected_label = st.sidebar.selectbox("Timeframe", list(timeframe_options.keys()), index=1)
lookback_days = timeframe_options[selected_label]

# --- Optimized Data Fetching ---
@st.cache_data(ttl=3600)
def get_all_mag7_data():
    # Download all 7 at once for the comparison feature
    df = yf.download(tickers, period="5y", multi_level_index=False)
    # yfinance returns a MultiIndex if multiple tickers are downloaded. 
    # We'll extract 'Close' prices for simplicity.
    return df['Close']

all_close_prices = get_all_mag7_data()

# --- Tab Layout ---
tab1, tab2 = st.tabs(["Single Stock Analysis", "Mag7 Comparison"])

with tab1:
    # (Existing single stock logic goes here - metrics, chart, stats)
    st.info(f"Detailed view for {selected_ticker} is active.")
    # For brevity, I'm focusing on the new Comparison feature below.

with tab2:
    st.subheader("Relative Performance Comparison (Normalized)")
    st.markdown("All stocks are rebased to **100** at the start of the selected period to show relative % growth.")

    # 1. Slice data based on timeframe
    comp_df = all_close_prices.tail(lookback_days).copy()

    # 2. Normalize: (Current Price / Starting Price) * 100
    normalized_df = (comp_df / comp_df.iloc[0]) * 100

    # 3. Build Multi-Line Plotly Chart
    fig_comp = go.Figure()

    for ticker in tickers:
        # Highlight the selected ticker with a thicker line
        is_selected = (ticker == selected_ticker)
        fig_comp.add_trace(go.Scatter(
            x=normalized_df.index, 
            y=normalized_df[ticker],
            name=ticker,
            line=dict(width=4 if is_selected else 1.5),
            opacity=1 if is_selected else 0.7
        ))

    fig_comp.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        yaxis_title="Normalized Price (Base 100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # 4. Summary Table of Returns
    st.write("### Total Return in Period")
    returns = ((comp_df.iloc[-1] / comp_df.iloc[0]) - 1) * 100
    returns_df = pd.DataFrame(returns).transpose()
    returns_df.index = ['Return %']
    st.dataframe(returns_df.style.format("{:.2f}%").background_gradient(cmap='RdYlGn', axis=1))
