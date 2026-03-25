import streamlit as st
import yfinance as yf
import pandas as pd
import requests_cache # Optional: pip install requests-cache
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# --- 1. Create a Hardened Session ---
def get_hardened_session():
    session = Session()
    # Spoof a real browser header to avoid the Rate Limit
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    })
    # Add retries for stability
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

custom_session = get_hardened_session()

# --- 2. Update the Data Engine ---
@st.cache_data(ttl=3600)
def fetch_terminal_data(ticker_list, focus_ticker):
    # Pass the custom session to yfinance
    all_data = yf.download(
        ticker_list, 
        period="5y", 
        multi_level_index=False, 
        session=custom_session # <--- CRITICAL FIX
    )['Close']
    
    focus_obj = yf.Ticker(focus_ticker, session=custom_session) # <--- CRITICAL FIX
    info = focus_obj.info
    
    # ... (Rest of your earnings logic remains the same)
    next_e = "N/A"
    try:
        raw_ts = info.get('earningsTimestamp') or info.get('nextEarningsDate')
        if raw_ts:
            next_e = pd.to_datetime(raw_ts, unit='s').strftime('%Y-%m-%d')
    except:
        next_e = "Check IR"
        
    return all_data, info, next_e
