# 🚀 Mag7 Research Terminal & Strategic Optimizer

An interactive Quant-Finance dashboard built to analyze, compare, and optimize the "Magnificent 7" technology equities (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA). 

This application bridges the gap between **Fundamental Equity Research** and **Modern Portfolio Theory (MPT)** by allowing users to simulate forward-looking market scenarios.

---

## 📊 Core Features

### 1. Equity Research Terminal
* **Real-time Ingestion:** Integrated with the Yahoo Finance API to pull live pricing and corporate metadata.
* **Fundamental Analysis:** Displays TTM PE Ratios, Beta (5Y Monthly), EPS, and 1Y Analyst Target Estimates.
* **Event Tracking:** Dynamic monitoring of upcoming Earnings Dates and Ex-Dividend dates.

### 2. Relative Performance Comparison
* **Vectorized Normalization:** Uses a "Base 100" rebasing technique to compare the growth of $100 across different stock price scales.
* **Interactive Visuals:** Built with Plotly for x-unified hovering and detailed time-series exploration.

### 3. Strategic Portfolio Optimizer
* **Mean-Variance Optimization (MVO):** Implements the Markowitz framework to find the "Tangency Portfolio."
* **Scenario Simulator:** A custom-built engine allowing users to input their own "Expected Returns" to see how the model reallocates capital to maximize the **Sharpe Ratio**.
* **Risk Metrics:** Calculates expected Annual Return, Volatility (Risk), and risk-adjusted performance.

---

## 🛠️ Technical Stack

* **Language:** Python 3.11+
* **Framework:** [Streamlit](https://streamlit.io/) (Web UI)
* **Data Science:** Pandas, NumPy
* **Finance Logic:** [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) (Quadratic Programming)
* **Visualization:** Plotly Open-Source Graphing Library
* **API:** yfinance (Real-time market data)

---

## 🚀 Getting Started

### Local Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/mag7-optimizer.git](https://github.com/your-username/mag7-optimizer.git)
   cd mag7-optimizer

---
*Developed by Truc Nguyen, MSFinMath, MSc candidate in AI - Data Analytics specialization, Product Support Specialist & Knowledge Lead at Moody's Analytics, Inc.*
