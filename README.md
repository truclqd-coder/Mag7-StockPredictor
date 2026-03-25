# 📈 Mag7 Quant Research Terminal & Portfolio Optimizer

An end-to-end financial analytics dashboard built to analyze, compare, and optimize the **"Magnificent 7"** equities (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA). 

This tool integrates live market data with **Mean-Variance Optimization (MVO)** to provide both descriptive insights and predictive strategic allocations.

---

## 🚀 Key Features

### 1. Equity Research & Volatility Alerts
* **Real-time Fundamentals:** Tracks Market Cap, P/E Ratios, Beta, and 1Y Target Estimates via `yfinance`.
* **Earnings Monitor:** High-visibility alerts for upcoming earnings dates to signal periods of expected high volatility.

### 2. Relative Performance Comparison
* **Vectorized Normalization:** Rebase all 7 stocks to a "Base 100" starting point to compare true percentage growth across different price scales.
* **Interactive Time-Series:** Dynamic Plotly charts for granular historical analysis.

### 3. Strategic Portfolio Optimizer (Scenario Simulator)
* **Markowitz Framework:** Calculates the **Maximum Sharpe Ratio** portfolio using the `PyPortfolioOpt` engine.
* **Active View Simulation:** Allows users to input custom **Expected Returns (%)** to see how the mathematical "optimal" weights shift in real-time.

### 4. Risk Diagnostics (Correlation Heatmap)
* **Systemic Risk Analysis:** A daily-return correlation matrix to identify sector concentration and diversification gaps within the Mag7.

---

## 🛠️ Tech Stack

* **Language:** Python 3.11+
* **Framework:** Streamlit (Web UI)
* **Optimization:** PyPortfolioOpt (Quadratic Programming)
* **Data Science:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Plotly, Seaborn, Matplotlib

---

## 📂 Project Structure

- `src/app.py`: Main application logic and Streamlit UI.
- `requirements.txt`: Project dependencies.
- `README.md`: Project documentation and financial logic overview.

---

## 💡 Financial Engineering Insights
This project demonstrates the **"Instability Problem"** of Mean-Variance Optimization. By using the **Scenario Simulator**, users can see how sensitive the Efficient Frontier is to changes in expected return inputs—a core concept in institutional Asset-Liability Management (ALM).

---

## 👤 Author
**Truc Nguyen** *Product Support Specialist in Financial Engineering | M.S. Finance | M.S. AI & Data Analytics Candidate*
