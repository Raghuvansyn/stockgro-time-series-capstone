# StockGro Time Series Capstone Project

A complete time-series forecasting and portfolio construction project for the StockGro virtual trading challenge.

- Final Code + Report Drive Folder: [Add Google Drive link here]
- Bonus Dashboard: [Add Streamlit/GitHub dashboard link here]
- GitHub Repository: https://github.com/Raghuvansyn/stockgro-time-series-capstone

This project is for educational virtual trading only. It is not financial advice and does not use real-money trading APIs.

## Project Overview

The project builds an end-to-end workflow for a virtual StockGro portfolio with INR 10,00,000 starting capital. It fetches NSE-listed stock data, preprocesses time series, evaluates forecasting models, estimates volatility and diversification metrics, constructs portfolio allocation strategies, and provides templates for StockGro execution and predicted-vs-actual comparison.

## Key Project Settings

- Dataset period: 2021-01-01 to 2025-12-31
- Training split: before 2025-07-01
- Validation/test split: 2025-07-01 to 2025-12-31
- Forecast horizon: next 5 trading days
- Portfolio capital: INR 10,00,000
- Final selected strategy: Strategy C, Correlation-Diversified
- Target capital: INR 10,00,000
- Estimated executed cost: INR 9,76,262.78
- Virtual cash reserve: INR 23,737.22

## Stock Universe

The universe uses 12 liquid NSE stocks diversified across Banking, IT, Pharma, Energy, FMCG, Auto, Metals, and Infrastructure:

- `HDFCBANK.NS`, `ICICIBANK.NS`
- `TCS.NS`, `INFY.NS`
- `SUNPHARMA.NS`, `DRREDDY.NS`
- `RELIANCE.NS`
- `ITC.NS`, `HINDUNILVR.NS`
- `MARUTI.NS`
- `TATASTEEL.NS`
- `LT.NS`

Selection is justified with average return, annualized volatility, rolling 20-day volatility, liquidity proxy, return correlation, Sharpe-like score, and log-price trend strength.

## Methodology Summary

1. Data fetching with `yfinance`, with one cached CSV per ticker in `data/raw/`.
2. Preprocessing with adjusted-close preference, missing-value handling, and log-return creation.
3. ADF stationarity testing on price levels, first differences, and log returns.
4. Exploratory plots for price trends, returns, rolling statistics, STL decomposition, and correlation.
5. Forecasting with Naive, Moving Average, Exponential Smoothing, and ARIMA when available.
6. Model evaluation using RMSE, MAPE, and directional accuracy.
7. Volatility analysis using annualized volatility, rolling volatility, Sharpe-like scores, and correlation.
8. Portfolio allocation with forecast-guided, volatility-aware, and correlation-diversified strategies.
9. Predicted-vs-actual comparison using a manual Day 2 StockGro close template.

GARCH was skipped gracefully because the optional `arch` package is not installed. Prophet and LSTM are optional and non-blocking; the core project does not depend on them.

## Final Allocation Summary

The selected portfolio is saved in `outputs/tables/final_recommended_portfolio.csv`.

| Metric | Value |
|---|---:|
| Selected strategy | Strategy C: Correlation-Diversified |
| Target capital | INR 10,00,000 |
| Estimated executed cost | INR 9,76,262.78 |
| Virtual cash reserve | INR 23,737.22 |

## Setup

```bash
cd stockgro-time-series-capstone
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the Pipeline

```bash
python run_pipeline.py
```

To force fresh downloads from yfinance:

```bash
python run_pipeline.py --force-download
```

Open the executed notebook:

```bash
jupyter notebook notebooks/stockgro_capstone_full_pipeline.ipynb
```

## Running the Dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

The dashboard reads generated CSV files from `data/processed/`, `outputs/forecasts/`, and `outputs/tables/` using project-relative paths.

## Deploying on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app.
4. Select the repository: `Raghuvansyn/stockgro-time-series-capstone`
5. Set the main file path as: `dashboard/app.py`
6. Deploy the app.
7. Use the deployed app URL as the bonus dashboard submission link.

## Updating Actual StockGro Results

After manually placing virtual StockGro orders, fill:

- `outputs/tables/stockgro_execution_summary_template.csv`
- `outputs/tables/predicted_vs_actual_market_comparison_template.csv`
- `outputs/tables/stockgro_actuals_template.csv`

Then compute the Day 2 predicted-vs-actual comparison:

```python
from src.evaluation import compute_predicted_actual_comparison

comparison = compute_predicted_actual_comparison(
    five_day_forecasts="outputs/forecasts/all_5day_forecasts.csv",
    actuals_csv="outputs/tables/stockgro_actuals_template.csv",
    output_path="outputs/tables/predicted_vs_actual_comparison.csv",
)
print(comparison)
```

## Submission Contents

The final submission package `StockGro_Final_Submission.zip` contains:

- README and setup instructions
- Executed notebook
- Final report
- Modular source code
- Forecast outputs
- Portfolio tables
- StockGro execution and comparison templates
- Figures and report assets
- Streamlit dashboard

## Folder Structure

```text
stockgro-time-series-capstone/
├── README.md
├── requirements.txt
├── run_pipeline.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── stockgro_capstone_full_pipeline.ipynb
├── reports/
│   └── StockGro_Time_Series_Capstone_Report.md
├── src/
├── dashboard/
│   └── app.py
└── outputs/
    ├── figures/
    ├── tables/
    ├── forecasts/
    └── report_assets/
```

## Date Clarification

The original brief references StockGro execution between May 11 and May 15, 2025, while the modeling dataset runs through December 31, 2025. This project follows the specified modeling split through December 2025 and provides the StockGro execution section as a manual template because the live event window cannot be reproduced automatically after the fact.

## Disclaimer

Forecasts are uncertain, simplified models can be wrong, and virtual-trading results should not be interpreted as investment recommendations.
