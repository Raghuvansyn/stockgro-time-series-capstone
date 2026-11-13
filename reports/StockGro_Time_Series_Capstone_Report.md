# Time Series Forecasting Based Portfolio Construction for StockGro

**Disclaimer:** This report is for an educational virtual-trading exercise only. It is not financial advice and does not use real-money trading APIs.

## 1. Executive Summary

This project builds an end-to-end time series pipeline for a virtual INR 10,00,000 StockGro portfolio. It uses NSE-listed equities from 2021-01-01 to 2025-12-31, trains forecasting models before 2025-07-01, validates them on 2025-07-01 to 2025-12-31, and forecasts the next 5 business days beyond the dataset. The final portfolio is selected from forecast-guided, volatility-aware, and correlation-diversified strategies subject to long-only allocation constraints.

## 2. Project Objective

The objective is to combine time series forecasting with risk-aware portfolio construction. The workflow fetches historical data, cleans and tests stationarity, evaluates forecasting models, estimates risk metrics, allocates virtual capital, and provides a manual StockGro execution and predicted-vs-actual comparison template.

**Assumptions and Date Clarification:** The challenge brief mentions StockGro execution between May 11 and May 15, 2025, while the modeling dataset extends to December 31, 2025. This project follows the provided modeling split through December 2025. The StockGro execution section is provided as a manual template because live event execution cannot be reproduced automatically after the event window.

## 3. Stock Universe and Selection Rationale

The starting universe contains 12 liquid NSE stocks diversified across Banking, IT, Pharma, Energy, FMCG, Auto, Metals, and Infrastructure: HDFC Bank, ICICI Bank, TCS, Infosys, Sun Pharma, Dr. Reddy's, Reliance Industries, ITC, Hindustan Unilever, Maruti Suzuki, Tata Steel, and Larsen & Toubro.

The notebook and pipeline compute selection metrics from historical data:

- Average daily and annualized log returns
- Annualized volatility and rolling 20-day volatility
- Liquidity proxy using traded value where volume is available
- Return correlation and average absolute correlation
- Sharpe-like score with risk-free rate assumed as zero
- Log-price trend strength and trend fit

See `data/processed/selection_metrics.csv` and `data/processed/return_correlation.csv`.

## 4. Data Collection and Preprocessing

Data is fetched using `yfinance` and cached under `data/raw/` so repeated runs do not redownload unnecessarily. The pipeline uses Adjusted Close when available, otherwise Close. Missing observations are forward-filled and back-filled where appropriate, then daily log returns are computed.

ADF tests are run on price levels, first differences, and log returns. Price levels are expected to often behave as non-stationary financial time series, so ARIMA models use differencing through integrated order selection. Log returns are used for risk and correlation analysis.

## 5. Exploratory Time Series Analysis

The EDA section saves:

- Normalized price trends: `outputs/figures/price_trends_normalized.png`
- Daily log returns: `outputs/figures/daily_log_returns.png`
- Rolling mean and volatility for a representative stock
- Correlation heatmap: `outputs/figures/return_correlation_heatmap.png`
- STL decomposition when `statsmodels` is installed

Interpretation is kept empirical: trend, volatility, and correlation are treated as inputs to model selection and allocation rather than as guarantees of future performance.

## 6. Forecasting Methodology

The required models are implemented for every stock:

- Naive baseline: next price equals the latest observed price
- Moving average baseline: 20-day walk-forward moving average
- Exponential smoothing baseline: EWMA with alpha 0.30
- ARIMA: compact AIC-based order search when `statsmodels` is available

Optional packages are handled defensively. If `statsmodels`, `arch`, Prophet, or LSTM dependencies are unavailable, the project does not fail; it records the skip and continues with supported models.

## 7. Model Evaluation and Comparison

Models are evaluated on the validation period from 2025-07-01 to 2025-12-31 using:

- RMSE
- MAPE
- Directional accuracy

The implementation selects per-stock models using the custom score `MAPE * (1 - 0.75 * directional_accuracy)`, then additional sort keys. The comparison mixes one-step walk-forward baselines with fixed-origin ARIMA forecasts over the whole validation period, and the same validation period is used for model selection. These outputs do not establish a fair model ranking, independent test performance, or trading profitability. See [evaluation status and reproduction command](../docs/evaluation.md). Outputs are saved in `outputs/forecasts/model_evaluation.csv`, `outputs/forecasts/validation_predictions.csv`, and `outputs/forecasts/best_model_summary.csv`.

## 8. Volatility, Trend, and Risk Analysis

Risk metrics include annualized volatility, latest 20-day annualized volatility, Sharpe-like score, skewness, kurtosis, and average absolute correlation. GARCH(1,1) is attempted only if the optional `arch` package is installed; otherwise it is skipped with a documented note.

These metrics are used to prevent the final recommendation from relying only on point forecasts.

## 9. Portfolio Allocation Strategies

Three allocation strategies are implemented:

**Strategy A: Forecast-Guided Allocation** ranks stocks by expected 5-day forecast return and gives larger weights to higher positive forecasts.

**Strategy B: Volatility-Aware Allocation** uses inverse volatility sizing so lower-volatility stocks receive higher weights.

**Strategy C: Correlation-Diversified Allocation** combines positive forecast return, volatility, and average correlation penalties to prefer stocks with stronger forecasts and lower overlap with the rest of the basket.

All strategies are long-only and apply max 25% and min 5% constraints for selected holdings. Results are saved in `outputs/tables/portfolio_strategy_comparison.csv` and summarized in `outputs/tables/strategy_summary.csv`.

Current generated comparison:

| Strategy | Holdings | Expected 5D Return | Annualized Portfolio Volatility | Cash After Rounding |
|---|---:|---:|---:|---:|
| A: Forecast-Guided | 10 | 0.47% | 14.86% | INR 24,222.83 |
| B: Volatility-Aware | 10 | 0.25% | 13.79% | INR 24,027.99 |
| C: Correlation-Diversified | 10 | 0.47% | 14.84% | INR 23,737.22 |

## 10. Final Recommended Portfolio

The pipeline selects **Strategy C: Correlation-Diversified** because it has the strongest risk-adjusted expected 5-day return in the generated comparison. The final holdings table includes ticker, company, sector, allocation percentage, rupee allocation, latest price, estimated quantity, forecast return, annualized volatility, model, and rationale.

Final output: `outputs/tables/final_recommended_portfolio.csv`.

Any cash reserve shown in the final table comes from integer-share rounding and is kept as virtual cash rather than forcing an imprecise over-allocation.

| Ticker | Sector | Weight | Target Allocation | Quantity | Forecast 5D Return |
|---|---|---:|---:|---:|---:|
| SUNPHARMA.NS | Pharma | 11.88% | INR 118,769.66 | 69 | 0.50% |
| TCS.NS | IT | 25.00% | INR 250,000.00 | 79 | 1.36% |
| INFY.NS | IT | 25.00% | INR 250,000.00 | 154 | 1.15% |
| ICICIBANK.NS | Banking | 8.12% | INR 81,230.34 | 60 | 0.39% |
| HDFCBANK.NS | Banking | 5.00% | INR 50,000.00 | 50 | 0.04% |
| ITC.NS | FMCG | 5.00% | INR 50,000.00 | 126 | -0.03% |
| MARUTI.NS | Auto | 5.00% | INR 50,000.00 | 2 | -0.47% |
| LT.NS | Infrastructure | 5.00% | INR 50,000.00 | 12 | -0.56% |
| TATASTEEL.NS | Metals | 5.00% | INR 50,000.00 | 277 | -3.14% |
| RELIANCE.NS | Energy | 5.00% | INR 50,000.00 | 31 | -0.86% |

Small positions with negative model forecasts are retained only for sector diversification at the minimum allowed weight.

## 11. StockGro Execution Plan

Manual StockGro execution cannot be automated from this project. Use the template in `outputs/tables/stockgro_execution_template.csv`.

Execution steps:

1. Register or log in to StockGro using mobile number and OTP.
2. Confirm the portfolio uses virtual capital only.
3. Search each selected stock ticker/name.
4. Place market or limit orders based on the recommended quantity.
5. Record execution date, entry price, quantity, and screenshot/reference manually.
6. Observe performance over two consecutive trading days.
7. Record Day 2 closing price for each holding.

## 12. Predicted vs Actual Outcomes

The ready-to-fill comparison template is saved as `outputs/tables/stockgro_actuals_template.csv`. After entering actual Day 2 closes, run:

```python
from src.evaluation import compute_predicted_actual_comparison
compute_predicted_actual_comparison()
```

The generated comparison includes stock, forecasted price, forecasted return, actual Day 2 close, actual return, forecast direction, actual direction, direction correctness, error percentage, and notes.

## 13. Limitations

The forecasts use historical daily prices only and do not include news, events, earnings, macro variables, order-book information, transaction costs, or exchange holidays beyond a simple business-day calendar. ARIMA forecasts are limited by univariate assumptions. Portfolio weights are educational and simplified for a virtual challenge.

## 14. Conclusion

This capstone provides a complete reproducible workflow for time series forecasting, validation, risk analysis, and virtual portfolio construction. It is submission-ready as a notebook-driven data science project and includes the manual StockGro execution templates needed to complete the predicted-vs-actual comparison after virtual trades are observed.
