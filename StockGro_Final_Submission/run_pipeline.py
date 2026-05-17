"""Run the full StockGro capstone pipeline from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TICKERS, ensure_directories
from src.data_fetch import fetch_universe_data
from src.evaluation import make_actual_results_template, make_stockgro_execution_template
from src.features import compute_risk_metrics, compute_selection_metrics, try_garch_volatility
from src.forecasting import run_forecasting_pipeline
from src.portfolio import build_allocation_strategies, prepare_portfolio_inputs
from src.preprocessing import preprocess_universe, run_adf_tests
from src.visualization import generate_core_figures


def run(force_download: bool = False) -> None:
    """Execute data, model, portfolio, and reporting artifacts."""

    ensure_directories()
    print("Fetching/caching yfinance data...")
    raw_data = fetch_universe_data(TICKERS, force=force_download)
    if not raw_data:
        raise RuntimeError(
            "No market data available. Install yfinance, enable network access, "
            "or place cached ticker CSV files in data/raw/."
        )

    print("Preprocessing and computing diagnostics...")
    cleaned, prices, returns = preprocess_universe(raw_data)
    run_adf_tests(prices, returns)
    selection_metrics = compute_selection_metrics(cleaned, prices, returns)
    compute_risk_metrics(returns)
    try_garch_volatility(returns)

    print("Training validation forecasts and 5-day forecasts...")
    evaluation, validation_predictions, five_day_forecasts = run_forecasting_pipeline(prices)

    print("Building portfolio strategies...")
    best_summary = five_day_forecasts.sort_values(["Ticker", "Horizon"]).groupby("Ticker").tail(1)
    best_model_summary = (
        best_summary[
            [
                "Ticker",
                "Best Model",
                "Last Price",
                "Forecast Price",
                "Forecast Return",
            ]
        ]
        .rename(
            columns={
                "Forecast Price": "Forecast Day 5 Price",
                "Forecast Return": "Forecast Return 5D",
            }
        )
        .merge(
            evaluation.sort_values(["Ticker", "Selection Score"]).groupby("Ticker").head(1)[
                ["Ticker", "RMSE", "MAPE", "Directional Accuracy"]
            ],
            on="Ticker",
            how="left",
        )
        .rename(
            columns={
                "RMSE": "Best Validation RMSE",
                "MAPE": "Best Validation MAPE",
                "Directional Accuracy": "Best Directional Accuracy",
            }
        )
    )
    portfolio_inputs = prepare_portfolio_inputs(best_model_summary, selection_metrics)
    _, strategy_summary, final_portfolio = build_allocation_strategies(portfolio_inputs, returns)

    print("Creating templates and figures...")
    make_stockgro_execution_template(final_portfolio)
    make_actual_results_template(five_day_forecasts)
    generate_core_figures(prices, returns, validation_predictions, final_portfolio)

    print("Done.")
    print(f"Final strategy: {final_portfolio['Strategy'].iloc[0] if not final_portfolio.empty else 'N/A'}")
    print(strategy_summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the StockGro time series capstone pipeline.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore cached CSVs and redownload data from yfinance.",
    )
    args = parser.parse_args()
    run(force_download=args.force_download)
