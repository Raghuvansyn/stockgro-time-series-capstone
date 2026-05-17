"""Evaluation metrics and predicted-vs-actual StockGro comparison helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import FORECASTS_DIR, TABLES_DIR, ensure_directories


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """Root mean squared error."""

    aligned = pd.concat([actual, predicted], axis=1).dropna()
    if aligned.empty:
        return np.nan
    errors = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(np.sqrt(np.mean(np.square(errors))))


def mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean absolute percentage error in percent."""

    aligned = pd.concat([actual, predicted], axis=1).dropna()
    aligned = aligned.loc[aligned.iloc[:, 0] != 0]
    if aligned.empty:
        return np.nan
    pct_error = np.abs((aligned.iloc[:, 0] - aligned.iloc[:, 1]) / aligned.iloc[:, 0])
    return float(pct_error.mean() * 100)


def directional_accuracy(
    actual: pd.Series,
    predicted: pd.Series,
    previous_actual: pd.Series,
) -> float:
    """Share of periods where predicted and actual price direction match."""

    aligned = pd.concat([actual, predicted, previous_actual], axis=1).dropna()
    if aligned.empty:
        return np.nan
    actual_direction = np.sign(aligned.iloc[:, 0] - aligned.iloc[:, 2])
    predicted_direction = np.sign(aligned.iloc[:, 1] - aligned.iloc[:, 2])
    return float((actual_direction == predicted_direction).mean())


def evaluate_forecast(
    ticker: str,
    model_name: str,
    actual: pd.Series,
    predicted: pd.Series,
    train_last_price: float,
) -> dict[str, object]:
    """Evaluate a validation forecast with error and direction metrics."""

    prev_actual = actual.shift(1)
    if len(prev_actual) > 0:
        prev_actual.iloc[0] = train_last_price

    model_mape = mape(actual, predicted)
    model_dir_acc = directional_accuracy(actual, predicted, prev_actual)
    selection_score = (
        model_mape * (1 - 0.75 * model_dir_acc)
        if pd.notna(model_mape) and pd.notna(model_dir_acc)
        else np.nan
    )

    return {
        "Ticker": ticker,
        "Model": model_name,
        "RMSE": rmse(actual, predicted),
        "MAPE": model_mape,
        "Directional Accuracy": model_dir_acc,
        "Selection Score": selection_score,
    }


def make_actual_results_template(
    five_day_forecasts: pd.DataFrame,
    output_path: Path = TABLES_DIR / "stockgro_actuals_template.csv",
) -> pd.DataFrame:
    """Create a CSV template for entering Day 2 StockGro actual closing prices."""

    ensure_directories()
    if five_day_forecasts.empty:
        template = pd.DataFrame(
            columns=[
                "Ticker",
                "Execution Date",
                "Entry Price",
                "Quantity",
                "Actual Day 2 Close",
                "Notes",
            ]
        )
    else:
        tickers = sorted(five_day_forecasts["Ticker"].unique())
        template = pd.DataFrame(
            {
                "Ticker": tickers,
                "Execution Date": "",
                "Entry Price": np.nan,
                "Quantity": np.nan,
                "Actual Day 2 Close": np.nan,
                "Notes": "",
            }
        )
    template.to_csv(output_path, index=False)
    return template


def make_stockgro_execution_template(
    final_portfolio: pd.DataFrame,
    output_path: Path = TABLES_DIR / "stockgro_execution_template.csv",
) -> pd.DataFrame:
    """Create the manual execution log template for StockGro orders."""

    ensure_directories()
    if final_portfolio.empty:
        template = pd.DataFrame(
            columns=[
                "Ticker",
                "Company",
                "Sector",
                "Recommended Weight",
                "Target Rupee Allocation",
                "Estimated Quantity",
                "Order Type",
                "Execution Date",
                "Entry Price",
                "Screenshot/Reference",
                "Day 2 Close",
                "Notes",
            ]
        )
    else:
        template = final_portfolio[
            [
                "Ticker",
                "Company",
                "Sector",
                "Weight",
                "Target Rupee Allocation",
                "Estimated Quantity",
            ]
        ].rename(columns={"Weight": "Recommended Weight"})
        template["Order Type"] = "Market or Limit"
        template["Execution Date"] = ""
        template["Entry Price"] = np.nan
        template["Screenshot/Reference"] = ""
        template["Day 2 Close"] = np.nan
        template["Notes"] = ""

    template.to_csv(output_path, index=False)
    return template


def compute_predicted_actual_comparison(
    five_day_forecasts: pd.DataFrame | str | Path = FORECASTS_DIR / "all_5day_forecasts.csv",
    actuals_csv: str | Path = TABLES_DIR / "stockgro_actuals_template.csv",
    output_path: str | Path = TABLES_DIR / "predicted_vs_actual_comparison.csv",
) -> pd.DataFrame:
    """Compare the Day 2 forecast with manually entered StockGro actuals."""

    ensure_directories()
    forecasts = (
        pd.read_csv(five_day_forecasts, parse_dates=["Forecast Date"])
        if not isinstance(five_day_forecasts, pd.DataFrame)
        else five_day_forecasts.copy()
    )
    actuals = pd.read_csv(actuals_csv)

    if forecasts.empty:
        comparison = pd.DataFrame()
        comparison.to_csv(output_path, index=False)
        return comparison

    day2 = (
        forecasts.sort_values(["Ticker", "Horizon"])
        .loc[lambda df: df["Horizon"] == 2]
        .rename(columns={"Forecast Price": "Forecasted Price", "Forecast Return": "Forecasted Return"})
    )
    merged = day2.merge(actuals, on="Ticker", how="left")

    entry = pd.to_numeric(merged["Entry Price"], errors="coerce").fillna(merged["Last Price"])
    actual_close = pd.to_numeric(merged["Actual Day 2 Close"], errors="coerce")
    actual_return = actual_close / entry - 1
    forecast_direction = np.where(merged["Forecasted Return"] >= 0, "Up/Flat", "Down")
    actual_direction = np.where(actual_return >= 0, "Up/Flat", "Down")

    comparison = pd.DataFrame(
        {
            "Stock": merged["Ticker"],
            "Forecasted Price": merged["Forecasted Price"],
            "Forecasted Return": merged["Forecasted Return"],
            "Actual Day 2 Close": actual_close,
            "Actual Return": actual_return,
            "Forecast Direction": forecast_direction,
            "Actual Direction": actual_direction,
            "Direction Correct?": np.where(
                actual_close.notna(),
                forecast_direction == actual_direction,
                np.nan,
            ),
            "Error %": np.where(
                actual_close.notna() & (actual_close != 0),
                (merged["Forecasted Price"] - actual_close).abs() / actual_close * 100,
                np.nan,
            ),
            "Notes": merged.get("Notes", ""),
        }
    )
    comparison.to_csv(output_path, index=False)
    return comparison
