"""Forecasting models for validation and 5-day forward prediction."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:  # pragma: no cover - dependency fallback
    ARIMA = None

from .config import (
    FORECASTS_DIR,
    FORECAST_HORIZON,
    TRAIN_END_EXCLUSIVE,
    VALIDATION_END,
    VALIDATION_START,
    ensure_directories,
)
from .evaluation import evaluate_forecast


ForecastFunction = Callable[[pd.Series, pd.Series], pd.Series]


def split_train_validation(
    series: pd.Series,
    train_end_exclusive: str = TRAIN_END_EXCLUSIVE,
    validation_start: str = VALIDATION_START,
    validation_end: str = VALIDATION_END,
) -> tuple[pd.Series, pd.Series]:
    """Split a price series into train and validation according to the brief."""

    series = series.dropna().sort_index()
    train = series.loc[series.index < pd.Timestamp(train_end_exclusive)]
    validation = series.loc[
        (series.index >= pd.Timestamp(validation_start))
        & (series.index <= pd.Timestamp(validation_end))
    ]
    return train, validation


def next_business_days(last_date: pd.Timestamp, horizon: int = FORECAST_HORIZON) -> pd.DatetimeIndex:
    """Return the next business days. Exchange holidays are not modeled."""

    return pd.bdate_range(pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=horizon)


def naive_validation_forecast(train: pd.Series, validation: pd.Series) -> pd.Series:
    """One-step naive forecast: tomorrow equals the latest observed price."""

    history = train.copy()
    preds: list[float] = []
    for date, actual in validation.items():
        preds.append(float(history.iloc[-1]))
        history.loc[date] = actual
    return pd.Series(preds, index=validation.index, name="Naive")


def moving_average_validation_forecast(
    train: pd.Series,
    validation: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Walk-forward moving-average forecast."""

    history = train.copy()
    preds: list[float] = []
    for date, actual in validation.items():
        preds.append(float(history.tail(window).mean()))
        history.loc[date] = actual
    return pd.Series(preds, index=validation.index, name=f"MovingAverage{window}")


def exp_smoothing_validation_forecast(
    train: pd.Series,
    validation: pd.Series,
    alpha: float = 0.30,
) -> pd.Series:
    """Walk-forward exponentially weighted moving-average forecast."""

    history = train.copy()
    preds: list[float] = []
    for date, actual in validation.items():
        preds.append(float(history.ewm(alpha=alpha, adjust=False).mean().iloc[-1]))
        history.loc[date] = actual
    return pd.Series(preds, index=validation.index, name=f"ExpSmoothing{alpha:.2f}")


def _fit_arima(series: pd.Series, order: tuple[int, int, int]) -> object:
    """Fit ARIMA with warnings suppressed for cleaner batch output."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(
            series.astype(float),
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()


def select_arima_order(
    train: pd.Series,
    candidate_orders: tuple[tuple[int, int, int], ...] = (
        (0, 1, 0),
        (1, 1, 0),
        (0, 1, 1),
        (1, 1, 1),
        (2, 1, 0),
        (0, 1, 2),
        (2, 1, 1),
        (1, 1, 2),
    ),
) -> tuple[tuple[int, int, int] | None, float]:
    """Select a compact ARIMA order by AIC."""

    if ARIMA is None:
        return None, np.nan

    best_order: tuple[int, int, int] | None = None
    best_aic = np.inf
    for order in candidate_orders:
        try:
            fitted = _fit_arima(train, order)
            if fitted.aic < best_aic:
                best_order = order
                best_aic = float(fitted.aic)
        except Exception:
            continue
    return best_order, best_aic


def arima_validation_forecast(
    train: pd.Series,
    validation: pd.Series,
    order: tuple[int, int, int] | None = None,
) -> tuple[pd.Series | None, tuple[int, int, int] | None, str]:
    """Fit ARIMA on training data and forecast the validation horizon."""

    if ARIMA is None:
        return None, None, "statsmodels unavailable; ARIMA skipped."
    if len(train) < 80 or len(validation) == 0:
        return None, None, "insufficient data for ARIMA."

    selected_order = order or select_arima_order(train)[0]
    if selected_order is None:
        return None, None, "no ARIMA order converged."

    try:
        fitted = _fit_arima(train, selected_order)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = fitted.forecast(steps=len(validation))
        pred = pd.Series(pred.to_numpy(), index=validation.index, name=f"ARIMA{selected_order}")
        return pred, selected_order, ""
    except Exception as exc:
        return None, selected_order, f"ARIMA failed: {exc}"


def future_forecast_from_model(
    model_name: str,
    full_series: pd.Series,
    horizon: int = FORECAST_HORIZON,
    arima_order: tuple[int, int, int] | None = None,
) -> pd.Series:
    """Forecast the next horizon for one selected model."""

    forecast_index = next_business_days(full_series.index.max(), horizon=horizon)

    if model_name == "Naive":
        values = [float(full_series.iloc[-1])] * horizon
    elif model_name == "MovingAverage20":
        history = full_series.copy()
        values = []
        for date in forecast_index:
            prediction = float(history.tail(20).mean())
            values.append(prediction)
            history.loc[date] = prediction
    elif model_name == "ExpSmoothing0.30":
        history = full_series.copy()
        values = []
        for date in forecast_index:
            prediction = float(history.ewm(alpha=0.30, adjust=False).mean().iloc[-1])
            values.append(prediction)
            history.loc[date] = prediction
    elif model_name.startswith("ARIMA") and ARIMA is not None:
        order = arima_order or (1, 1, 1)
        fitted = _fit_arima(full_series, order)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = fitted.forecast(steps=horizon).to_numpy()
    else:
        values = [float(full_series.iloc[-1])] * horizon

    return pd.Series(values, index=forecast_index, name=model_name)


def _choose_best_model(evaluation: pd.DataFrame) -> pd.Series:
    """Pick lowest error model, with directional accuracy as a tie-breaker."""

    available = evaluation.dropna(subset=["Selection Score"]).copy()
    if available.empty:
        available = evaluation.dropna(subset=["MAPE"]).copy()
    if available.empty:
        return evaluation.iloc[0]

    return available.sort_values(
        ["Selection Score", "MAPE", "RMSE", "Directional Accuracy"],
        ascending=[True, True, True, False],
    ).iloc[0]


def run_forecasting_pipeline(
    prices: pd.DataFrame,
    forecasts_dir: Path = FORECASTS_DIR,
    horizon: int = FORECAST_HORIZON,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train, validate, choose models, and forecast 5 future business days."""

    ensure_directories()
    forecasts_dir.mkdir(parents=True, exist_ok=True)

    evaluation_records: list[dict[str, object]] = []
    validation_frames: list[pd.DataFrame] = []
    future_frames: list[pd.DataFrame] = []
    best_records: list[dict[str, object]] = []

    for ticker in prices.columns:
        series = prices[ticker].dropna().sort_index()
        train, validation = split_train_validation(series)
        if train.empty or validation.empty:
            print(f"[WARN] {ticker}: missing train or validation split; skipped.")
            continue

        model_predictions: dict[str, pd.Series] = {
            "Naive": naive_validation_forecast(train, validation),
            "MovingAverage20": moving_average_validation_forecast(train, validation, window=20),
            "ExpSmoothing0.30": exp_smoothing_validation_forecast(train, validation, alpha=0.30),
        }

        arima_pred, arima_order, arima_note = arima_validation_forecast(train, validation)
        if arima_pred is not None:
            model_predictions[f"ARIMA{arima_order}"] = arima_pred
        else:
            evaluation_records.append(
                {
                    "Ticker": ticker,
                    "Model": "ARIMA",
                    "RMSE": np.nan,
                    "MAPE": np.nan,
                    "Directional Accuracy": np.nan,
                    "Selection Score": np.nan,
                    "Note": arima_note,
                }
            )

        for model_name, prediction in model_predictions.items():
            record = evaluate_forecast(
                ticker=ticker,
                model_name=model_name,
                actual=validation,
                predicted=prediction,
                train_last_price=float(train.iloc[-1]),
            )
            record["Note"] = ""
            evaluation_records.append(record)
            validation_frames.append(
                pd.DataFrame(
                    {
                        "Date": validation.index,
                        "Ticker": ticker,
                        "Model": model_name,
                        "Actual": validation.to_numpy(),
                        "Predicted": prediction.reindex(validation.index).to_numpy(),
                    }
                )
            )

        eval_for_ticker = pd.DataFrame([r for r in evaluation_records if r["Ticker"] == ticker])
        best = _choose_best_model(eval_for_ticker)
        best_model = str(best["Model"])
        selected_order = arima_order if best_model.startswith("ARIMA") else None
        future = future_forecast_from_model(
            best_model,
            full_series=series,
            horizon=horizon,
            arima_order=selected_order,
        )

        last_price = float(series.iloc[-1])
        future_returns = future / last_price - 1
        ticker_future = pd.DataFrame(
            {
                "Forecast Date": future.index,
                "Ticker": ticker,
                "Best Model": best_model,
                "Horizon": range(1, horizon + 1),
                "Last Price": last_price,
                "Forecast Price": future.to_numpy(),
                "Forecast Return": future_returns.to_numpy(),
            }
        )
        future_frames.append(ticker_future)
        best_records.append(
            {
                "Ticker": ticker,
                "Best Model": best_model,
                "Best Validation MAPE": best.get("MAPE", np.nan),
                "Best Validation RMSE": best.get("RMSE", np.nan),
                "Best Directional Accuracy": best.get("Directional Accuracy", np.nan),
                "ARIMA Order": selected_order,
                "Last Price": last_price,
                "Forecast Day 5 Price": float(future.iloc[-1]),
                "Forecast Return 5D": float(future_returns.iloc[-1]),
            }
        )

    evaluation = pd.DataFrame(evaluation_records)
    validation_predictions = (
        pd.concat(validation_frames, ignore_index=True) if validation_frames else pd.DataFrame()
    )
    five_day_forecasts = pd.concat(future_frames, ignore_index=True) if future_frames else pd.DataFrame()
    best_summary = pd.DataFrame(best_records)

    if save:
        evaluation.to_csv(forecasts_dir / "model_evaluation.csv", index=False)
        validation_predictions.to_csv(forecasts_dir / "validation_predictions.csv", index=False)
        five_day_forecasts.to_csv(forecasts_dir / "all_5day_forecasts.csv", index=False)
        best_summary.to_csv(forecasts_dir / "best_model_summary.csv", index=False)

        for ticker in validation_predictions["Ticker"].unique() if not validation_predictions.empty else []:
            validation_predictions.loc[validation_predictions["Ticker"] == ticker].to_csv(
                forecasts_dir / f"{ticker.replace('.', '_')}_validation_predictions.csv",
                index=False,
            )
        for ticker in five_day_forecasts["Ticker"].unique() if not five_day_forecasts.empty else []:
            five_day_forecasts.loc[five_day_forecasts["Ticker"] == ticker].to_csv(
                forecasts_dir / f"{ticker.replace('.', '_')}_5day_forecast.csv",
                index=False,
            )

    return evaluation, validation_predictions, five_day_forecasts
