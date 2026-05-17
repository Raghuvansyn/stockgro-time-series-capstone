"""Clean raw market data and prepare stationary return series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller
except ImportError:  # pragma: no cover - dependency fallback
    adfuller = None

from .config import PROCESSED_DATA_DIR, STOCK_METADATA, ensure_directories


def choose_price_column(frame: pd.DataFrame) -> str:
    """Prefer adjusted close when it exists and has non-null observations."""

    if "Adj Close" in frame.columns and frame["Adj Close"].notna().sum() > 0:
        return "Adj Close"
    if "Close" in frame.columns and frame["Close"].notna().sum() > 0:
        return "Close"
    raise ValueError("Frame must contain either 'Adj Close' or 'Close'.")


def clean_ticker_frame(ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Clean one raw yfinance frame and add modeling columns."""

    data = frame.copy()
    if "Date" not in data.columns:
        raise ValueError(f"{ticker}: expected a Date column.")

    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
    data = data.sort_values("Date").drop_duplicates("Date")
    data = data.set_index("Date")

    numeric_cols = [col for col in data.columns if col != "Ticker"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.ffill().bfill()
    price_col = choose_price_column(data)
    data["Price"] = data[price_col]
    data["LogPrice"] = np.log(data["Price"])
    data["LogReturn"] = data["LogPrice"].diff()
    data["SimpleReturn"] = data["Price"].pct_change()
    data["Ticker"] = ticker

    meta = STOCK_METADATA.get(ticker)
    data["Company"] = meta.company if meta else ticker
    data["Sector"] = meta.sector if meta else "Unknown"

    return data.reset_index()


def preprocess_universe(
    raw_data: dict[str, pd.DataFrame],
    processed_dir: Path = PROCESSED_DATA_DIR,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean all tickers and return long data, price matrix, and return matrix."""

    ensure_directories()
    cleaned_frames: list[pd.DataFrame] = []
    for ticker, frame in raw_data.items():
        try:
            cleaned_frames.append(clean_ticker_frame(ticker, frame))
        except Exception as exc:
            print(f"[WARN] Preprocessing skipped for {ticker}: {exc}")

    if not cleaned_frames:
        raise ValueError("No cleaned ticker data available.")

    cleaned = pd.concat(cleaned_frames, ignore_index=True)
    prices = cleaned.pivot(index="Date", columns="Ticker", values="Price").sort_index()
    returns = cleaned.pivot(index="Date", columns="Ticker", values="LogReturn").sort_index()

    prices = prices.ffill().bfill()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    if save:
        processed_dir.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(processed_dir / "cleaned_long.csv", index=False)
        prices.to_csv(processed_dir / "price_matrix.csv")
        returns.to_csv(processed_dir / "log_return_matrix.csv")

    return cleaned, prices, returns


def run_adf_tests(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    processed_dir: Path = PROCESSED_DATA_DIR,
    save: bool = True,
) -> pd.DataFrame:
    """Run ADF stationarity tests on prices, first differences, and returns."""

    records: list[dict[str, object]] = []
    if adfuller is None:
        for ticker in prices.columns:
            records.append(
                {
                    "Ticker": ticker,
                    "Series": "Price",
                    "ADF Statistic": np.nan,
                    "p-value": np.nan,
                    "Stationary at 5%": "statsmodels unavailable",
                }
            )
        result = pd.DataFrame(records)
        if save:
            result.to_csv(processed_dir / "adf_tests.csv", index=False)
        return result

    series_map: dict[str, pd.DataFrame] = {
        "Price": prices,
        "Price Difference": prices.diff(),
        "Log Return": returns,
    }
    for series_name, matrix in series_map.items():
        for ticker in matrix.columns:
            values = matrix[ticker].dropna()
            if len(values) < 30:
                continue
            try:
                stat, p_value, used_lag, n_obs, critical_values, _ = adfuller(
                    values,
                    autolag="AIC",
                )
                records.append(
                    {
                        "Ticker": ticker,
                        "Series": series_name,
                        "ADF Statistic": stat,
                        "p-value": p_value,
                        "Used Lag": used_lag,
                        "Observations": n_obs,
                        "5% Critical Value": critical_values.get("5%"),
                        "Stationary at 5%": bool(p_value < 0.05),
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "Ticker": ticker,
                        "Series": series_name,
                        "ADF Statistic": np.nan,
                        "p-value": np.nan,
                        "Stationary at 5%": f"failed: {exc}",
                    }
                )

    result = pd.DataFrame(records)
    if save:
        processed_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(processed_dir / "adf_tests.csv", index=False)
    return result


def load_processed_data(
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cleaned long data, price matrix, and log return matrix from disk."""

    cleaned = pd.read_csv(processed_dir / "cleaned_long.csv", parse_dates=["Date"])
    prices = pd.read_csv(processed_dir / "price_matrix.csv", parse_dates=["Date"], index_col="Date")
    returns = pd.read_csv(
        processed_dir / "log_return_matrix.csv",
        parse_dates=["Date"],
        index_col="Date",
    )
    return cleaned, prices, returns
