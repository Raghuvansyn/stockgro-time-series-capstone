"""Download and cache NSE historical data with yfinance."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    yf = None

from .config import (
    END_DATE,
    RAW_DATA_DIR,
    START_DATE,
    TICKERS,
    ensure_directories,
    ticker_to_filename,
)


REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def cache_path_for_ticker(ticker: str, raw_dir: Path = RAW_DATA_DIR) -> Path:
    """Return the CSV cache path for a ticker."""

    return raw_dir / f"{ticker_to_filename(ticker)}.csv"


def _download_end_exclusive(end_date: str) -> str:
    """yfinance end dates are exclusive, so ask for the following calendar day."""

    parsed = datetime.strptime(end_date, "%Y-%m-%d")
    return (parsed + timedelta(days=1)).strftime("%Y-%m-%d")


def _standardize_yfinance_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance output and retain expected OHLCV columns where present."""

    if data.empty:
        return data

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    data = data.reset_index()
    if "Date" not in data.columns and "Datetime" in data.columns:
        data = data.rename(columns={"Datetime": "Date"})

    keep_cols = ["Date"] + [col for col in REQUIRED_PRICE_COLUMNS if col in data.columns]
    data = data[keep_cols].copy()
    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
    data = data.sort_values("Date").drop_duplicates("Date")
    return data


def load_cached_ticker(ticker: str, raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame | None:
    """Load one cached ticker CSV if available."""

    path = cache_path_for_ticker(ticker, raw_dir)
    if not path.exists():
        return None
    try:
        data = pd.read_csv(path, parse_dates=["Date"])
    except Exception as exc:
        print(f"[WARN] Could not read cache for {ticker}: {exc}")
        return None
    return data


def download_ticker_data(
    ticker: str,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    raw_dir: Path = RAW_DATA_DIR,
    force: bool = False,
) -> pd.DataFrame | None:
    """Download one ticker, cache it, and return a standardized DataFrame."""

    ensure_directories()
    cached = load_cached_ticker(ticker, raw_dir)
    if cached is not None and not force:
        return cached

    if yf is None:
        print(
            f"[WARN] yfinance is not installed. Cannot download {ticker}. "
            "Install requirements.txt or place cached CSVs in data/raw/."
        )
        return cached

    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=_download_end_exclusive(end_date),
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        print(f"[WARN] yfinance download failed for {ticker}: {exc}")
        return cached

    data = _standardize_yfinance_frame(data)
    if data.empty:
        print(f"[WARN] No rows returned for {ticker}.")
        return cached

    path = cache_path_for_ticker(ticker, raw_dir)
    data.to_csv(path, index=False)
    return data


def fetch_universe_data(
    tickers: Iterable[str] | None = None,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    raw_dir: Path = RAW_DATA_DIR,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch all requested tickers and skip failures without stopping the run."""

    ensure_directories()
    tickers = list(tickers or TICKERS)
    data_by_ticker: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        frame = download_ticker_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            raw_dir=raw_dir,
            force=force,
        )
        if frame is None or frame.empty:
            print(f"[WARN] Skipping {ticker}; no usable data available.")
            continue
        data_by_ticker[ticker] = frame

    if not data_by_ticker:
        print("[WARN] No ticker data loaded. Check network access or cached CSV files.")
    return data_by_ticker


if __name__ == "__main__":
    fetched = fetch_universe_data()
    print(f"Loaded {len(fetched)} tickers: {', '.join(fetched)}")
