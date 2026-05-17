"""Visualization utilities that save capstone figures to outputs/figures."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_cache_root = Path(tempfile.gettempdir()) / "stockgro_time_series_cache"
_mpl_cache = _cache_root / "matplotlib"
_xdg_cache = _cache_root / "xdg"
_mpl_cache.mkdir(parents=True, exist_ok=True)
_xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_cache))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from statsmodels.tsa.seasonal import STL
except ImportError:  # pragma: no cover - dependency fallback
    STL = None

from .config import FIGURES_DIR, ensure_directories, ticker_to_filename


sns.set_theme(style="whitegrid", context="notebook")


def _save(fig: plt.Figure, path: Path) -> Path:
    """Save and close a matplotlib figure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_price_trends(prices: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> Path:
    """Plot normalized price trends for the full universe."""

    normalized = prices / prices.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(12, 7))
    normalized.plot(ax=ax, linewidth=1.5)
    ax.set_title("Normalized Price Trends (Base = 100)")
    ax.set_ylabel("Indexed Price")
    ax.set_xlabel("")
    ax.legend(ncol=3, fontsize=8)
    return _save(fig, figures_dir / "price_trends_normalized.png")


def plot_daily_returns(returns: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> Path:
    """Plot daily log returns for all tickers."""

    fig, ax = plt.subplots(figsize=(12, 7))
    returns.plot(ax=ax, alpha=0.65, linewidth=0.8)
    ax.set_title("Daily Log Returns")
    ax.set_ylabel("Log Return")
    ax.set_xlabel("")
    ax.legend(ncol=3, fontsize=8)
    return _save(fig, figures_dir / "daily_log_returns.png")


def plot_rolling_stats(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    ticker: str,
    figures_dir: Path = FIGURES_DIR,
    window: int = 20,
) -> Path:
    """Plot rolling mean and volatility for one selected stock."""

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    price = prices[ticker].dropna()
    price.plot(ax=axes[0], color="#1f77b4", label="Price")
    price.rolling(window).mean().plot(ax=axes[0], color="#ff7f0e", label=f"{window}D Mean")
    axes[0].set_title(f"{ticker}: Price and Rolling Mean")
    axes[0].legend()

    vol = returns[ticker].rolling(window).std() * np.sqrt(252)
    vol.plot(ax=axes[1], color="#d62728", label=f"{window}D Annualized Volatility")
    axes[1].set_title(f"{ticker}: Rolling Volatility")
    axes[1].set_ylabel("Annualized Vol")
    axes[1].legend()
    return _save(fig, figures_dir / f"{ticker_to_filename(ticker)}_rolling_stats.png")


def plot_correlation_heatmap(returns: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> Path:
    """Plot a return correlation heatmap."""

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        returns.corr(),
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap of Daily Log Returns")
    return _save(fig, figures_dir / "return_correlation_heatmap.png")


def plot_stl_decomposition(
    prices: pd.DataFrame,
    ticker: str,
    figures_dir: Path = FIGURES_DIR,
    period: int = 252,
) -> Path | None:
    """Run STL decomposition for one ticker when statsmodels is available."""

    if STL is None:
        print("[WARN] statsmodels unavailable; STL plot skipped.")
        return None
    series = prices[ticker].dropna()
    if len(series) < period * 2:
        print(f"[WARN] {ticker}: not enough observations for STL period {period}.")
        return None

    result = STL(series, period=period, robust=True).fit()
    fig = result.plot()
    fig.set_size_inches(11, 8)
    fig.suptitle(f"{ticker}: STL Decomposition", y=1.02)
    return _save(fig, figures_dir / f"{ticker_to_filename(ticker)}_stl_decomposition.png")


def plot_validation_forecasts(
    validation_predictions: pd.DataFrame,
    ticker: str,
    model: str | None = None,
    figures_dir: Path = FIGURES_DIR,
) -> Path | None:
    """Plot predicted vs actual validation prices for one ticker/model."""

    data = validation_predictions.loc[validation_predictions["Ticker"] == ticker].copy()
    if data.empty:
        return None
    if model is not None:
        data = data.loc[data["Model"] == model]
    if data.empty:
        return None
    data["Date"] = pd.to_datetime(data["Date"])

    fig, ax = plt.subplots(figsize=(12, 6))
    actual = data.drop_duplicates("Date").sort_values("Date")
    ax.plot(actual["Date"], actual["Actual"], label="Actual", color="#111111", linewidth=2)
    for model_name, group in data.groupby("Model"):
        group = group.sort_values("Date")
        ax.plot(group["Date"], group["Predicted"], label=model_name, linewidth=1.4, alpha=0.85)
    ax.set_title(f"{ticker}: Validation Predicted vs Actual")
    ax.set_ylabel("Price")
    ax.set_xlabel("")
    ax.legend(fontsize=8)
    return _save(fig, figures_dir / f"{ticker_to_filename(ticker)}_validation_forecast.png")


def plot_portfolio_allocation(
    final_portfolio: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
) -> Path | None:
    """Plot final portfolio weights."""

    if final_portfolio.empty:
        return None
    data = final_portfolio.sort_values("Weight", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(data["Ticker"], data["Weight"], color="#2a9d8f")
    ax.set_title(f"Final Portfolio Allocation: {data['Strategy'].iloc[0]}")
    ax.set_xlabel("Weight")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    return _save(fig, figures_dir / "final_portfolio_allocation.png")


def generate_core_figures(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    validation_predictions: pd.DataFrame | None = None,
    final_portfolio: pd.DataFrame | None = None,
    figures_dir: Path = FIGURES_DIR,
) -> list[Path]:
    """Generate the project figures used by the notebook/report."""

    ensure_directories()
    paths = [
        plot_price_trends(prices, figures_dir),
        plot_daily_returns(returns, figures_dir),
        plot_correlation_heatmap(returns, figures_dir),
    ]
    if len(prices.columns) > 0:
        representative = prices.columns[0]
        rolling = plot_rolling_stats(prices, returns, representative, figures_dir)
        paths.append(rolling)
        stl = plot_stl_decomposition(prices, representative, figures_dir)
        if stl is not None:
            paths.append(stl)

    if validation_predictions is not None and not validation_predictions.empty:
        first_ticker = validation_predictions["Ticker"].iloc[0]
        forecast_plot = plot_validation_forecasts(validation_predictions, first_ticker, figures_dir=figures_dir)
        if forecast_plot is not None:
            paths.append(forecast_plot)

    if final_portfolio is not None and not final_portfolio.empty:
        allocation = plot_portfolio_allocation(final_portfolio, figures_dir)
        if allocation is not None:
            paths.append(allocation)

    return paths
