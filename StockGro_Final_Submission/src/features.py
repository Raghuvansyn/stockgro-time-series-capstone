"""Feature engineering and stock-selection metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    PROCESSED_DATA_DIR,
    STOCK_METADATA,
    TRADING_DAYS_PER_YEAR,
    ensure_directories,
)


def _trend_strength(log_price: pd.Series) -> tuple[float, float]:
    """Estimate trend slope and R-squared from a linear fit to log prices."""

    y = log_price.dropna()
    if len(y) < 30:
        return np.nan, np.nan
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y.to_numpy(), deg=1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y.to_numpy() - fitted) ** 2))
    ss_tot = float(np.sum((y.to_numpy() - y.mean()) ** 2))
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    annualized_slope = np.expm1(slope * TRADING_DAYS_PER_YEAR)
    return annualized_slope, r2


def compute_selection_metrics(
    cleaned: pd.DataFrame,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    processed_dir: Path = PROCESSED_DATA_DIR,
    save: bool = True,
) -> pd.DataFrame:
    """Compute data-driven metrics used to justify the stock universe."""

    ensure_directories()
    correlation = returns.corr()
    avg_abs_corr = correlation.abs().replace(1.0, np.nan).mean()
    records: list[dict[str, object]] = []

    for ticker in prices.columns:
        ticker_rows = cleaned.loc[cleaned["Ticker"] == ticker].copy()
        ret = returns[ticker].dropna()
        price = prices[ticker].dropna()
        log_price = np.log(price)
        annual_trend, trend_r2 = _trend_strength(log_price)

        volume = (
            pd.to_numeric(ticker_rows.get("Volume"), errors="coerce")
            if "Volume" in ticker_rows
            else pd.Series(dtype=float)
        )
        close = (
            pd.to_numeric(ticker_rows.get("Close"), errors="coerce")
            if "Close" in ticker_rows
            else pd.Series(dtype=float)
        )
        liquidity_proxy = (volume * close).replace([np.inf, -np.inf], np.nan).mean()
        latest_price = float(price.iloc[-1]) if len(price) else np.nan

        meta = STOCK_METADATA.get(ticker)
        annual_return = float(ret.mean() * TRADING_DAYS_PER_YEAR)
        annual_vol = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        sharpe_like = np.nan if annual_vol == 0 else annual_return / annual_vol

        records.append(
            {
                "Ticker": ticker,
                "Company": meta.company if meta else ticker,
                "Sector": meta.sector if meta else "Unknown",
                "Start Date": price.index.min(),
                "End Date": price.index.max(),
                "Observations": int(price.count()),
                "Latest Price": latest_price,
                "Avg Daily Log Return": float(ret.mean()),
                "Annualized Return": annual_return,
                "Annualized Volatility": annual_vol,
                "Rolling 20D Vol Latest": float(ret.rolling(20).std().iloc[-1] * np.sqrt(TRADING_DAYS_PER_YEAR)),
                "Sharpe Like": sharpe_like,
                "Liquidity Proxy": float(liquidity_proxy) if pd.notna(liquidity_proxy) else np.nan,
                "Avg Abs Correlation": float(avg_abs_corr.get(ticker, np.nan)),
                "Annualized Trend Strength": annual_trend,
                "Trend R2": trend_r2,
            }
        )

    metrics = pd.DataFrame(records)
    metrics["Selection Score"] = (
        metrics["Sharpe Like"].rank(pct=True)
        + metrics["Liquidity Proxy"].rank(pct=True)
        + metrics["Annualized Trend Strength"].rank(pct=True)
        - metrics["Annualized Volatility"].rank(pct=True) * 0.5
        - metrics["Avg Abs Correlation"].rank(pct=True) * 0.25
    )
    metrics = metrics.sort_values("Selection Score", ascending=False)

    if save:
        processed_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(processed_dir / "selection_metrics.csv", index=False)
        correlation.to_csv(processed_dir / "return_correlation.csv")

    return metrics


def compute_risk_metrics(
    returns: pd.DataFrame,
    processed_dir: Path = PROCESSED_DATA_DIR,
    save: bool = True,
) -> pd.DataFrame:
    """Compute volatility and simple risk-adjusted return metrics."""

    records: list[dict[str, object]] = []
    for ticker in returns.columns:
        ret = returns[ticker].dropna()
        annual_return = ret.mean() * TRADING_DAYS_PER_YEAR
        annual_vol = ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        records.append(
            {
                "Ticker": ticker,
                "Mean Daily Log Return": ret.mean(),
                "Annualized Return": annual_return,
                "Annualized Volatility": annual_vol,
                "Latest 20D Annualized Vol": ret.rolling(20).std().iloc[-1]
                * np.sqrt(TRADING_DAYS_PER_YEAR),
                "Sharpe Like": np.nan if annual_vol == 0 else annual_return / annual_vol,
                "Skewness": ret.skew(),
                "Kurtosis": ret.kurtosis(),
            }
        )

    risk = pd.DataFrame(records)
    if save:
        processed_dir.mkdir(parents=True, exist_ok=True)
        risk.to_csv(processed_dir / "risk_metrics.csv", index=False)
    return risk


def try_garch_volatility(
    returns: pd.DataFrame,
    processed_dir: Path = PROCESSED_DATA_DIR,
    save: bool = True,
) -> pd.DataFrame:
    """Estimate GARCH(1,1) volatility when the optional arch package is available."""

    try:
        from arch import arch_model
    except ImportError:
        result = pd.DataFrame(
            {
                "Ticker": list(returns.columns),
                "GARCH Available": False,
                "Latest Conditional Volatility": np.nan,
                "Note": "arch package unavailable; skipped gracefully.",
            }
        )
        if save:
            result.to_csv(processed_dir / "garch_volatility.csv", index=False)
        return result

    records: list[dict[str, object]] = []
    for ticker in returns.columns:
        ret = returns[ticker].dropna() * 100
        try:
            model = arch_model(ret, vol="GARCH", p=1, q=1, mean="Constant", rescale=False)
            fitted = model.fit(disp="off")
            latest_cond_vol = fitted.conditional_volatility.iloc[-1] / 100 * np.sqrt(TRADING_DAYS_PER_YEAR)
            records.append(
                {
                    "Ticker": ticker,
                    "GARCH Available": True,
                    "Latest Conditional Volatility": latest_cond_vol,
                    "Note": "",
                }
            )
        except Exception as exc:
            records.append(
                {
                    "Ticker": ticker,
                    "GARCH Available": True,
                    "Latest Conditional Volatility": np.nan,
                    "Note": f"fit failed: {exc}",
                }
            )

    result = pd.DataFrame(records)
    if save:
        result.to_csv(processed_dir / "garch_volatility.csv", index=False)
    return result
