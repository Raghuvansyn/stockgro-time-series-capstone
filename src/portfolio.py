"""Portfolio construction strategies for the virtual StockGro challenge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    INITIAL_CAPITAL,
    STOCK_METADATA,
    TABLES_DIR,
    TRADING_DAYS_PER_YEAR,
    ensure_directories,
)


def constrained_weights(
    scores: pd.Series,
    min_weight: float = 0.05,
    max_weight: float = 0.25,
) -> pd.Series:
    """Normalize scores into long-only weights under min/max constraints."""

    scores = pd.to_numeric(scores, errors="coerce").fillna(0).clip(lower=0)
    if scores.empty:
        return scores

    n = len(scores)
    min_weight = min(min_weight, 1 / n)
    max_weight = max(max_weight, 1 / n)

    if scores.sum() <= 0:
        weights = pd.Series(1 / n, index=scores.index)
    else:
        weights = scores / scores.sum()

    fixed: dict[str, float] = {}
    remaining = list(weights.index)
    remaining_total = 1.0

    for _ in range(20):
        if not remaining:
            break
        remaining_scores = scores.loc[remaining]
        if remaining_scores.sum() <= 0:
            proposed = pd.Series(remaining_total / len(remaining), index=remaining)
        else:
            proposed = remaining_scores / remaining_scores.sum() * remaining_total

        low = proposed[proposed < min_weight]
        high = proposed[proposed > max_weight]
        if low.empty and high.empty:
            for ticker, value in proposed.items():
                fixed[ticker] = float(value)
            break

        to_fix = pd.concat([low, high])
        for ticker, value in to_fix.items():
            fixed[ticker] = min_weight if value < min_weight else max_weight

        remaining = [ticker for ticker in remaining if ticker not in fixed]
        remaining_total = 1.0 - sum(fixed.values())
        if remaining_total <= 0:
            break

    weights = pd.Series(fixed, dtype=float).reindex(scores.index).fillna(0)
    if weights.sum() == 0:
        weights = pd.Series(1 / n, index=scores.index)
    else:
        weights = weights / weights.sum()
    return weights


def prepare_portfolio_inputs(
    best_model_summary: pd.DataFrame,
    selection_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Merge forecast, risk, sector, and correlation metrics."""

    keep_cols = [
        "Ticker",
        "Company",
        "Sector",
        "Annualized Volatility",
        "Sharpe Like",
        "Avg Abs Correlation",
        "Liquidity Proxy",
        "Annualized Trend Strength",
    ]
    metrics = selection_metrics[keep_cols].copy()
    combined = best_model_summary.merge(metrics, on="Ticker", how="left")
    combined["Company"] = combined.apply(
        lambda row: row["Company"]
        if pd.notna(row.get("Company"))
        else STOCK_METADATA.get(row["Ticker"], row["Ticker"]).company,
        axis=1,
    )
    combined["Sector"] = combined.apply(
        lambda row: row["Sector"]
        if pd.notna(row.get("Sector"))
        else STOCK_METADATA.get(row["Ticker"], row["Ticker"]).sector,
        axis=1,
    )
    combined["Risk Adjusted Forecast"] = combined["Forecast Return 5D"] / combined[
        "Annualized Volatility"
    ].replace(0, np.nan)
    return combined


def select_portfolio_candidates(
    metrics: pd.DataFrame,
    min_holdings: int = 6,
    max_holdings: int = 10,
) -> pd.DataFrame:
    """Select a diversified basket with preference for positive 5-day forecasts."""

    ranked = metrics.copy()
    ranked["Positive Forecast"] = ranked["Forecast Return 5D"] > 0
    ranked["Candidate Score"] = (
        ranked["Forecast Return 5D"].fillna(0).clip(lower=0).rank(pct=True)
        + ranked["Risk Adjusted Forecast"].fillna(0).rank(pct=True)
        + ranked["Sharpe Like"].fillna(0).rank(pct=True) * 0.5
        - ranked["Avg Abs Correlation"].fillna(ranked["Avg Abs Correlation"].median()).rank(pct=True)
        * 0.25
    )
    ranked = ranked.sort_values(
        ["Positive Forecast", "Candidate Score", "Forecast Return 5D"],
        ascending=[False, False, False],
    )

    selected_rows = []
    seen_sectors: set[str] = set()
    for _, row in ranked.iterrows():
        if len(selected_rows) >= max_holdings:
            break
        if row["Sector"] not in seen_sectors or len(selected_rows) < min_holdings:
            selected_rows.append(row)
            seen_sectors.add(str(row["Sector"]))

    if len(selected_rows) < min_holdings:
        already = {row["Ticker"] for row in selected_rows}
        for _, row in ranked.iterrows():
            if row["Ticker"] not in already:
                selected_rows.append(row)
                already.add(row["Ticker"])
            if len(selected_rows) >= min_holdings:
                break

    selected = pd.DataFrame(selected_rows).drop_duplicates("Ticker")
    return selected.head(max_holdings)


def _portfolio_table(
    strategy_name: str,
    selected: pd.DataFrame,
    weights: pd.Series,
    capital: float,
) -> pd.DataFrame:
    """Build a holdings table with target allocation and integer quantities."""

    table = selected.set_index("Ticker").copy()
    table["Strategy"] = strategy_name
    table["Weight"] = weights.reindex(table.index).fillna(0)
    table["Target Rupee Allocation"] = table["Weight"] * capital
    table["Estimated Quantity"] = np.floor(
        table["Target Rupee Allocation"] / table["Last Price"].replace(0, np.nan)
    ).fillna(0).astype(int)
    table["Estimated Cost"] = table["Estimated Quantity"] * table["Last Price"]
    table["Cash After Rounding"] = capital - table["Estimated Cost"].sum()
    table["Rationale"] = table.apply(
        lambda row: (
            f"{row['Sector']} exposure; {row['Best Model']} forecast "
            f"{row['Forecast Return 5D']:.2%}; volatility {row['Annualized Volatility']:.2%}."
        ),
        axis=1,
    )
    ordered = [
        "Strategy",
        "Company",
        "Sector",
        "Weight",
        "Target Rupee Allocation",
        "Last Price",
        "Estimated Quantity",
        "Estimated Cost",
        "Forecast Return 5D",
        "Annualized Volatility",
        "Avg Abs Correlation",
        "Best Model",
        "Rationale",
        "Cash After Rounding",
    ]
    return table.reset_index()[["Ticker"] + ordered]


def build_allocation_strategies(
    portfolio_inputs: pd.DataFrame,
    returns: pd.DataFrame,
    capital: float = INITIAL_CAPITAL,
    min_weight: float = 0.05,
    max_weight: float = 0.25,
    save_dir: Path = TABLES_DIR,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create forecast, volatility, and correlation-aware portfolios."""

    ensure_directories()
    selected = select_portfolio_candidates(portfolio_inputs)
    selected = selected.set_index("Ticker")

    positive_return = selected["Forecast Return 5D"].clip(lower=0)
    forecast_scores = positive_return + positive_return[positive_return > 0].median(skipna=True) * 0.10
    forecast_weights = constrained_weights(forecast_scores.fillna(0), min_weight, max_weight)

    inv_vol_scores = 1 / selected["Annualized Volatility"].replace(0, np.nan)
    inv_vol_scores = inv_vol_scores.where(selected["Forecast Return 5D"] >= 0, inv_vol_scores * 0.35)
    volatility_weights = constrained_weights(inv_vol_scores, min_weight, max_weight)

    corr_penalty = 1 - selected["Avg Abs Correlation"].fillna(selected["Avg Abs Correlation"].median())
    corr_scores = selected["Forecast Return 5D"].clip(lower=0) * corr_penalty
    corr_scores = corr_scores / selected["Annualized Volatility"].replace(0, np.nan)
    correlation_weights = constrained_weights(corr_scores.fillna(0), min_weight, max_weight)

    selected_reset = selected.reset_index()
    portfolios = pd.concat(
        [
            _portfolio_table("A: Forecast-Guided", selected_reset, forecast_weights, capital),
            _portfolio_table("B: Volatility-Aware", selected_reset, volatility_weights, capital),
            _portfolio_table("C: Correlation-Diversified", selected_reset, correlation_weights, capital),
        ],
        ignore_index=True,
    )

    strategy_summary = summarize_strategies(portfolios, returns)
    final_strategy = strategy_summary.sort_values(
        ["Risk Adjusted Expected Return", "Expected 5D Return"],
        ascending=False,
    ).iloc[0]["Strategy"]
    final_portfolio = portfolios.loc[portfolios["Strategy"] == final_strategy].copy()

    if save:
        save_dir.mkdir(parents=True, exist_ok=True)
        portfolios.to_csv(save_dir / "portfolio_strategy_comparison.csv", index=False)
        strategy_summary.to_csv(save_dir / "strategy_summary.csv", index=False)
        final_portfolio.to_csv(save_dir / "final_recommended_portfolio.csv", index=False)

    return portfolios, strategy_summary, final_portfolio


def summarize_strategies(portfolios: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Summarize expected return, volatility, diversification, and cash use."""

    records: list[dict[str, object]] = []
    cov = returns.cov() * TRADING_DAYS_PER_YEAR
    for strategy, group in portfolios.groupby("Strategy"):
        weights = group.set_index("Ticker")["Weight"]
        common = [ticker for ticker in weights.index if ticker in cov.index]
        if common:
            w = weights.loc[common].to_numpy()
            cov_sub = cov.loc[common, common].to_numpy()
            portfolio_vol = float(np.sqrt(w.T @ cov_sub @ w))
        else:
            portfolio_vol = np.nan
        expected_5d_return = float((group["Weight"] * group["Forecast Return 5D"]).sum())
        cash_after_rounding = float(group["Cash After Rounding"].iloc[0])
        records.append(
            {
                "Strategy": strategy,
                "Holdings": group["Ticker"].nunique(),
                "Expected 5D Return": expected_5d_return,
                "Annualized Portfolio Volatility": portfolio_vol,
                "Risk Adjusted Expected Return": np.nan
                if portfolio_vol == 0 or pd.isna(portfolio_vol)
                else expected_5d_return / portfolio_vol,
                "Average Abs Correlation": group["Avg Abs Correlation"].mean(),
                "Cash After Rounding": cash_after_rounding,
            }
        )
    return pd.DataFrame(records)
