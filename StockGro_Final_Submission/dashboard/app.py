"""Streamlit dashboard for the StockGro capstone outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FORECAST_DIR = PROJECT_ROOT / "outputs" / "forecasts"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Load a CSV if present; otherwise return an empty DataFrame."""

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def main() -> None:
    """Render the dashboard."""

    import streamlit as st

    st.set_page_config(page_title="StockGro Capstone Dashboard", layout="wide")
    st.title("StockGro Time Series Capstone")
    st.caption("Educational virtual-trading project. Not investment advice.")

    prices = load_csv(PROCESSED_DIR / "price_matrix.csv", parse_dates=["Date"])
    returns = load_csv(PROCESSED_DIR / "log_return_matrix.csv", parse_dates=["Date"])
    forecasts = load_csv(FORECAST_DIR / "all_5day_forecasts.csv", parse_dates=["Forecast Date"])
    validation = load_csv(FORECAST_DIR / "validation_predictions.csv", parse_dates=["Date"])
    evaluation = load_csv(FORECAST_DIR / "model_evaluation.csv")
    portfolio = load_csv(TABLE_DIR / "final_recommended_portfolio.csv")

    if prices.empty:
        st.warning("No processed outputs found. Run `python run_pipeline.py` from the project root first.")
        return

    ticker_cols = [col for col in prices.columns if col != "Date"]
    selected_ticker = st.sidebar.selectbox("Ticker", ticker_cols)

    st.subheader("Stock Price Trend")
    price_view = prices[["Date", selected_ticker]].set_index("Date")
    st.line_chart(price_view)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Validation Forecast")
        if not validation.empty:
            view = validation.loc[validation["Ticker"] == selected_ticker]
            if not view.empty:
                chart_data = view.pivot(index="Date", columns="Model", values="Predicted")
                chart_data["Actual"] = view.drop_duplicates("Date").set_index("Date")["Actual"]
                st.line_chart(chart_data)
            else:
                st.info("No validation forecast for selected ticker.")
        else:
            st.info("Validation forecasts not available yet.")

    with col2:
        st.subheader("5-Day Forecast")
        if not forecasts.empty:
            st.dataframe(
                forecasts.loc[forecasts["Ticker"] == selected_ticker][
                    ["Forecast Date", "Best Model", "Horizon", "Forecast Price", "Forecast Return"]
                ],
                use_container_width=True,
            )
        else:
            st.info("5-day forecasts not available yet.")

    st.subheader("Return Correlation")
    if not returns.empty:
        corr = returns.drop(columns=["Date"], errors="ignore").corr()
        st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)

    st.subheader("Model Evaluation")
    if not evaluation.empty:
        st.dataframe(evaluation, use_container_width=True)

    st.subheader("Final Recommended Portfolio")
    if not portfolio.empty:
        chart = portfolio.set_index("Ticker")["Weight"]
        st.bar_chart(chart)
        st.dataframe(portfolio, use_container_width=True)
    else:
        st.info("Portfolio output not available yet.")


if __name__ == "__main__":
    main()
