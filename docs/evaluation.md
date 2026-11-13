# Evaluation status and reproducibility

This note describes the current implementation. The portfolio outputs are educational calculations, not evidence of live trading returns.

## Current protocol

`src/forecasting.py:split_train_validation` splits each ticker by date. Training observations precede July 1, 2025; validation covers July 1 through December 31, 2025. The validation results also select the final model. There is no independent held-out test in the modular pipeline.

| Model | Information available for each validation prediction |
|---|---|
| Naive | Previous actual price, including prior validation observations |
| MovingAverage20 | Mean of the previous 20 observed prices, updated after each validation observation |
| ExpSmoothing0.30 | Exponentially weighted observed history, updated after each validation observation |
| ARIMA | Training history only; fit once and forecast the entire validation period |

These represent different forecasting tasks. A ranking across them confounds model quality with the amount of recent information available. Align forecast origins, update rules, and horizons before claiming that one model outperforms another. The five-business-day future outputs have no measured five-day holdout performance here.

`src/evaluation.py:evaluate_forecast` reports RMSE in price units, MAPE as a percentage, and directional accuracy as a fraction. Directional accuracy compares the predicted change against the previous actual price. A Naive prediction is a zero change, so it counts as correct only when the observed price is unchanged. The implementation selects using `MAPE * (1 - 0.75 * directional_accuracy)` before other sort keys; directional accuracy is therefore part of the primary score, rather than only a tie-breaker. This custom score has not been externally validated.

## Reproduced baseline calculations

On November 14, 2026, the following calculations were reproduced from the committed `data/raw/DRREDDY_NS.csv` cache using Python 3.13.0, pandas 2.2.3, and NumPy 2.2.6. Both matched their rows in `outputs/forecasts/model_evaluation.csv`.

- Training: January 1, 2021 through June 30, 2025, **1,110 observations**.
- Validation: July 1 through December 31, 2025, **126 observations**.
- The inspected raw ticker cache contained no missing values or duplicate dates.

| Model | RMSE | MAPE (%) | Directional accuracy |
|---|---:|---:|---:|
| Naive | 14.376964 | 0.837665 | 0.000000 |
| MovingAverage20 | 30.893355 | 2.103135 | 0.547619 |

This is a calculation check for one ticker and two existing baselines, not a new model experiment. It does not establish generalization, profitability, performance on other tickers, or superiority to ARIMA. The committed ARIMA, allocation, notebook, and dashboard artifacts were inspected but not regenerated in this check. No live data download, deployment, or StockGro execution was verified.

After installing the documented requirements, run this from the repository root. It reads the committed cache and stored metrics without saving new outputs or calling an external service:

```bash
python - <<'PY'
from math import isclose
from pathlib import Path
import pandas as pd
from src.data_fetch import load_cached_ticker
from src.preprocessing import clean_ticker_frame
from src.forecasting import (
    split_train_validation,
    naive_validation_forecast,
    moving_average_validation_forecast,
)
from src.evaluation import evaluate_forecast

ticker = "DRREDDY.NS"
raw = load_cached_ticker(ticker)
if raw is None:
    raise SystemExit("Missing committed data/raw/DRREDDY_NS.csv cache")
series = clean_ticker_frame(ticker, raw).set_index("Date")["Price"]
train, validation = split_train_validation(series)
print(f"Training: {len(train)}; validation: {len(validation)} observations")
stored = pd.read_csv(Path("outputs/forecasts/model_evaluation.csv"))
predictions = {
    "Naive": naive_validation_forecast(train, validation),
    "MovingAverage20": moving_average_validation_forecast(train, validation),
}
for name, prediction in predictions.items():
    result = evaluate_forecast(ticker, name, validation, prediction, float(train.iloc[-1]))
    expected = stored.loc[(stored["Ticker"] == ticker) & (stored["Model"] == name)]
    if len(expected) != 1:
        raise SystemExit(f"Expected one stored metric row for {ticker}/{name}")
    for metric in ("RMSE", "MAPE", "Directional Accuracy", "Selection Score"):
        if not isclose(result[metric], float(expected.iloc[0][metric]), rel_tol=1e-10, abs_tol=1e-10):
            raise SystemExit(f"Stored metric mismatch: {name}/{metric}")
    print(name, result)
print("Both baseline calculations match the committed metrics.")
PY
```

## Remaining validity and data questions

- Align the forecast protocol across models and reserve a later test period that is not used to select a model or tune the custom score.
- Evaluate the intended five-day horizon explicitly, with documented origins, an appropriate baseline, and sample counts.
- Preprocessing currently applies forward-fill and backward-fill before splitting. Backward-fill can use future observations when gaps occur. That issue did not arise in the inspected raw DRREDDY cache, but the pipeline needs training-aware missing-data handling before wider evaluation.
- Future dates use pandas business days, which do not exclude exchange holidays.
- Cached CSVs are described as `yfinance` data by `src/data_fetch.py`. An upstream data license and permission to redistribute those CSVs are not recorded in the repository; confirm the applicable provider terms before reuse or redistribution.
- No historical challenge placement or live trading return is established by the manual execution templates.

The root source and documentation are the current project. `StockGro_Final_Submission/` and its ZIP are historical submission snapshots; they may retain earlier descriptions and should not be used as current evaluation evidence.
