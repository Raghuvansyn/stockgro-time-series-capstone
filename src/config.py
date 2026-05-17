"""Central configuration for the StockGro time series capstone."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
FORECASTS_DIR = OUTPUTS_DIR / "forecasts"
REPORT_ASSETS_DIR = OUTPUTS_DIR / "report_assets"
REPORTS_DIR = PROJECT_ROOT / "reports"

START_DATE = "2021-01-01"
END_DATE = "2025-12-31"
TRAIN_END_EXCLUSIVE = "2025-07-01"
VALIDATION_START = "2025-07-01"
VALIDATION_END = "2025-12-31"
FORECAST_HORIZON = 5

INITIAL_CAPITAL = 1_000_000
RISK_FREE_RATE = 0.0
TRADING_DAYS_PER_YEAR = 252
RANDOM_SEED = 42


@dataclass(frozen=True)
class StockMeta:
    """Metadata for one NSE-listed security used in the universe."""

    ticker: str
    company: str
    sector: str


STOCK_UNIVERSE: tuple[StockMeta, ...] = (
    StockMeta("HDFCBANK.NS", "HDFC Bank", "Banking"),
    StockMeta("ICICIBANK.NS", "ICICI Bank", "Banking"),
    StockMeta("TCS.NS", "Tata Consultancy Services", "IT"),
    StockMeta("INFY.NS", "Infosys", "IT"),
    StockMeta("SUNPHARMA.NS", "Sun Pharmaceutical", "Pharma"),
    StockMeta("DRREDDY.NS", "Dr. Reddy's Laboratories", "Pharma"),
    StockMeta("RELIANCE.NS", "Reliance Industries", "Energy"),
    StockMeta("ITC.NS", "ITC", "FMCG"),
    StockMeta("HINDUNILVR.NS", "Hindustan Unilever", "FMCG"),
    StockMeta("MARUTI.NS", "Maruti Suzuki", "Auto"),
    StockMeta("TATASTEEL.NS", "Tata Steel", "Metals"),
    StockMeta("LT.NS", "Larsen & Toubro", "Infrastructure"),
)

TICKERS = [stock.ticker for stock in STOCK_UNIVERSE]
STOCK_METADATA = {stock.ticker: stock for stock in STOCK_UNIVERSE}


def ensure_directories() -> None:
    """Create all expected project directories."""

    for path in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        FORECASTS_DIR,
        REPORT_ASSETS_DIR,
        REPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ticker_to_filename(ticker: str) -> str:
    """Return a filesystem-safe filename stem for an NSE ticker."""

    return ticker.replace(".", "_").replace("/", "_")
