"""Central configuration for the gold price analysis system.

Every module imports paths / ticker mappings / hyper-parameters from here so
that the pipeline can be re-pointed by editing one file only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Project layout
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

for _d in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

# --------------------------------------------------------------------------
# Data specification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Asset:
    """One column of the modelling dataset."""

    key: str          # column name used throughout the pipeline
    ticker: str       # Yahoo Finance symbol
    label: str        # human readable label (plots / UI)


# Gold is the target; the rest are macro drivers / proxies.
ASSETS: List[Asset] = [
    Asset("gold", "GC=F", "Gold Futures (USD/oz)"),
    Asset("usdinr", "INR=X", "USD/INR"),
    Asset("crude_oil", "CL=F", "Crude Oil WTI (USD/bbl)"),
    Asset("sp500", "^GSPC", "S&P 500"),
    Asset("us10y", "^TNX", "US 10Y Yield (%)"),
    Asset("dxy", "DX-Y.NYB", "US Dollar Index"),
    Asset("silver", "SI=F", "Silver Futures (USD/oz)"),
    Asset("gld", "GLD", "GLD ETF (spot proxy)"),
]

TARGET: str = "gold"
HORIZON_DAYS: int = int(os.getenv("FORECAST_HORIZON", "5"))  # predict T+5
START_DATE: str = os.getenv("DATA_START_DATE", "2010-01-01")
END_DATE: str | None = os.getenv("DATA_END_DATE") or None

# Column of the merged frame that holds the raw target level.
TARGET_COL: str = "gold_close"

# Lags / rolling windows used by the feature builder.
LAGS: List[int] = [1, 2, 3, 5, 10, 21]
ROLL_WINDOWS: List[int] = [5, 10, 21, 63]

# FRED series used as the inflation driver (fetched as CSV, no API key needed).
FRED_CPI_SERIES = "CPIAUCSL"


@dataclass
class TrainConfig:
    """Split and model hyper-parameters."""

    test_size: float = 0.15          # fraction of most recent rows held out
    valid_size: float = 0.10         # fraction used for early stopping / tuning
    n_splits_cv: int = 5             # walk-forward folds
    rf_n_estimators: int = 400
    rf_max_depth: int | None = 18
    xgb_n_estimators: int = 600
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 6
    xgb_subsample: float = 0.9
    xgb_colsample: float = 0.9
    xgb_early_stopping: int = 50
    random_state: int = 42


TRAIN_CFG = TrainConfig()

# Feature blocks that must never be scaled together with the target.
DATE_COL: str = "date"
META_COLUMNS: List[str] = [DATE_COL]


def ticker_map() -> Dict[str, str]:
    """Return ``{column_key: yahoo_ticker}``."""
    return {a.key: a.ticker for a in ASSETS}


def asset_label(key: str) -> str:
    """Friendly label for a column key, falling back to the key itself."""
    for a in ASSETS:
        if a.key == key:
            return a.label
    return key
