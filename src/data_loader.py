"""Multi-source market data layer.

Design decisions
----------------
* One source of truth per asset (Yahoo Finance close + FRED CPI).
* Every download is cached as CSV under ``data/raw`` so the pipeline can be
  re-run offline and so results are reproducible.
* Longest-common trading calendar is used for alignment: markets have
  different holidays, therefore we inner-join on business days and then
  forward-fill at most a couple of days of gaps (a 1-2 day gap is a holiday,
  a 10 day gap is real missing data and is left alone for the EDA step).
"""
from __future__ import annotations

import io
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from configs import settings as S

# yfinance internals still use the deprecated ``inplace=`` keyword against
# pandas >= 3.1; the noise would bury our own log lines.
warnings.filterwarnings("ignore", message=".*inplace keyword.*")

logger = logging.getLogger(__name__)

_RAW_CACHE_VERSION = "v1"


def _cache_path(name: str) -> Path:
    return S.RAW_DIR / f"{name}.{_RAW_CACHE_VERSION}.csv"


# --------------------------------------------------------------------------
# Yahoo Finance
# --------------------------------------------------------------------------
def fetch_yahoo_history(
    ticker: str,
    start: str = S.START_DATE,
    end: Optional[str] = S.END_DATE,
    use_cache: bool = True,
) -> pd.Series:
    """Return the daily *Close* series for one Yahoo symbol.

    Falls back to the on-disk CSV cache when the network (or yfinance rate
    limiting) is unavailable, which keeps the rest of the pipeline runnable.
    """
    cache = _cache_path(ticker.replace("^", "idx_").replace("=", "eq").replace(".", "_"))

    if use_cache and cache.exists():
        cached = pd.read_csv(cache, index_col=0, parse_dates=True)["close"]
        cached.index = pd.DatetimeIndex(cached.index)
        logger.debug("cache hit: %s (%d rows)", ticker, len(cached))
        return cached

    try:
        import yfinance as yf

        raw = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,       # we want the actual quoted close
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            raise ValueError(f"yfinance returned an empty frame for {ticker}")

        # yfinance >= 0.2.50 returns a MultiIndex column (field, ticker).
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        close = pd.to_numeric(raw["Close"], errors="coerce").rename("close")
        close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
        close = close[~close.index.duplicated(keep="last")].dropna()

        close.to_csv(cache)
        logger.info("downloaded %-10s -> %d rows", ticker, len(close))
        return close

    except Exception as exc:  # network / rate-limit / schema change
        if cache.exists():
            logger.warning("download failed for %s (%s); using cache", ticker, exc)
            return fetch_yahoo_history(ticker, use_cache=True)
        raise RuntimeError(
            f"Could not download {ticker} and no cache exists. "
            "Check your connection or pass synthetic=True to build_dataset()."
        ) from exc


# --------------------------------------------------------------------------
# Inflation (FRED, keyless CSV endpoint)
# --------------------------------------------------------------------------
def fetch_cpi(start: str = S.START_DATE, use_cache: bool = True) -> pd.Series:
    """US CPI (all urban consumers, monthly, NSA) from FRED.

    FRED publishes a plain CSV dump that needs no API key. The monthly series
    is forward-filled onto daily frequency inside the merged frame.
    """
    cache = _cache_path(f"fred_{S.FRED_CPI_SERIES}")
    if use_cache and cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df["cpi"]

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    try:
        resp = requests.get(
            url,
            params={"id": S.FRED_CPI_SERIES, "cosd": start},
            timeout=30,
        )
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        col = df.columns[1]                      # e "."observation_date"/"value" like names
        out = (
            df.rename(columns={df.columns[0]: "date", col: "cpi"})
            .assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce"))
            .set_index("date")["cpi"]
            .pipe(pd.to_numeric, errors="coerce")
            .dropna()
        )
        out.to_csv(cache)
        logger.info("downloaded FRED %s -> %d rows", S.FRED_CPI_SERIES, len(out))
        return out
    except Exception as exc:
        if cache.exists():
            logger.warning("FRED fetch failed (%s); using cache", exc)
            return fetch_cpi(use_cache=True)
        logger.warning("inflation data unavailable (%s) - continuing without CPI", exc)
        return pd.Series(dtype=float, name="cpi")


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def build_dataset(
    start: str = S.START_DATE,
    end: Optional[str] = S.END_DATE,
    use_cache: bool = True,
    with_inflation: bool = True,
) -> pd.DataFrame:
    """Merge every asset into one daily wide frame.

    Returns
    -------
    DataFrame indexed by ``date`` with columns named after
    ``settings.ASSETS`` keys plus ``cpi`` and ``cpi_yoy``.
    """
    series_map: dict[str, pd.Series] = {}
    for asset in S.ASSETS:
        try:
            series_map[asset.key] = fetch_yahoo_history(asset.ticker, start, end, use_cache)
        except RuntimeError as exc:
            logger.error("skipping %s: %s", asset.key, exc)

    if S.TARGET not in series_map:
        raise SystemExit("Target series (gold) could not be loaded - aborting.")

    df = pd.concat(series_map, axis=1)

    # Inner-join on the intersection of trading days, then tolerate short
    # holiday gaps (<= 3 business days) with a limited ffill.
    df = df.dropna(how="all").sort_index()
    df = df.asfreq("B")
    df = df.ffill(limit=3)

    # Rename to readable, pipeline-stable column names.
    df = df.rename(columns={S.TARGET: S.TARGET_COL})
    df.index.name = S.DATE_COL

    if with_inflation:
        cpi = fetch_cpi(start, use_cache)
        if not cpi.empty:
            # 1) YoY at the NATIVE monthly frequency. Doing pct_change(12) on a
            #    daily forward-filled copy would usually land inside the same
            #    month and return 0.0 - the classic resample bug.
            #    Grouping by Period avoids the freq-alias differences between
            #    pandas 2.x ("M") and 3.x ("ME").
            monthly = cpi.groupby(cpi.index.to_period("M")).last()
            yoy_monthly = monthly.pct_change(12).mul(100)
            yoy_monthly.index = yoy_monthly.index.to_timestamp(how="end")
            # 2) publish both onto the daily trading calendar
            df["cpi"] = cpi.reindex(df.index, method="ffill")
            df["cpi_yoy"] = yoy_monthly.reindex(df.index, method="ffill")

    logger.info("dataset assembled: %s rows x %s cols", *df.shape)
    return df


def load_processed(path: Path) -> pd.DataFrame:
    """Read a previously saved modelling frame."""
    df = pd.read_csv(path, parse_dates=[S.DATE_COL]).set_index(S.DATE_COL)
    return df


def synthetic_dataset(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    """Deterministic fake panel for offline demos and unit tests.

    It reproduces the *shape* of the real data (regime drift in gold, negative
    gold/dxy relation, positive gold/silver relation) so downstream code can be
    exercised without network access. Never use it for modelling conclusions.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    shocks = rng.normal(0, 0.008, n)
    drift = np.linspace(0, 0.55, n)
    gold = 1200 * np.exp(drift + np.cumsum(shocks))
    dxy = 100 * np.exp(np.cumsum(rng.normal(-0.0002, 0.004, n)) - 0.25 * np.cumsum(shocks))
    silver = 18 * np.exp(drift * 0.8 + np.cumsum(rng.normal(0, 0.012, n)))
    us10y = 2.5 + 1.5 * np.cumsum(rng.normal(0, 0.02, n))
    sp500 = 2000 * np.exp(np.cumsum(rng.normal(0.0004, 0.009, n)))
    crude = 55 * np.exp(np.cumsum(rng.normal(0, 0.013, n)))
    usdinr = 65 * np.exp(np.cumsum(rng.normal(0.00015, 0.0025, n)))
    gld = gold / 10.4 + rng.normal(0, 1.5, n)
    cpi = 240 * np.exp(np.linspace(0, 0.35, n))

    df = pd.DataFrame(
        {
            S.TARGET_COL: gold,
            "usdinr": usdinr,
            "crude_oil": crude,
            "sp500": sp500,
            "us10y": np.clip(us10y, 0.2, 6),
            "dxy": dxy,
            "silver": silver,
            "gld": gld,
            "cpi": cpi,
        },
        index=idx,
    )
    df["cpi_yoy"] = df["cpi"].pct_change(12).mul(100)
    df.index.name = S.DATE_COL
    # inject realistic-looking holes so the preprocessing step has work to do
    for col in ("crude_oil", "sp500", "dxy"):
        holes = rng.choice(n, size=int(n * 0.008), replace=False)
        df.iloc[holes, df.columns.get_loc(col)] = np.nan
    return df
