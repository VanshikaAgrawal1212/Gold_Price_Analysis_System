"""Feature engineering, leakage-safe splitting and scaling.

The single most important rule in this module: **a feature at row *t* may only
use information available at the close of business day *t***, while the label
is the price at *t + horizon*. Anything looser (e.g. ``rolling(center=True)``,
or ``shift(-1)`` on a driver) silently leaks the answer into training and
produces a RMSE that collapses the moment you deploy it.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from configs import settings as S

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Feature builder
# --------------------------------------------------------------------------
# Columns that are already transformed / already binary: differencing or
# lagging them is meaningless, and pct_change on a 0/1 flag divides by zero
# whenever the previous day was 0, which manufactures NaN features.
NO_TRANSFORM_COLS = frozenset({"extreme_move_flag", "gold_vol_regime"})


def add_returns_features(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """1/5/21-day simple returns and daily log returns for each driver.

    The log is taken over a *clipped positive* ratio: a zero or negative print
    (WTI went negative in April 2020) would otherwise produce -inf/+inf, and a
    single inf silently poisons every scaled model downstream.

    Flag/regime columns and any other 0/1 series are skipped - see
    ``NO_TRANSFORM_COLS``.
    """
    out = df.copy()
    cols = cols or [c for c in df.columns if c not in ("cpi", "cpi_yoy")]
    for col in cols:
        if col not in out or col in NO_TRANSFORM_COLS:
            continue
        values = out[col].dropna().unique()
        if len(values) <= 2:  # binary or constant: returns are noise
            continue
        for n in (1, 5, 21):
            out[f"{col}_ret_{n}d"] = out[col].pct_change(n)
        ratio = out[col] / out[col].shift(1)
        out[f"{col}_logret_1d"] = np.log(ratio.where(ratio > 0))
    return out


def add_lag_features(df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
    """Lagged 1-day *returns*, never lagged price levels.

    This distinction is the single biggest accuracy lever in financial ML.
    A tree model cannot extrapolate: a split learned at "gold level ~ 1900"
    is meaningless in a test window where gold trades at 3,900, and the model
    silently falls back to the behaviour of its training regime. Lagged
    returns are stationary and scale-free, so the same relationship holds in
    2010 and in 2026.
    """
    out = df.copy()
    for col in cols:
        src = f"{col}_ret_1d"
        if src not in out:
            continue
        for lag in lags:
            out[f"{col}_ret1d_lag_{lag}"] = out[src].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame, cols: List[str], windows: List[int]
) -> pd.DataFrame:
    """Trailing, scale-free trend/volatility descriptors.

    Deliberately excluded: the rolling *mean level* and rolling *std of level*,
    both non-stationary for the same extrapolation reason as lagged levels.
    Kept instead:
      * ``_rng_pos_``  - where the price sits inside its own trailing range (0-1)
      * ``_z_``        - distance from the trailing mean, in std units
      * ``_rvol_``     - rolling std of daily returns (annualisable vol)

    ``min_period=max(2, w // 2)`` keeps early rows from being all-NaN while
    still refusing to compute a 63-day statistic on 5 observations.
    """
    out = df.copy()
    for col in cols:
        if col not in out:
            continue
        ret_col = f"{col}_ret_1d"
        for w in windows:
            roll = out[col].rolling(w, min_periods=max(2, w // 2))
            span = (roll.max() - roll.min()).replace(0, np.nan)
            out[f"{col}_rng_pos_{w}"] = (out[col] - roll.min()) / span
            out[f"{col}_z_{w}"] = (out[col] - roll.mean()) / roll.std().replace(0, np.nan)
            if ret_col in out:
                out[f"{col}_rvol_{w}"] = out[ret_col].rolling(w).std()
    return out


def add_target_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features built *from the target history only* (always allowed)."""
    out = df.copy()
    g = out[S.TARGET_COL]
    out["gold_ret_1d"] = g.pct_change()
    out["gold_ret_5d"] = g.pct_change(5)
    out["gold_ret_21d"] = g.pct_change(21)
    out["gold_vol_21"] = g.pct_change().rolling(21).std()
    out["gold_vol_63"] = g.pct_change().rolling(63).std()
    # realised-vol regime flag: models handle level shifts better with a flag
    out["gold_vol_regime"] = (
        out["gold_vol_21"] > out["gold_vol_21"].rolling(252).median()
    ).astype(int)
    out["gold_ma_ratio_21_63"] = (
        g.rolling(21).mean() / g.rolling(63).mean()
    ) - 1.0
    out["gold_change_sign_5d"] = np.sign(g.pct_change(5)).fillna(0)
    return out


def add_macro_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-asset relationships with economic meaning.

    * gold/silver ratio  - classic relative-value signal
    * real yield proxy   - gold's main competitor is the inflation-adjusted
      bond yield; TIPS are not on Yahoo, so 10Y nominal minus CPI YoY.
    * gold/oil ratio     - terms-of-trade / inflation sentiment
    * dollar pressure    - dxy 21d change (gold is USD-denominated)
    """
    out = df.copy()
    if {"gold_close", "silver"} <= set(out.columns):
        out["gold_silver_ratio"] = out["gold_close"] / out["silver"]
        out["gold_silver_ratio_rm21"] = out["gold_silver_ratio"].rolling(21).mean()
    if {"us10y", "cpi_yoy"} <= set(out.columns):
        out["real_yield_proxy"] = out["us10y"] - out["cpi_yoy"]
        out["real_yield_z_252"] = (
            (out["us10y"] - out["cpi_yoy"])
            - (out["us10y"] - out["cpi_yoy"]).rolling(252).mean()
        ) / (out["us10y"] - out["cpi_yoy"]).rolling(252).std()
    if {"gold_close", "crude_oil"} <= set(out.columns):
        out["gold_oil_ratio"] = out["gold_close"] / out["crude_oil"]
    if "dxy" in out.columns:
        out["dxy_chg_21"] = out["dxy"].pct_change(21)
    if {"usdinr", "gold_close"} <= set(out.columns):
        # INR-quoted gold: the price an Indian buyer actually sees
        out["gold_inr"] = out["gold_close"] * out["usdinr"]
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Seasonality and roll effects; cyclical encodings avoid 11≈-1 wraps."""
    out = df.copy()
    idx: pd.DatetimeIndex = out.index
    out["dow"] = idx.day_of_week
    out["month"] = idx.month
    out["quarter"] = idx.quarter
    out["day_of_year"] = idx.dayofyear
    out["week_of_year"] = idx.isocalendar().week.astype(int)
    out["year"] = idx.year
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    # Indian festive demand seasonality - gold's largest physical bid
    out["is_festive_season"] = out["month"].isin([9, 10, 11]).astype(int)
    out["is_year_end"] = out["month"].isin([12, 1]).astype(int)
    # month-end / month-start rebalancing flows
    out["is_month_start"] = idx.is_month_start.astype(int)
    out["is_month_end"] = idx.is_month_end.astype(int)
    return out


def build_features(
    df: pd.DataFrame,
    horizon: int = S.HORIZON_DAYS,
    lags: List[int] = S.LAGS,
    windows: List[int] = S.ROLL_WINDOWS,
    dropna: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Full feature pipeline -> (frame with ``y_target``, feature names).

    The label is ``gold_close`` at *t + horizon* expressed as a **return**
    rather than a level. Two reasons:
      * a return target is stationary, so one model is valid across a decade
        of price regimes (1200 USD and 2700 USD need different level models);
      * it converts the evaluation metrics into directly actionable terms
        ("expected move"), while still being convertible back to a price.
    """
    data = df.copy()
    data = add_target_derived_features(data)
    data = add_returns_features(data)
    data = add_lag_features(data, [S.TARGET_COL, "dxy", "us10y", "sp500", "crude_oil"], lags)
    data = add_rolling_features(data, [S.TARGET_COL, "dxy", "us10y", "silver"], windows)
    data = add_macro_spread_features(data)
    data = add_calendar_features(data)

    # Ratio features divide by things that can be zero (a flat rolling range,
    # a zero silver print). Normalise every infinity to NaN *before* the label
    # step so the imputer - which only knows about NaN - can handle it.
    data = data.replace([np.inf, -np.inf], np.nan)

    # ---- LABEL: forward return over the horizon (uses FUTURE data by design)
    data["y_target"] = data[S.TARGET_COL].pct_change(horizon).shift(-horizon)
    # also keep the future level so we can report predictions in USD
    data["y_level"] = data[S.TARGET_COL].shift(-horizon)

    # Feature set = everything numeric except the label block and raw levels
    # we deliberately keep un-scaled out of the model.
    exclude = {
        "y_target",
        "y_level",
        "cpi",                      # non-stationary index level
        "gold_inr",                 # level x level: non-stationary, display only
        S.TARGET_COL,               # the price we are predicting (leakage!)
    }
    feature_cols = [
        c
        for c in data.select_dtypes(include="number").columns
        if c not in exclude
    ]
    data = data.dropna(subset=["y_target"]) if dropna else data

    dropped = data[feature_cols].isna().all().sum()
    if dropped:
        all_nan = data[feature_cols].isna().all()
        data = data.drop(columns=all_nan[all_nan].index)
        feature_cols = [c for c in feature_cols if c in data.columns]
        logger.info("dropped %d all-NaN feature columns", dropped)

    logger.info("feature matrix: %d rows x %d features", len(data), len(feature_cols))
    return data, feature_cols


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
@dataclass
class TimeSplit:
    """Contiguous chronological split - never random for time series."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    feature_cols: List[str]

    def summary(self) -> str:
        def rng(d: pd.DataFrame) -> str:
            return f"{d.index.min():%Y-%m-%d}..{d.index.max():%Y-%m-%d} ({len(d)})"

        return (
            f"train {rng(self.train)} | valid {rng(self.valid)} | test {rng(self.test)}"
        )


def time_series_split(
    data: pd.DataFrame,
    feature_cols: List[str],
    test_size: float = S.TRAIN_CFG.test_size,
    valid_size: float = S.TRAIN_CFG.valid_size,
) -> TimeSplit:
    """Chronological 70/15/10-style split with an embargo gap.

    The ``embargo = horizon`` rows removed between blocks stop a training
    label (which looks ``horizon`` days forward) from overlapping the
    validation window.
    """
    n = len(data)
    embargo = S.HORIZON_DAYS
    n_test = int(n * test_size)
    n_valid = int(n * valid_size)
    n_train = n - n_test - n_valid - 2 * embargo

    if min(n_train, n_valid, n_test) < 50:
        raise ValueError(
            f"Not enough rows ({n}) for a {test_size}/{valid_size} split with "
            f"horizon {embargo}. Extend the date range or lower the horizon."
        )

    train = data.iloc[:n_train]
    valid = data.iloc[n_train + embargo : n_train + embargo + n_valid]
    test = data.iloc[n_train + embargo + n_valid + embargo :]
    if len(test) < 20:
        raise ValueError("test block collapsed after embargo trimming")
    return TimeSplit(train, valid, test, feature_cols)


def walk_forward_folds(
    data: pd.DataFrame,
    n_splits: int = S.TRAIN_CFG.n_splits_cv,
):
    """Expanding-window CV indices, an alternative to a single hold-out.

    Yields (train_idx, test_idx) where the training set always ends
    ``horizon`` rows before the test block starts.
    """
    n = len(data)
    embargo = S.HORIZON_DAYS
    min_train = int(n * 0.4)
    fold_size = (n - min_train) // (n_splits + 1)
    if fold_size < 20:
        raise ValueError("Series too short for walk-forward CV")

    for i in range(n_splits):
        train_end = min_train + (i + 1) * fold_size
        test_start = train_end + embargo
        test_end = min(test_start + fold_size, n)
        if test_start >= n or test_end <= test_start:
            break
        yield np.arange(0, train_end), np.arange(test_start, test_end)


# --------------------------------------------------------------------------
# Scaling
# --------------------------------------------------------------------------
class lagged_impute(BaseEstimator, TransformerMixin):
    """Fill NaNs with the column median *learned on train only*.

    Needed because rolling features leave NaNs at the start of the frame and
    scikit-learn estimators refuse NaN input. Medians come from the fit set, so
    no statistics from the future reach the past.
    """

    def fit(self, X, y=None):
        Xa = np.asarray(X, dtype=float)
        # inf is as unusable as NaN here, and far more damaging if it survives
        Xa = np.where(np.isfinite(Xa), Xa, np.nan)
        with warnings.catch_warnings():
            # a column that is entirely NaN inside a small CV fold slice makes
            # nanmedian shout; the 0.0 fallback below is the intended answer
            warnings.simplefilter("ignore", RuntimeWarning)
            self.median_ = np.nanmedian(Xa, axis=0)
        # an all-NaN column has no median: fall back to 0.0
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        return self

    def transform(self, X):
        Xa = np.asarray(X, dtype=float)
        bad = ~np.isfinite(Xa)
        rows, cols = np.where(bad)
        Xa[rows, cols] = self.median_[cols]
        return Xa


def make_scaler(kind: str = "robust"):
    """``robust`` (median/IQR) is the default: price returns have fat tails,
    so StandardScaler's mean/sigma gets dragged by crash days."""
    if kind == "robust":
        return RobustScaler()
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        from sklearn.preprocessing import MinMaxScaler

        return MinMaxScaler()
    raise ValueError(f"unknown scaler: {kind}")


def build_preprocess_pipeline(scaler: str = "robust") -> Pipeline:
    """impute -> scale, as one object that can be joblib-saved with the model."""
    from sklearn.compose import ColumnTransformer  # noqa: F401  (kept for clarity)

    from sklearn.pipeline import Pipeline as SkPipeline

    return SkPipeline(
        steps=[("impute", lagged_impute()), ("scale", make_scaler(scaler))]
    )


def to_xy(data: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y_return, y_level) as float arrays."""
    X = data[feature_cols].to_numpy(dtype=float)
    y = data["y_target"].to_numpy(dtype=float)
    y_level = data["y_level"].to_numpy(dtype=float)
    return X, y, y_level
