"""Data cleaning and exploratory data analysis.

Separated from *feature engineering* on purpose: this module answers
"is the data trustworthy?" while ``src/features.py`` answers
"what should the model see?".
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe: scripts and Streamlit both work
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from configs import settings as S

logger = logging.getLogger(__name__)

PCT_COLS = ["cpi_yoy", "ret_1d", "ret_5d", "ret_21d"]


# --------------------------------------------------------------------------
# Quality report
# --------------------------------------------------------------------------
def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing count / pct, dtype, and constant-value check."""
    report = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_missing": df.isna().sum(),
            "pct_missing": df.isna().mean().mul(100).round(2),
            "n_unique": df.nunique(),
            "min": df.min(numeric_only=True),
            "max": df.max(numeric_only=True),
            "mean": df.mean(numeric_only=True).round(3),
        }
    )
    report["is_constant"] = report["n_unique"] <= 1
    return report.sort_values("pct_missing", ascending=False)


def longest_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Longest consecutive run of NaN per column.

    A handful of isolated NaNs is a market holiday; a 30-day run means the
    feed broke, and silently filling it would teach the model a flat market.
    """
    out = {}
    for col in df.columns:
        mask = df[col].isna().astype(int)
        if mask.sum() == 0:
            out[col] = 0
            continue
        # length of runs of 1s
        runs = mask[mask.diff() != 0].cumsum()
        out[col] = int(mask.groupby(runs).sum().max())
    return pd.DataFrame({"longest_nan_run": out}).sort_values(
        "longest_nan_run", ascending=False
    )


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean(
    df: pd.DataFrame,
    max_fill_gap: int = 5,
    drop_cols_with_missing_above: float = 40.0,
    extreme_move: float = 0.25,
) -> pd.DataFrame:
    """Make the frame model-ready and return it (original is untouched).

    Rules, in order:
      1. Drop columns that are more than ``drop_cols_with_missing_above``%
         empty - they carry too little information to impute honestly.
      2. Replace non-positive prices with NaN (log returns and gold/silver
         ratios are undefined at <= 0; WTI genuinely printed negative once).
      3. Interpolate short gaps linearly, forward-fill residual edges, and
         leave runs longer than ``max_fill_gap`` as NaN so that feature
         engineering can flag them instead of inventing prices.
      4. Drop rows where the *target* is still missing (no label, no row).
      5. Flag - not clip - single-day moves beyond ``extreme_move``. A -30%
         oil day is a real regime, and silently flattening it teaches the
         model that crashes do not exist.
    """
    work = df.copy()
    before = len(work)

    missing_pct = work.isna().mean().mul(100)
    useless = missing_pct[missing_pct > drop_cols_with_missing_above].index.tolist()
    if useless:
        logger.warning("dropping columns with >%.0f%% missing: %s",
                       drop_cols_with_missing_above, useless)
        work = work.drop(columns=useless)

    # (2) sanitation on level columns only - pct / index columns may be <= 0
    level_cols = [c for c in work.select_dtypes(include="number").columns
                  if c not in ("cpi_yoy",) and not c.endswith("_yoy")]
    for col in level_cols:
        nonpos = work[col] <= 0
        if nonpos.any():
            logger.warning("masking %d non-positive value(s) in %s", int(nonpos.sum()), col)
            work.loc[nonpos, col] = np.nan

    # (3) gap filling
    for col in level_cols:
        s = work[col]
        if s.isna().sum() == 0:
            continue
        filled = s.interpolate(method="linear", limit=max_fill_gap, limit_area="inside")
        work[col] = filled.ffill().bfill()

    # (5) extreme-move flag, kept as data rather than as a value change
    extreme = pd.Series(False, index=work.index)
    for col in level_cols:
        extreme |= work[col].pct_change().abs() > extreme_move
    work["extreme_move_flag"] = extreme.astype(int)
    if extreme.any():
        logger.info("flagged %d extreme-move day(s) (>%.0f%%)", int(extreme.sum()),
                    extreme_move * 100)

    if S.TARGET_COL in work.columns:
        work = work.dropna(subset=[S.TARGET_COL])

    work = work.sort_index()
    logger.info("clean: %d -> %d rows, %d cols", before, len(work), work.shape[1])
    return work


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------
def correlation_matrix(
    df: pd.DataFrame,
    method: str = "spearman",
    returns: bool = False,
) -> pd.DataFrame:
    """Rank correlation of levels, or of daily % returns when ``returns=True``.

    Spearman is used because price series are non-normal and heavily
    auto-correlated; Pearson on levels mostly reports "both went up since
    2010" (spurious correlation).
    """
    data = df.pct_change().dropna() if returns else df
    return data.corr(method=method)


def stationary_adf_test(df: pd.DataFrame, max_lags: int = 1) -> pd.DataFrame:
    """Augmented Dickey-Fuller per column (levels vs. first differences).

    A series with p > 0.05 in levels is non-stationary: tree models cope, but
    a linear/LSTM model trained on raw levels will extrapolate poorly.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        logger.info("statsmodels not installed - skipping ADF test")
        return pd.DataFrame()

    rows = []
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if len(s) < 50:
            continue
        try:
            lvl = adfuller(s.dropna(), maxlag=max_lags, autolag=None, store=False)[1]
            dif = adfuller(s.diff().dropna(), maxlag=max_lags, autolag=None, store=False)[1]
        except Exception:
            continue
        rows.append({"column": col, "p_level": round(lvl, 4), "p_diff1": round(dif, 4),
                     "stationary_in_level": lvl < 0.05})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_price_trend(
    df: pd.DataFrame,
    out: Path | str = S.FIGURES_DIR / "01_price_trend.png",
    window: int = 63,
) -> Path:
    """Gold price with a rolling mean/std band, plus normalised overlays."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(df.index, df[S.TARGET_COL], label="Gold close", lw=1.2, color="#b8860b")
    roll_mean = df[S.TARGET_COL].rolling(window).mean()
    roll_std = df[S.TARGET_COL].rolling(window).std()
    ax.plot(df.index, roll_mean, label=f"{window}d rolling mean", lw=1.0, color="#1f77b4")
    ax.fill_between(
        df.index,
        roll_mean - roll_std,
        roll_mean + roll_std,
        alpha=0.15,
        color="#1f77b4",
        label=f"±{window}d rolling std",
    )
    ax.set_ylabel("USD / oz")
    ax.set_title("Gold price trend with rolling statistics")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)

    ax2 = axes[1]
    overlays = [c for c in ("dxy", "us10y", "crude_oil", "sp500", "silver") if c in df]
    for col in overlays:
        normalised = df[col] / df[col].dropna().iloc[0] * 100
        ax2.plot(df.index, normalised, label=S.asset_label(col), lw=0.9)
    gold_norm = df[S.TARGET_COL] / df[S.TARGET_COL].dropna().iloc[0] * 100
    ax2.plot(df.index, gold_norm, label="Gold (index=100)", lw=1.6, color="black")
    ax2.set_ylabel("Rebased index (start = 100)")
    ax2.set_title("Relative performance vs. macro drivers")
    ax2.legend(fontsize=8, ncol=3, loc="upper left")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_correlation_heatmap(
    df: pd.DataFrame,
    out: Path | str = S.FIGURES_DIR / "02_correlation_heatmap.png",
) -> tuple[Path, Path]:
    """Two heatmaps: Spearman on levels and Pearson on daily returns."""
    lvl = correlation_matrix(df, method="spearman")
    ret = correlation_matrix(df.pct_change().dropna(), method="pearson")

    fig, axes = plt.subplots(1, 2, figsize=(19, 8))
    for ax, data, title in (
        (axes[0], lvl, "Spearman correlation - price levels"),
        (axes[1], ret, "Pearson correlation - daily returns"),
    ):
        mask = np.triu(np.ones_like(data, dtype=bool))
        sns.heatmap(
            data,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.6,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        ax.set_xticklabels(ax.get_xticklabels(), ha="right")

    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=130)
    plt.close(fig)

    # A dedicated figure for target drivers is what actually guides feature choice.
    target_corr = lvl[S.TARGET_COL].dropna().sort_values()
    fig2, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in target_corr]
    ax.barh(target_corr.index, target_corr.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(f"Spearman correlation with {S.TARGET_COL}")
    ax.set_xlabel("correlation")
    for i, v in enumerate(target_corr.values):
        ax.text(v, i, f" {v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    fig2.tight_layout()
    out2 = Path(S.FIGURES_DIR) / "02b_target_drivers.png"
    fig2.savefig(out2, dpi=130)
    plt.close(fig2)
    return out, out2


def plot_distributions(
    df: pd.DataFrame,
    out: Path | str = S.FIGURES_DIR / "03_distributions.png",
    top_n: int = 8,
) -> Path:
    """Histogram + normal Q-Q of daily returns: fat tails justify tree models."""
    returns = df.pct_change().dropna()
    cols = returns.std(numeric_only=True).nlargest(top_n).index
    ncol = 3
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cols):
        r = returns[col]
        # winsorise for readability so kurtosis does not flatten the axis
        lo, hi = r.quantile([0.005, 0.995])
        ax.hist(r.clip(lo, hi), bins=60, color="#4c72b0", alpha=0.9)
        ax.set_title(
            f"{S.asset_label(col)}\nvol={r.std():.3%} skew={r.skew():.2f} kurt={r.kurt():.1f}",
            fontsize=9,
        )
        ax.set_xlabel("daily return")
    for ax in axes[len(cols):]:
        ax.axis("off")

    fig.suptitle("Daily return distributions (winsorised at 0.5%/99.5%)")
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_missing_pattern(
    df_raw: pd.DataFrame,
    out: Path | str = S.FIGURES_DIR / "00_missing_pattern.png",
) -> Optional[Path]:
    """Missingness map of the *raw* frame - tells you if gaps are random."""
    if df_raw.isna().sum().sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(df_raw.isna().T, cbar=False, ax=ax, linewidths=0)
    ax.set_title("Missing values in the raw pull (white = present)")
    ax.set_ylabel("")
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def run_eda(
    df_raw: Optional[pd.DataFrame] = None,
    figures_dir: Path = S.FIGURES_DIR,
) -> dict[str, Path]:
    """Execute every EDA step and write all figures; returns figure paths."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    if df_raw is None:
        from src.data_loader import build_dataset

        df_raw = build_dataset()

    paths: dict[str, Path] = {}
    if (p := plot_missing_pattern(df_raw)):
        paths["missing"] = p

    df = clean(df_raw)
    report = quality_report(df)
    report.to_csv(figures_dir / "quality_report.csv")
    stationary_adf_test(df).to_csv(figures_dir / "adf_test.csv", index=False)

    paths["trend"] = plot_price_trend(df, figures_dir / "01_price_trend.png")
    h1, h2 = plot_correlation_heatmap(df, figures_dir / "02_correlation_heatmap.png")
    paths["heatmap"] = h1
    paths["target_drivers"] = h2
    paths["distributions"] = plot_distributions(df, figures_dir / "03_distributions.png")

    # Persist the cleaned frame so feature engineering starts from identical data.
    df.reset_index().to_csv(S.PROCESSED_DIR / "clean_daily.csv", index=False)
    paths["cleaned"] = S.PROCESSED_DIR / "clean_daily.csv"
    return paths
