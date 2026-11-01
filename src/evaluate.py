"""Evaluation: regression metrics translated into trading meaning.

Raw RMSE on a forward *return* is nearly useless to a reader in isolation
(0.008 could be excellent or terrible depending on the volatility regime), so
every metric here is reported twice:

* **return space** - RMSE / MAE / R2, comparable across models;
* **price space**  - the same errors multiplied by the entry price, i.e. USD
  per ounce, which is what a desk actually cares about;
* **direction**    - hit rate and a skill score vs. the persistence baseline,
  because a model can have low RMSE and still be useless if it always
  predicts "flat".
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from configs import settings as S

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error (sklearn>=1.6 moved ``squared=False`` out)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error of the *forecast price* (USD terms)."""
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of *called* days where the predicted sign matches the realised one.

    Rows are only counted when both sides have a sign: including a zero
    prediction would score a constant-zero forecaster at 0% (every comparison
    is "wrong") even though it simply never placed a trade. Excluding it yields
    NaN - "no calls, no opinion" - which is the honest answer.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = (y_true != 0) & (y_pred != 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))


@dataclass
class MetricSet:
    """All numbers for one model on one dataset."""

    model: str
    split: str
    n: int
    rmse_ret: float
    mae_ret: float
    r2_ret: float
    rmse_usd: float
    mae_usd: float
    hit_rate: float
    naive_rmse_usd: Optional[float] = None
    skill_vs_naive: Optional[float] = None
    note: str = ""

    def to_row(self) -> Dict[str, object]:
        return asdict(self)


def evaluate_predictions(
    y_true_ret: np.ndarray,
    y_pred_ret: np.ndarray,
    entry_price: np.ndarray,
    model_name: str,
    split: str = "test",
    naive_pred_ret: Optional[np.ndarray] = None,
) -> MetricSet:
    """Bundle return-space and price-space metrics for one model.

    ``entry_price`` is the *known* price at t (the row the forecast is made
    from); multiplying a return error by it converts to USD error on the
    T+horizon price.
    """
    y_true_ret = np.asarray(y_true_ret, dtype=float)
    y_pred_ret = np.asarray(y_pred_ret, dtype=float)
    entry = np.asarray(entry_price, dtype=float)

    true_price = entry * (1 + y_true_ret)
    pred_price = entry * (1 + y_pred_ret)

    naive_rmse_usd = skill = None
    if naive_pred_ret is not None:
        naive_price = entry * (1 + np.asarray(naive_pred_ret, dtype=float))
        naive_rmse_usd = rmse(true_price, naive_price)
        ours = rmse(true_price, pred_price)
        # >0 means we beat naive; 0.10 == 10% error reduction
        skill = 1 - ours / naive_rmse_usd if naive_rmse_usd > 0 else None

    return MetricSet(
        model=model_name,
        split=split,
        n=len(y_true_ret),
        rmse_ret=rmse(y_true_ret, y_pred_ret),
        mae_ret=mae(y_true_ret, y_pred_ret),
        r2_ret=r2(y_true_ret, y_pred_ret),
        rmse_usd=rmse(true_price, pred_price),
        mae_usd=mae(true_price, pred_price),
        hit_rate=directional_accuracy(y_true_ret, y_pred_ret),
        naive_rmse_usd=naive_rmse_usd,
        skill_vs_naive=skill,
    )


def metrics_table(results: List[MetricSet]) -> pd.DataFrame:
    """Sortable comparison frame, best skill first."""
    df = pd.DataFrame([r.to_row() for r in results])
    df = df.set_index(["model", "split"]).sort_values(
        "skill_vs_naive", ascending=False
    )
    return df.round(5)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_predictions(
    dates: pd.DatetimeIndex,
    actual_price: np.ndarray,
    preds_usd: Dict[str, np.ndarray],
    out: Path | str = S.FIGURES_DIR / "04_predictions.png",
    title: str = "Held-out window: actual vs. predicted gold price",
) -> Path:
    """Overlay each model's implied price path on the realised one."""
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(dates, actual_price, label="Actual gold (T+%d)" % S.HORIZON_DAYS,
            lw=2.0, color="black")
    cmap = plt.get_cmap("tab10")
    for i, (name, pred) in enumerate(preds_usd.items()):
        ax.plot(dates, pred, label=name, lw=1.2, alpha=0.85, color=cmap(i % 10))

    ax.set_title(title)
    ax.set_ylabel("USD / oz")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_error_distribution(
    errors_usd: Dict[str, np.ndarray],
    out: Path | str = S.FIGURES_DIR / "05_errors.png",
) -> Path:
    """Forecast-error distributions: width tells volatility risk, the mean's
    offset from zero tells you the model is biased (e.g. always pessimistic)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    keys = list(errors_usd)
    axes[0].violinplot([errors_usd[k] for k in keys], showmeans=True, widths=0.85)
    axes[0].set_xticks(range(1, len(keys) + 1))
    axes[0].set_xticklabels(keys, rotation=30, ha="right")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("USD error (pred - actual)")
    axes[0].set_title("Forecast error by model")

    for k in keys:
        e = errors_usd[k]
        axes[1].hist(e, bins=45, alpha=0.55, label=f"{k} (μ={e.mean():+.1f})")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("USD error")
    axes[1].set_title("Error histograms - look for a non-zero mean")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_walk_forward(
    cv: pd.DataFrame,
    out: Path | str = S.FIGURES_DIR / "06_walk_forward.png",
) -> Path:
    """Per-fold RMSE for every model: exposes instability a single holdout hides."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for model, grp in cv.groupby("model"):
        grp = grp.sort_values("fold")
        ax.plot(grp["fold"], grp["rmse_usd"], marker="o", label=model, lw=1.4)
    ax.set_xlabel("walk-forward fold (later folds = more training data)")
    ax.set_ylabel("RMSE (USD)")
    ax.set_title("Rolling-origin validation")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_feature_importance(
    importance: pd.Series,
    out: Path | str = S.FIGURES_DIR / "07_feature_importance.png",
    top_n: int = 20,
    model_name: str = "model",
) -> Path:
    top = importance.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.index, top.values, color="#2c7fb8")
    ax.set_title(f"Top {min(top_n, len(importance))} features - {model_name}")
    ax.set_xlabel("importance")
    for i, v in enumerate(top.values):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=7)
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
