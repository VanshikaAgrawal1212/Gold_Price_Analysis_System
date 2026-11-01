"""Serving layer shared by Streamlit and FastAPI.

Both front-ends call :func:`forecast` so that "the prediction" has exactly one
definition in this repository. Model loading is cached at module level because
an 8 MB joblib bundle should not be re-read on every HTTP request or every
Streamlit widget interaction.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Optional

import joblib
import pandas as pd

from configs import settings as S
from src.data_loader import build_dataset
from src.eda import clean
from src.pipeline import predict_next

logger = logging.getLogger(__name__)

BUNDLE_PATH = S.ARTIFACTS_DIR / "model_bundle.joblib"
META_PATH = S.ARTIFACTS_DIR / "model_meta.json"


class ModelNotTrained(RuntimeError):
    """Raised when the API is hit before ``python -m src.pipeline`` has run."""


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    if not BUNDLE_PATH.exists():
        raise ModelNotTrained(
            f"No model bundle at {BUNDLE_PATH}. Train first: `py -m src.pipeline`"
        )
    return joblib.load(BUNDLE_PATH)


@lru_cache(maxsize=1)
def load_meta() -> dict:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    return {}


@lru_cache(maxsize=1)
def _feature_frame() -> pd.DataFrame:
    """Cleaned price history used for feature look-back (cached, ~1 day TTL via
    ``clear_caches`` in the UI)."""
    return clean(build_dataset(use_cache=True))


def clear_caches() -> None:
    """Drop the memoised data/model objects (after retraining or new data)."""
    load_bundle.cache_clear()
    load_meta.cache_clear()
    _feature_frame.cache_clear()


def forecast(
    model_name: Optional[str] = None,
    horizon: Optional[int] = None,
    refresh_data: bool = False,
) -> dict:
    """Return the next-step gold forecast plus the context to explain it.

    Parameters
    ----------
    model_name : str, optional
        Any model kept in the bundle (``xgboost``, ``random_forest``, ...).
        Defaults to the trained best model.
    horizon : int, optional
        Business days ahead. Overrides the trained horizon *only* for the
        reported date arithmetic; the model itself was trained for its own
        label horizon, so changing it silently would be misleading.
    refresh_data : bool
        Bypass the cached feature frame and pull the latest closes.
    """
    bundle = load_bundle()
    meta = load_meta()
    models = bundle["all_models"]
    name = model_name or bundle.get("best_model") or meta.get("best_model")
    if name is None:
        name = max(models, key=lambda k: _rank_key(k, meta))
    if name not in models:
        raise KeyError(
            f"unknown model {name!r}; available: {sorted(models)}"
        )

    tidy = clean(build_dataset(use_cache=not refresh_data)) if refresh_data else _feature_frame()

    trained_horizon = int(bundle.get("horizon", meta.get("horizon_days", S.HORIZON_DAYS)))
    point = predict_next(
        models[name], bundle["feature_cols"], tidy, trained_horizon
    )
    point["model"] = name
    point["trained_horizon_days"] = trained_horizon
    point["requested_horizon_days"] = horizon or trained_horizon
    point["history"] = _recent_history(tidy)
    point["metrics"] = meta.get("test_metrics", {})
    return point


def _rank_key(name: str, meta: dict) -> float:
    """Sort helper so a fallback model choice is deterministic."""
    return 1.0 if name == meta.get("best_model") else 0.0


def _recent_history(tidy: pd.DataFrame, n: int = 500) -> pd.DataFrame:
    cols = [S.TARGET_COL, "dxy", "us10y", "silver", "usdinr"]
    keep = [c for c in cols if c in tidy.columns]
    return tidy[keep].tail(n).reset_index()


def available_models() -> list[str]:
    return sorted(load_bundle()["all_models"])


def metrics_table() -> Optional[pd.DataFrame]:
    path = S.ARTIFACTS_DIR / "metrics_test.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def walk_forward_table() -> Optional[pd.DataFrame]:
    path = S.ARTIFACTS_DIR / "metrics_walkforward.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)
