"""Model zoo: baselines, linear, Random Forest, XGBoost (+ optional LSTM).

All estimators are wrapped so that the caller only sees ``fit(X_train, y_train)``
and ``predict(X)`` on *raw feature frames*. Each wrapper owns its own
preprocessing pipeline (impute + scale fitted on the training slice only), so
leakage between folds is structurally impossible rather than left to discipline.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from configs import settings as S
from src.features import build_preprocess_pipeline, lagged_impute, make_scaler

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Wrappers
# --------------------------------------------------------------------------
class BaseForecaster:
    """Common contract: fit on (X, y) arrays, predict returns, expose levels."""

    name: str = "base"

    def __init__(self, scaler: str = "robust"):
        self.scaler_kind = scaler
        self.pipe_: Pipeline | None = None
        self.feature_names_: List[str] = []

    # -- plumbing ---------------------------------------------------------
    def _build_pipe(self) -> Pipeline:
        return build_preprocess_pipeline(self.scaler_kind)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None,
            X_valid: Optional[np.ndarray] = None, y_valid: Optional[np.ndarray] = None):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def feature_importance(self) -> Optional[pd.Series]:
        """Permutation-free importance when the estimator offers one."""
        return None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.__class__.__name__} name={self.name!r}>"


class LinearModel(BaseForecaster):
    """Ridge on scaled features - the honest benchmark for a noisy series.

    alpha is chosen by RidgeCV over a wide log grid using TimeSeriesSplit, not
    fixed at 1.0: the target is a forward return with a ~0.03 standard
    deviation, so a unit penalty is effectively no penalty and the linear fit
    explodes (R2 near -3.5 on real data).
    """

    name = "ridge"

    ALPHA_GRID = np.logspace(-3, 5, 25)

    def __init__(self, alpha: Optional[float] = None, scaler: str = "standard"):
        super().__init__(scaler)
        self.alpha = alpha

    def fit(self, X, y, feature_names=None, X_valid=None, y_valid=None):
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import TimeSeriesSplit

        self.feature_names_ = list(feature_names) if feature_names else []
        model = (
            Ridge(alpha=self.alpha)
            if self.alpha is not None
            else RidgeCV(alphas=self.ALPHA_GRID, cv=TimeSeriesSplit(n_splits=5))
        )
        self.pipe_ = Pipeline([
            ("impute", lagged_impute()),
            ("scale", make_scaler(self.scaler_kind)),
            ("model", model),
        ])
        self.pipe_.fit(X, y)
        self.alpha_selected_ = getattr(
            self.pipe_.named_steps["model"], "alpha_", self.alpha
        )
        logger.info("ridge alpha selected = %s", self.alpha_selected_)
        return self

    def predict(self, X):
        return self.pipe_.predict(X)

    def feature_importance(self):
        coef = self.pipe_.named_steps["model"].coef_
        return pd.Series(coef, index=self.feature_names_).sort_values(
            key=np.abs, ascending=False
        )


class ForestModel(BaseForecaster):
    """Random Forest: robust to non-linearity, needs no scaling, tolerant of
    the residual NaNs our rolling windows produce."""

    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = S.TRAIN_CFG.rf_n_estimators,
        max_depth: Optional[int] = S.TRAIN_CFG.rf_max_depth,
        min_samples_leaf: int = 25,
        max_features: float = 0.3,
        n_jobs: int = -1,
        random_state: int = S.TRAIN_CFG.random_state,
    ):
        super().__init__(scaler="none")
        self.params: Dict[str, Any] = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, X, y, feature_names=None, X_valid=None, y_valid=None):
        self.feature_names_ = list(feature_names) if feature_names else []
        # impute only: tree splits are scale-invariant
        self.pipe_ = Pipeline([("impute", lagged_impute()),
                               ("model", RandomForestRegressor(**self.params))])
        self.pipe_.fit(X, y)
        self.oob_r2_ = None  # RF would need oob_score=True; kept off for speed
        return self

    def predict(self, X):
        return self.pipe_.predict(X)

    def feature_importance(self):
        imp = self.pipe_.named_steps["model"].feature_importances_
        return pd.Series(imp, index=self.feature_names_).sort_values(ascending=False)


class BoostedModel(BaseForecaster):
    """XGBoost with early stopping on a *chronological* validation slice."""

    name = "xgboost"

    def __init__(
        self,
        n_estimators: int = S.TRAIN_CFG.xgb_n_estimators,
        learning_rate: float = S.TRAIN_CFG.xgb_learning_rate,
        max_depth: int = S.TRAIN_CFG.xgb_max_depth,
        subsample: float = S.TRAIN_CFG.xgb_subsample,
        colsample_bytree: float = S.TRAIN_CFG.xgb_colsample,
        reg_lambda: float = 1.0,
        early_stopping: int = S.TRAIN_CFG.xgb_early_stopping,
        random_state: int = S.TRAIN_CFG.random_state,
    ):
        super().__init__(scaler="none")
        self.params: Dict[str, Any] = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            random_state=random_state,
            tree_method="hist",
            eval_metric="rmse",
            early_stopping_rounds=early_stopping,
            n_jobs=-1,
        )
        self.best_iteration_: Optional[int] = None

    def fit(self, X, y, feature_names=None, X_valid=None, y_valid=None):
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "xgboost is not installed - run `pip install xgboost`"
            ) from exc

        self.feature_names_ = list(feature_names) if feature_names else []
        model = XGBRegressor(**self.params)
        if X_valid is not None and y_valid is not None and len(X_valid) > 50:
            model.fit(X, y, eval_set=[(X_valid, y_valid)], verbose=False)
        else:
            # early stopping needs an eval set; drop the kwarg if absent
            params = {k: v for k, v in self.params.items() if k != "early_stopping_rounds"}
            model.fit(X, y, verbose=False)
        self.best_iteration_ = getattr(model, "best_iteration", None)
        self.model_ = model
        # keep a bare imputer: XGBoost handles NaN natively, scaling is not used
        self.imputer_ = lagged_impute().fit(X)
        return self

    def _prep(self, X):
        return self.imputer_.transform(X)

    def predict(self, X):
        return self.model_.predict(self._prep(X))

    def feature_importance(self):
        imp = self.model_.feature_importances_
        return pd.Series(imp, index=self.feature_names_).sort_values(ascending=False)


# --------------------------------------------------------------------------
# Naive baselines - a model that cannot beat these is not a model
# --------------------------------------------------------------------------
class PersistenceBaseline(BaseForecaster):
    """'Tomorrow equals today': y_hat = 0 forward return (no drift learned)."""

    name = "naive_persistence"

    def fit(self, X, y, feature_names=None, X_valid=None, y_valid=None):
        self.feature_names_ = list(feature_names) if feature_names else []
        return self

    def predict(self, X):
        return np.zeros(len(X))


class DriftBaseline(BaseForecaster):
    """Persistence plus the mean historical move - the TSMOM sanity check."""

    name = "naive_drift"

    def fit(self, X, y, feature_names=None, X_valid=None, y_valid=None):
        self.feature_names_ = list(feature_names) if feature_names else []
        self.mu_ = float(np.nanmean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mu_)


class MomentumBaseline(BaseForecaster):
    """Pure 1-day momentum: predicts the forward move from yesterday's move.

    Many financial ML results are this signal with extra steps, so it is part
    of the benchmark set on purpose.
    """

    name = "naive_momentum"

    def fit(self, X, y, feature_names=None, X_valid=None, y_valid=None):
        self.feature_names_ = list(feature_names) if feature_names else []
        Xf = np.asarray(X, dtype=float)
        self.cols_ = [i for i, c in enumerate(self.feature_names_) if c == "gold_ret_1d"]
        self.beta_ = 0.0
        if self.cols_:
            # numpy has no nancov: mask the pairs by hand, then OLS slope
            col = Xf[:, self.cols_[0]]
            mask = np.isfinite(col) & np.isfinite(y)
            x, yy = col[mask], y[mask]
            if len(x) > 30:
                x = x - x.mean()
                yy = yy - yy.mean()
                denom = float(np.dot(x, x))
                self.beta_ = float(np.dot(x, yy) / denom) if denom > 0 else 0.0
        return self

    def predict(self, X):
        if not getattr(self, "cols_", None):
            return np.zeros(len(X))
        col = np.asarray(X, dtype=float)[:, self.cols_[0]]
        col = np.nan_to_num(col, nan=0.0)
        return self.beta_ * col


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
def get_models(include_baselines: bool = True) -> Dict[str, BaseForecaster]:
    """Instantiate the comparison set used by the training pipeline."""
    models: Dict[str, BaseForecaster] = {}
    if include_baselines:
        models["naive_persistence"] = PersistenceBaseline()
        models["naive_drift"] = DriftBaseline()
        models["naive_momentum"] = MomentumBaseline()
    models["ridge"] = LinearModel(scaler="standard")
    models["random_forest"] = ForestModel()
    try:
        import xgboost  # noqa: F401

        models["xgboost"] = BoostedModel()
    except ImportError:
        logger.warning("xgboost unavailable - comparing RF only")
    return models
