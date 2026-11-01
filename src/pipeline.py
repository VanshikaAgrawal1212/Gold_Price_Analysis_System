"""End-to-end orchestration: raw feed -> trained artefacts -> metrics report.

Run as a module from the project root:

    py -m src.pipeline                  # live Yahoo/FRED data, all models
    py -m src.pipeline --synthetic      # offline demo data
    py -m src.pipeline --lstm           # add the Keras LSTM (needs TF)
    py -m src.pipeline --horizon 10     # predict 10 business days ahead

The module also exposes :func:`predict_next` for the serving layer, so the
Streamlit UI and the FastAPI endpoint share exactly one implementation of
"what does a forecast mean".
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from configs import settings as S
from src import evaluate as ev
from src.data_loader import build_dataset, synthetic_dataset
from src.eda import clean
from src.features import build_features, time_series_split, to_xy, walk_forward_folds
from src.models import BaseForecaster, get_models

logger = logging.getLogger(__name__)

ART_MODEL = S.ARTIFACTS_DIR / "model_bundle.joblib"
ART_META = S.ARTIFACTS_DIR / "model_meta.json"
ART_METRICS = S.ARTIFACTS_DIR / "metrics_test.csv"
ART_CV = S.ARTIFACTS_DIR / "metrics_walkforward.csv"
ART_IMPORTANCE = S.ARTIFACTS_DIR / "feature_importance.csv"


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------
def prepare_data(
    synthetic: bool = False,
    horizon: int = S.HORIZON_DAYS,
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load -> clean -> engineer features. Returns (labelled frame, features)."""
    if synthetic:
        logger.warning("using SYNTHETIC data - metrics are meaningless, "
                       "use only to smoke-test the code path")
        raw = synthetic_dataset()
    else:
        raw = build_dataset(use_cache=use_cache)

    tidy = clean(raw)
    data, feature_cols = build_features(tidy, horizon=horizon)
    return data, feature_cols


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_and_evaluate(
    data: pd.DataFrame,
    feature_cols: List[str],
    include_baselines: bool = True,
    extra_models: Optional[Dict[str, BaseForecaster]] = None,
) -> dict:
    """Fit every model on train, early-stop on valid, score once on test."""
    split = time_series_split(data, feature_cols)
    logger.info("split -> %s", split.summary())

    X_tr, y_tr, lvl_tr = to_xy(split.train, feature_cols)
    X_va, y_va, _ = to_xy(split.valid, feature_cols)
    X_te, y_te, lvl_te = to_xy(split.test, feature_cols)

    models = get_models(include_baselines=include_baselines)
    if extra_models:
        models.update(extra_models)

    results: List[ev.MetricSet] = []
    preds_usd: Dict[str, np.ndarray] = {}
    errors_usd: Dict[str, np.ndarray] = {}
    fitted: Dict[str, BaseForecaster] = {}
    importances: Dict[str, pd.Series] = {}

    # reference prediction for the skill score: naive persistence (return = 0)
    naive_ret_test = np.zeros(len(y_te))

    for name, model in models.items():
        t0 = time.time()
        try:
            model.fit(X_tr, y_tr, feature_names=feature_cols,
                      X_valid=X_va, y_valid=y_va)
            pred_ret = model.predict(X_te)
        except Exception as exc:
            logger.error("model %s failed: %s", name, exc)
            continue

        fitted[name] = model
        metrics = ev.evaluate_predictions(
            y_te, pred_ret, lvl_te, model_name=name, split="test",
            naive_pred_ret=naive_ret_test,
        )
        results.append(metrics)

        pred_price = lvl_te * (1 + pred_ret)
        preds_usd[name] = pred_price
        errors_usd[name] = pred_price - lvl_te * (1 + y_te)
        elapsed = time.time() - t0
        logger.info(
            "%-18s RMSE=%.5f MAE=%.5f R2=%+.4f hit=%.1f%% skill=%+.3f (%.1fs)",
            name, metrics.rmse_ret, metrics.mae_ret, metrics.r2_ret,
            metrics.hit_rate * 100,
            metrics.skill_vs_naive if metrics.skill_vs_naive is not None else float("nan"),
            elapsed,
        )
        imp = model.feature_importance()
        if imp is not None:
            importances[name] = imp

    if not results:
        raise RuntimeError("no model trained successfully")

    table = ev.metrics_table(results)
    best_name = table.reset_index().sort_values(
        "skill_vs_naive", ascending=False
    ).iloc[0]["model"]
    # realised outcome is kept separate so it is drawn once, as the black line
    realised = lvl_te * (1 + y_te)

    return {
        "split": split,
        "fitted": fitted,
        "results": results,
        "table": table,
        "best_model": best_name,
        "preds_usd": preds_usd,
        "errors_usd": errors_usd,
        "importances": importances,
        "test_dates": split.test.index,
        "test_actual_usd": realised,
        "test_entry_usd": lvl_te,
    }


def cross_validate(
    data: pd.DataFrame,
    feature_cols: List[str],
    model_factory,
    n_splits: int = S.TRAIN_CFG.n_splits_cv,
) -> pd.DataFrame:
    """Rolling-origin CV for one fresh instance per fold (no state reuse)."""
    rows = []
    X_all, y_all, lvl_all = to_xy(data, feature_cols)
    for fold, (tr_idx, te_idx) in enumerate(walk_forward_folds(data, n_splits), start=1):
        model = model_factory()
        model.fit(
            X_all[tr_idx], y_all[tr_idx],
            feature_names=feature_cols,
            X_valid=X_all[tr_idx][-min(300, len(tr_idx) // 5):],
            y_valid=y_all[tr_idx][-min(300, len(tr_idx) // 5):],
        )
        pred = model.predict(X_all[te_idx])
        entry = lvl_all[te_idx]
        actual = entry * (1 + y_all[te_idx])
        naive = entry  # persistence in price space
        err_usd = float(np.sqrt(np.mean((entry * (1 + pred) - actual) ** 2)))
        naive_usd = float(np.sqrt(np.mean((naive - actual) ** 2)))
        rows.append({
            "model": model.name,
            "fold": fold,
            "n": len(te_idx),
            "start": str(data.index[te_idx[0]].date()),
            "end": str(data.index[te_idx[-1]].date()),
            "rmse_ret": float(np.sqrt(np.mean((pred - y_all[te_idx]) ** 2))),
            "rmse_usd": err_usd,
            "mae_usd": float(np.mean(np.abs(entry * (1 + pred) - actual))),
            "naive_rmse_usd": naive_usd,
            "skill_vs_naive": 1 - err_usd / naive_usd if naive_usd > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Serving helper
# --------------------------------------------------------------------------
def predict_next(
    model: BaseForecaster,
    feature_cols: List[str],
    clean_df: pd.DataFrame,
    horizon: int,
) -> dict:
    """Forecast the next ``horizon`` business days from the latest data row.

    Features are built with ``dropna=False`` so the most recent rows survive
    even though they have no label yet - that is precisely the live case.

    Preference order for the scoring row:
      1. the newest row where every feature is present;
      2. otherwise the newest row that has a price - the model's median
         imputer fills the remaining holes, which is far better than refusing
         to forecast on the one day everyone actually wants a number for.
    """
    data, _ = build_features(clean_df, horizon=horizon, dropna=False)
    complete = data.dropna(subset=feature_cols)
    if not complete.empty:
        row = complete.iloc[-1:]
    else:
        priced = data.dropna(subset=[S.TARGET_COL])
        if priced.empty:
            raise ValueError("no usable feature row; extend the history")
        row = priced.iloc[-1:]
        logger.warning("latest row has incomplete features - median-imputing")
    X = row[feature_cols].to_numpy(dtype=float)
    entry = float(row[S.TARGET_COL].iloc[-1])
    pred_ret = float(np.ravel(model.predict(X))[-1])

    return {
        "as_of_date": str(row.index[-1].date()),
        "entry_price_usd": entry,
        "horizon_days": horizon,
        "target_date": str((row.index[-1] + pd.tseries.offsets.BDay(horizon)).date()),
        "predicted_return": pred_ret,
        "predicted_price_usd": entry * (1 + pred_ret),
    }


# --------------------------------------------------------------------------
# Full run
# --------------------------------------------------------------------------
def run(
    synthetic: bool = False,
    with_lstm: bool = False,
    horizon: int = S.HORIZON_DAYS,
    use_cache: bool = True,
    do_cv: bool = True,
) -> dict:
    """Execute the whole pipeline and persist artefacts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-7s %(name)s | %(message)s",
    )
    logger.info("=== gold price pipeline | horizon=%dd ===", horizon)

    data, feature_cols = prepare_data(synthetic=synthetic, horizon=horizon,
                                      use_cache=use_cache)

    extra: Dict[str, BaseForecaster] = {}
    if with_lstm:
        from src.lstm_model import LSTMModel, tensorflow_available

        if tensorflow_available():
            extra["lstm"] = LSTMModel(look_back=min(60, max(20, len(data) // 40)))
        else:
            logger.warning("TensorFlow not importable - skipping LSTM")

    out = train_and_evaluate(data, feature_cols, extra_models=extra)
    out["feature_cols"] = feature_cols
    out["data"] = data

    # ---- figures ---------------------------------------------------------
    ev.plot_predictions(
        out["test_dates"], out["test_actual_usd"], out["preds_usd"],
        S.FIGURES_DIR / "04_predictions.png",
        title=f"Gold: actual vs. predicted price, {horizon}-day horizon",
    )
    ev.plot_error_distribution(out["errors_usd"], S.FIGURES_DIR / "05_errors.png")

    if out["importances"]:
        best_imp = out["importances"].get(
            out["best_model"], list(out["importances"].values())[0]
        )
        ev.plot_feature_importance(
            best_imp, S.FIGURES_DIR / "07_feature_importance.png",
            model_name=out["best_model"],
        )
        best_imp.rename("importance").to_csv(ART_IMPORTANCE)

    # ---- walk-forward CV on the two headline models -----------------------
    if do_cv:
        cv_frames = []
        for key in ("random_forest", "xgboost"):
            if key not in out["fitted"]:
                continue
            factory = (lambda k=key: get_models(include_baselines=False)[k])
            try:
                cv_frames.append(cross_validate(data, feature_cols, factory))
            except Exception as exc:
                logger.error("walk-forward CV failed for %s: %s", key, exc)
        if cv_frames:
            cv = pd.concat(cv_frames, ignore_index=True)
            cv.to_csv(ART_CV, index=False)
            ev.plot_walk_forward(cv, S.FIGURES_DIR / "06_walk_forward.png")
            out["cv"] = cv

    # ---- reports ---------------------------------------------------------
    out["table"].to_csv(ART_METRICS)
    logger.info("\n%s", out["table"].to_string())

    # ---- persist the winning model ---------------------------------------
    bundle = {
        "model": out["fitted"][out["best_model"]],
        "best_model": out["best_model"],
        "feature_cols": feature_cols,
        "horizon": horizon,
        "target_col": S.TARGET_COL,
        "all_models": out["fitted"],
    }
    joblib.dump(bundle, ART_MODEL)

    meta = {
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "best_model": out["best_model"],
        "horizon_days": horizon,
        "n_features": len(feature_cols),
        "n_rows": int(len(data)),
        "data_source": "synthetic" if synthetic else "yahoo+fred",
        "test_metrics": next(
            r.to_row() for r in out["results"] if r.model == out["best_model"]
        ),
        "artifacts": {
            "model_bundle": str(ART_MODEL),
            "metrics_test": str(ART_METRICS),
            "metrics_walkforward": str(ART_CV),
        },
    }
    ART_META.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    logger.info("best model = %s -> saved %s", out["best_model"], ART_MODEL)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Train the gold price models")
    p.add_argument("--synthetic", action="store_true",
                   help="use generated offline data (code path test only)")
    p.add_argument("--lstm", action="store_true", help="include the Keras LSTM")
    p.add_argument("--horizon", type=int, default=S.HORIZON_DAYS,
                   help="forecast horizon in business days (default %(default)s)")
    p.add_argument("--no-cache", action="store_true", help="force a fresh download")
    p.add_argument("--no-cv", action="store_true", help="skip walk-forward CV")
    args = p.parse_args()

    run(
        synthetic=args.synthetic,
        with_lstm=args.lstm,
        horizon=args.horizon,
        use_cache=not args.no_cache,
        do_cv=not args.no_cv,
    )


if __name__ == "__main__":
    main()
