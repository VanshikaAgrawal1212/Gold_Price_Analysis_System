"""LSTM forecaster (optional dependency).

Why this is quarantined in its own module
-----------------------------------------
TensorFlow ships wheels for a narrow set of Python versions, and the deep
branch is not required to run the comparison. Every Keras import happens
*inside* functions, and :func:`tensorflow_available` lets the pipeline skip
the model with a clear message instead of crashing.

Sequence design
---------------
The network sees a sliding window of ``look_back`` days of *scaled returns and
features*, and outputs the single forward return at t + horizon. Using a
window of returns (not raw price levels) is what makes the LSTM generalise
across price regimes: a level-based LSTM memorises "gold is around 1800" and
drifts badly once gold trades at 4000.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from src.models import BaseForecaster

logger = logging.getLogger(__name__)


def tensorflow_available() -> bool:
    """True only if Keras can actually be imported in this interpreter."""
    try:
        import tensorflow as tf  # noqa: F401

        return True
    except Exception:
        return False


def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    look_back: int = 60,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a 2-D tabular matrix into 3-D ``(samples, look_back, features)``.

    The window for sample *i* covers rows ``[i, i+look_back)``, and its label
    is the *last* row's target, so the window never contains its own label.
    """
    n = len(X)
    if n <= look_back:
        raise ValueError(
            f"need more than look_back={look_back} rows, got {n}. "
            "Reduce look_back or extend the date range."
        )
    idx = np.arange(0, n - look_back, stride)
    Xs = np.stack([X[i : i + look_back] for i in idx])
    ys = y[idx + look_back - 1]
    return Xs.astype(np.float32), ys.astype(np.float32)


class LSTMModel(BaseForecaster):
    """Stacked LSTM -> Dense head, trained with early stopping.

    Implements the same ``fit``/``predict`` contract as the tree models so the
    evaluation loop treats it identically. ``predict`` accepts the *2-D tabular*
    matrix and rebuilds windows internally, which keeps callers simple.
    """

    name = "lstm"

    def __init__(
        self,
        look_back: int = 60,
        units: int = 96,
        dropout: float = 0.2,
        l2_reg: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 80,
        patience: int = 12,
        scaler: str = "robust",
    ):
        super().__init__(scaler=scaler)
        self.look_back = look_back
        self.units = units
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.model_ = None
        self.history_ = None

    # -- internals ---------------------------------------------------------
    def _fit_scaler(self, X: np.ndarray):
        """Fit imputer + scaler on the *training rows only*."""
        from sklearn.pipeline import Pipeline

        from src.features import lagged_impute, make_scaler

        self.prep_ = Pipeline([
            ("impute", lagged_impute()),
            ("scale", make_scaler(self.scaler_kind)),
        ])
        self.prep_.fit(X)
        return self.prep_.transform(X)

    def build_network(self, n_features: int):
        from tensorflow import keras

        layers = keras.layers
        reg = keras.regularizers.l2(self.l2_reg) if self.l2_reg else None
        net = keras.Sequential(
            [
                layers.Input(shape=(self.look_back, n_features)),
                layers.LSTM(self.units, return_sequences=True, kernel_regularizer=reg),
                layers.Dropout(self.dropout),
                layers.LSTM(self.units // 2, return_sequences=False, kernel_regularizer=reg),
                layers.Dropout(self.dropout),
                layers.Dense(32, activation="relu", kernel_regularizer=reg),
                layers.Dense(1),  # linear activation: regression on forward return
            ]
        )
        net.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="huber",          # huber damps the pull of fat-tailed days
            metrics=["mae"],
        )
        return net

    # -- contract ----------------------------------------------------------
    def fit(self, X, y, feature_names=None, X_valid=None, y_valid=None):
        if not tensorflow_available():
            raise ImportError(
                "TensorFlow is not installed in this interpreter. Create a "
                "Python 3.11/3.12 virtualenv and `pip install -r "
                "requirements-lstm.txt`, or run without the --lstm flag."
            )
        from tensorflow import keras

        self.feature_names_ = list(feature_names) if feature_names else []
        Xs_scaled = self._fit_scaler(np.asarray(X, dtype=float))

        X_seq, y_seq = make_sequences(Xs_scaled, np.asarray(y, dtype=float), self.look_back)

        if X_valid is not None and y_valid is not None and len(X_valid) > self.look_back + 30:
            # valid features are transformed with the *train* scaler
            V_scaled = self.prep_.transform(np.asarray(X_valid, dtype=float))
            V_seq, v_seq = make_sequences(V_scaled, np.asarray(y_valid, dtype=float),
                                          self.look_back)
            # never shuffle across time: the window is a sequence
            eval_kwargs = {"validation_data": (V_seq, v_seq)}
        else:
            eval_kwargs = {}

        net = self.build_network(X_seq.shape[2])
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if eval_kwargs else "loss",
                mode="min",
                patience=self.patience,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5),
        ]
        self.history_ = net.fit(
            X_seq,
            y_seq,
            epochs=self.epochs,
            batch_size=min(self.batch_size, max(8, len(y_seq) // 20)),
            validation_split=0.0 if eval_kwargs else 0.15,
            callbacks=callbacks,
            verbose=0,
            **eval_kwargs,
        )
        self.model_ = net
        logger.info(
            "LSTM trained on %s windows, best epoch %d",
            X_seq.shape,
            int(np.argmin(self.history_.history["loss"]) + 1),
        )
        return self

    def predict(self, X):
        """Predict forward returns for a tabular block.

        The first ``look_back - 1`` rows have no window yet, so they receive
        the mean of the rest: a defined value rather than NaN, which would
        break the metric computation.
        """
        scaled = self.prep_.transform(np.asarray(X, dtype=float))
        n = len(scaled)
        if n <= self.look_back:
            return np.zeros(n)
        seq = np.stack(
            [scaled[i : i + self.look_back] for i in range(n - self.look_back + 1)]
        )
        preds = self.model_.predict(seq, verbose=0).ravel()
        pad = np.full(self.look_back - 1, float(preds.mean()))
        return np.concatenate([pad, preds])

    def feature_importance(self):
        """Sequence models have no native importance; use permutation."""
        return None


def permutation_importance_lstm(
    model: LSTMModel,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 5,
    seed: int = 0,
) -> "pd.Series":
    """Drop-column importance: re-predict with each feature shuffled.

    Shuffling is done *within* the sequence (per column, per time step) so the
    temporal structure the LSTM relies on is not destroyed wholesale.
    """
    import pandas as pd
    from sklearn.metrics import mean_squared_error

    rng = np.random.default_rng(seed)
    scaled = model.prep_.transform(np.asarray(X, dtype=float))
    base = mean_squared_error(y, model.predict(X), squared=False)

    scores = {}
    for j, name in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            perturbed = scaled.copy()
            col = perturbed[:, j]
            finite = np.isfinite(col)
            col[finite] = rng.permutation(col[finite])
            perturbed[:, j] = col
            # predict() re-applies the scaler, so feed it back un-scaled
            inv = perturbed
            deltas.append(mean_squared_error(y, _predict_scaled(model, inv), squared=False))
        scores[name] = float(np.mean(deltas)) - base
    return pd.Series(scores).sort_values(ascending=False)


def _predict_scaled(model: LSTMModel, scaled: np.ndarray) -> np.ndarray:
    """Predict from an already-scaled matrix (bypasses the prep pipeline)."""
    n = len(scaled)
    seq = np.stack([scaled[i : i + model.look_back] for i in range(n - model.look_back + 1)])
    preds = model.model_.predict(seq, verbose=0).ravel()
    pad = np.full(model.look_back - 1, float(preds.mean()))
    return np.concatenate([pad, preds])

