"""Table 2 model families, temporal split, and disaster-class metrics.

The perception head is the manuscript 1D-CNN (1,395 trainable parameters):
Conv1D(16, k=2) → MaxPool(2) → Flatten → Dense(32) → Dropout(0.2) → Softmax(3).

Physical ``risk_label`` is 4-way (none / landslide / flood / multi). The
artifact emits 3 logits, so multi-hazard hours are mapped to flood. Binary
disaster recall still counts those hours as positives.

Poisson ``p_risk`` may enter only as a sample weight. It is never ``y``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

MODEL_FAMILIES = (
    "gaussian_nb",
    "cart",
    "softmax",
    "mlp",
    "cnn1d",
)

# Manuscript perception classes. Multi-hazard → flood so Dense(3) stays 1,395 params.
PERCEPTION_NONE = 0
PERCEPTION_FLOOD = 1
PERCEPTION_LANDSLIDE = 2
N_PERCEPTION_CLASSES = 3

# Physical risk_label from dataset.py
PHYS_NONE = 0
PHYS_LANDSLIDE = 1
PHYS_FLOOD = 2
PHYS_MULTI = 3


def map_perception_label(risk_label: np.ndarray) -> np.ndarray:
    """4-class physical label → 3-class perception target."""
    y = np.asarray(risk_label, dtype=np.int32)
    out = np.zeros(y.shape, dtype=np.int32)
    out[(y == PHYS_FLOOD) | (y == PHYS_MULTI)] = PERCEPTION_FLOOD
    out[y == PHYS_LANDSLIDE] = PERCEPTION_LANDSLIDE
    return out


def poisson_sample_weight(p_risk: np.ndarray, scale: float) -> np.ndarray:
    """w = 1 + scale * P_risk. Quiet hours stay at 1. Never used as y."""
    p = np.asarray(p_risk, dtype=np.float64)
    return 1.0 + float(scale) * np.clip(p, 0.0, 1.0)


def temporal_per_station_indices(
    station_ids: np.ndarray,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Last ``test_fraction`` of each station is test. Rows must be time-sorted."""
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for sid in np.unique(station_ids):
        idx = np.flatnonzero(station_ids == sid)
        n = int(len(idx))
        n_test = max(1, int(round(n * float(test_fraction))))
        n_train = n - n_test
        if n_train < 1:
            train_parts.append(idx)
            continue
        train_parts.append(idx[:n_train])
        test_parts.append(idx[n_train:])
    return np.concatenate(train_parts), np.concatenate(test_parts)


def temporal_val_indices(
    station_ids: np.ndarray,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split an already-selected train index list into fit / early-stop val."""
    fit_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    n_local = np.arange(len(station_ids))
    for sid in np.unique(station_ids):
        idx = n_local[station_ids == sid]
        n = int(len(idx))
        n_val = max(1, int(round(n * float(val_fraction)))) if n > 1 else 0
        n_fit = n - n_val
        if n_fit < 1:
            fit_parts.append(idx)
            continue
        fit_parts.append(idx[:n_fit])
        if n_val:
            val_parts.append(idx[n_fit:])
    val = np.concatenate(val_parts) if val_parts else np.empty(0, dtype=np.int64)
    return np.concatenate(fit_parts), val


def _windows_one_station(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Windows stay inside one station. Label is the last hour of the window."""
    n, n_feat = x.shape
    if n < window:
        return None
    x = np.ascontiguousarray(x)
    views = np.lib.stride_tricks.sliding_window_view(x, window_shape=window, axis=0)
    # NumPy 2 with axis=0 yields (n-w+1, n_feat, window). Keras Conv1D wants
    # (n-w+1, window, n_feat). Accept either layout.
    if views.shape[1] == n_feat and views.shape[-1] == window:
        views = np.transpose(views, (0, 2, 1))
    if views.shape[1:] != (window, n_feat):
        raise RuntimeError(f"Window tensor has shape {views.shape}, expected (*, {window}, {n_feat})")
    xw = np.array(views, copy=True)
    sl = slice(window - 1, None)
    return xw, y[sl].copy(), weights[sl].copy(), x[sl].copy()


def build_windowed_split(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    station_ids: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-station 6-hour tensors plus the aligned last-hour tabular rows."""
    x_win: list[np.ndarray] = []
    y_out: list[np.ndarray] = []
    w_out: list[np.ndarray] = []
    x_tab: list[np.ndarray] = []
    for sid in np.unique(station_ids):
        mask = station_ids == sid
        built = _windows_one_station(x[mask], y[mask], weights[mask], window)
        if built is None:
            continue
        xw, yw, ww, xt = built
        x_win.append(xw)
        y_out.append(yw)
        w_out.append(ww)
        x_tab.append(xt)
    if not x_win:
        raise RuntimeError("No station had enough hours for the sliding window.")
    return (
        np.concatenate(x_win, axis=0),
        np.concatenate(y_out, axis=0),
        np.concatenate(w_out, axis=0),
        np.concatenate(x_tab, axis=0),
    )


def hazard_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Binary disaster metrics: any non-zero class vs none."""
    yt = np.asarray(y_true) > 0
    yp = np.asarray(y_pred) > 0
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp))
    tn = int(np.sum(~yt & ~yp))
    rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "hazard_recall": rec,
        "hazard_precision": prec,
        "hazard_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_test": int(len(yt)),
        "n_test_hazard": int(np.sum(yt)),
        "n_pred_hazard": int(np.sum(yp)),
    }


def class_recall(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> dict[str, float]:
    out: dict[str, float] = {}
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    for c in range(n_classes):
        mask = yt == c
        out[f"recall_class_{c}"] = float(np.mean(yp[mask] == c)) if mask.any() else 0.0
    return out


def make_sklearn_estimator(family: str, cfg: dict[str, Any], seed: int):
    if family == "gaussian_nb":
        return GaussianNB()
    if family == "cart":
        return DecisionTreeClassifier(
            max_depth=int(cfg.get("cart_max_depth", 10)),
            min_samples_leaf=int(cfg.get("cart_min_samples_leaf", 50)),
            random_state=seed,
        )
    if family == "softmax":
        return LogisticRegression(
            solver="saga",
            max_iter=int(cfg.get("softmax_max_iter", 30)),
            tol=1e-3,
            n_jobs=-1,
            random_state=seed,
        )
    if family == "mlp":
        hidden = tuple(int(h) for h in cfg.get("mlp_hidden", [32]))
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            batch_size=int(cfg.get("mlp_batch_size", 256)),
            max_iter=int(cfg.get("mlp_max_iter", 15)),
            random_state=seed,
            early_stopping=False,
        )
    raise ValueError(f"Unknown sklearn family: {family}")


def count_sklearn_params(family: str, estimator) -> int:
    if family == "gaussian_nb":
        return int(estimator.theta_.size + estimator.var_.size + estimator.class_prior_.size)
    if family == "cart":
        return int(estimator.tree_.node_count)
    if family == "softmax":
        n = int(np.prod(estimator.coef_.shape))
        return n + int(np.size(estimator.intercept_))
    if family == "mlp":
        n = 0
        for coef, intercept in zip(estimator.coefs_, estimator.intercepts_):
            n += int(np.prod(coef.shape) + np.size(intercept))
        return n
    raise ValueError(family)


def build_cnn1d(n_features: int, window: int, dropout: float, n_classes: int = 3):
    """Manuscript TinyML head. Trainable count is 1,395 when n_classes=3, n_features=7, window=6."""
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window, n_features)),
            tf.keras.layers.Conv1D(filters=16, kernel_size=2, activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(float(dropout)),
            tf.keras.layers.Dense(int(n_classes), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_cnn1d(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(int(seed))
    model = build_cnn1d(
        n_features=int(x_train.shape[2]),
        window=int(x_train.shape[1]),
        dropout=float(cfg.get("dropout", 0.2)),
        n_classes=N_PERCEPTION_CLASSES,
    )
    callbacks = []
    patience = int(cfg.get("patience", 3))
    if len(x_val) and patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        )
    fit_kw: dict[str, Any] = {
        "epochs": int(cfg.get("epochs", 12)),
        "batch_size": int(cfg.get("batch_size", 256)),
        "sample_weight": w_train,
        "verbose": int(cfg.get("verbose", 2)),
        "callbacks": callbacks,
    }
    if len(x_val):
        fit_kw["validation_data"] = (x_val, y_val)
    model.fit(x_train, y_train, **fit_kw)
    return model


def predict_cnn1d(model, x: np.ndarray) -> np.ndarray:
    proba = model.predict(x, batch_size=1024, verbose=0)
    return np.argmax(proba, axis=1).astype(np.int32)
