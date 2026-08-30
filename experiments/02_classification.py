#!/usr/bin/env python3
"""Table 2 — embedded model families on physical labels.

Protocol
--------
* Target is the 3-class perception head (none / flood / landslide).
  Physical multi-hazard maps to flood so the CNN stays at 1,395 parameters.
* Metrics are binary disaster recall/precision (any non-zero vs none).
* Temporal per-station split; scaler fit on train hours only.
* Windows never cross a station or the train/test cut.
* Poisson P_risk is a sample weight only (regime ``poisson``). It is never y.
* The 1D-CNN row is a real Keras Conv1D, not an sklearn MLP.

Usage (after experiment 01, from repository root):

    python experiments/02_classification.py
    python experiments/02_classification.py --smoke
    python experiments/02_classification.py --skip-cnn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ews_ai.models import (  # noqa: E402
    MODEL_FAMILIES,
    build_windowed_split,
    class_recall,
    count_sklearn_params,
    hazard_metrics,
    make_sklearn_estimator,
    map_perception_label,
    poisson_sample_weight,
    predict_cnn1d,
    temporal_per_station_indices,
    temporal_val_indices,
    train_cnn1d,
)
from ews_ai.paths import PROCESSED_DIR, RESULTS_DIR, load_experiment_config  # noqa: E402

SKLEARN_FAMILIES = ("gaussian_nb", "cart", "softmax", "mlp")


def _load_table(path: Path, features: list[str], max_stations: int | None) -> pd.DataFrame:
    usecols = ["station_id", "datetime", "risk_label", "poisson_p_risk", *features]
    df = pd.read_csv(path, usecols=usecols)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", *features, "risk_label"]).sort_values(
        ["station_id", "datetime"]
    )
    if max_stations is not None:
        keep = sorted(df["station_id"].unique())[: int(max_stations)]
        df = df[df["station_id"].isin(keep)]
    return df.reset_index(drop=True)


def _pack_row(family: str, raw: dict, poi: dict, n_params: int) -> dict[str, object]:
    rec_raw = 100.0 * float(raw["hazard_recall"])
    rec_poi = 100.0 * float(poi["hazard_recall"])
    return {
        "model_family": family,
        "rec_raw_pct": round(rec_raw, 2),
        "rec_poi_pct": round(rec_poi, 2),
        "gain_pp": round(rec_poi - rec_raw, 2),
        "prec_poi_pct": round(100.0 * float(poi["hazard_precision"]), 2),
        "prec_raw_pct": round(100.0 * float(raw["hazard_precision"]), 2),
        "n_params": int(n_params),
        "flash_bytes_f32": int(n_params) * 4,
        "n_test": int(raw["n_test"]),
        "n_test_hazard": int(raw["n_test_hazard"]),
        "n_pred_hazard_raw": int(raw["n_pred_hazard"]),
        "n_pred_hazard_poi": int(poi["n_pred_hazard"]),
    }


def _fit_sklearn(family: str, cfg: dict, seed: int, x_tr, y_tr, w_tr, x_te):
    est = make_sklearn_estimator(family, cfg, seed)
    est.fit(x_tr, y_tr, sample_weight=w_tr)
    pred = np.asarray(est.predict(x_te), dtype=np.int32)
    return pred, count_sklearn_params(family, est)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="EWS-AI Table 2 classification.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument(
        "--max-stations",
        type=int,
        default=None,
        help="Use only the first N sorted station ids (smoke / debug).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="8 stations, short CNN/MLP budget. Does not write manuscript tables.",
    )
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="Subset of: gaussian_nb cart softmax mlp cnn1d",
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    cls_cfg = dict(cfg.get("classification") or {})
    features = list(cfg["features"])
    window = int(cfg.get("window_size", 6))
    seed = int(cfg.get("seed", 42))
    test_fraction = float(cfg["split"]["test_fraction"])
    val_fraction = float(cls_cfg.get("val_fraction", 0.1))
    poi_scale = float(cls_cfg.get("poisson_weight_scale", 9.0))

    max_stations = args.max_stations
    if args.smoke:
        max_stations = max_stations or 8
        cls_cfg["epochs"] = min(int(cls_cfg.get("epochs", 12)), 3)
        cls_cfg["mlp_max_iter"] = min(int(cls_cfg.get("mlp_max_iter", 15)), 4)
        cls_cfg["softmax_max_iter"] = min(int(cls_cfg.get("softmax_max_iter", 30)), 8)
        cls_cfg["patience"] = 1
        cls_cfg["verbose"] = 2

    dataset_path = args.dataset or (PROCESSED_DIR / "hourly_dataset.csv")
    if not dataset_path.is_file():
        print(f"Missing {dataset_path}. Run experiments/01_build_dataset.py first.")
        return 2

    print(f"Loading {dataset_path}")
    df = _load_table(dataset_path, features, max_stations)
    print(f"Rows {len(df):,}  stations {df['station_id'].nunique()}")

    x_all = df[features].to_numpy(dtype=np.float64)
    y_all = map_perception_label(df["risk_label"].to_numpy())
    p_all = df["poisson_p_risk"].to_numpy(dtype=np.float64)
    stations = df["station_id"].to_numpy()

    train_idx, test_idx = temporal_per_station_indices(stations, test_fraction)
    scaler = StandardScaler()
    scaler.fit(x_all[train_idx])
    x_scaled = scaler.transform(x_all)

    w_raw = np.ones(len(df), dtype=np.float64)
    w_poi = poisson_sample_weight(p_all, poi_scale)

    def _slice(idx: np.ndarray, weights: np.ndarray):
        return build_windowed_split(
            x_scaled[idx],
            y_all[idx],
            weights[idx],
            stations[idx],
            window,
        )

    print("Building per-station windows (no cross-station / no split leak)...")
    _, y_tr, wr_tr, xt_tr = _slice(train_idx, w_raw)
    xw_te, y_te, _, xt_te = _slice(test_idx, w_raw)
    _, y_tr_poi, wp_tr, _ = _slice(train_idx, w_poi)
    if not np.array_equal(y_tr, y_tr_poi):
        raise RuntimeError("Window alignment mismatch between raw and Poisson splits.")

    fit_local, val_local = temporal_val_indices(stations[train_idx], val_fraction)
    # val_local indexes the hourly train rows. Rebuild windows on the fit/val
    # hourly subsets so the CNN validation set stays temporal.
    xw_fit_raw, y_fit, wr_fit, _ = _slice(train_idx[fit_local], w_raw)
    if len(val_local):
        xw_val, y_val, _, _ = _slice(train_idx[val_local], w_raw)
    else:
        xw_val = np.empty((0, window, len(features)), dtype=np.float64)
        y_val = np.empty(0, dtype=np.int32)
    xw_fit_poi, _, wp_fit, _ = _slice(train_idx[fit_local], w_poi)

    print(
        f"Train windows {len(y_tr):,}  test windows {len(y_te):,}  "
        f"test hazard {int((y_te > 0).sum()):,}  "
        f"({100.0 * float((y_te > 0).mean()):.2f}%)"
    )
    print(
        f"CNN fit windows {len(y_fit):,}  val windows {len(y_val):,}  "
        f"(tabular models use the full train windows)"
    )

    wanted = tuple(args.families) if args.families else MODEL_FAMILIES
    for name in wanted:
        if name not in MODEL_FAMILIES:
            raise SystemExit(f"Unknown family {name!r}. Choose from {MODEL_FAMILIES}")

    raw_metrics: dict[str, dict] = {}
    poi_metrics: dict[str, dict] = {}
    params: dict[str, int] = {}
    extra: dict[str, dict] = {}

    np.random.seed(seed)

    for family in SKLEARN_FAMILIES:
        if family not in wanted:
            continue
        print(f"\n=== {family}  regime=raw ===")
        pred_raw, n_params = _fit_sklearn(family, cls_cfg, seed, xt_tr, y_tr, wr_tr, xt_te)
        raw_metrics[family] = {**hazard_metrics(y_te, pred_raw), **class_recall(y_te, pred_raw)}
        print(
            f"  hazard recall {100*raw_metrics[family]['hazard_recall']:.2f}%  "
            f"precision {100*raw_metrics[family]['hazard_precision']:.2f}%  "
            f"pred+ {raw_metrics[family]['n_pred_hazard']}"
        )

        print(f"=== {family}  regime=poisson ===")
        pred_poi, _ = _fit_sklearn(family, cls_cfg, seed, xt_tr, y_tr, wp_tr, xt_te)
        poi_metrics[family] = {**hazard_metrics(y_te, pred_poi), **class_recall(y_te, pred_poi)}
        params[family] = n_params
        extra[family] = {"n_params": n_params}
        print(
            f"  hazard recall {100*poi_metrics[family]['hazard_recall']:.2f}%  "
            f"precision {100*poi_metrics[family]['hazard_precision']:.2f}%  "
            f"pred+ {poi_metrics[family]['n_pred_hazard']}"
        )

    if "cnn1d" in wanted and not args.skip_cnn:
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            print("\nTensorFlow is not installed. CNN row skipped.")
            print('Install with:  pip install "tensorflow>=2.13"')
            wanted = tuple(f for f in wanted if f != "cnn1d")
        else:
            cnn_cfg = {
                "epochs": int(cls_cfg.get("epochs", 12)),
                "batch_size": int(cls_cfg.get("batch_size", 256)),
                "dropout": float(cls_cfg.get("dropout", 0.2)),
                "patience": int(cls_cfg.get("patience", 3)),
                "verbose": int(cls_cfg.get("verbose", 2)),
            }
            print("\n=== cnn1d  regime=raw ===")
            model_raw = train_cnn1d(
                xw_fit_raw, y_fit, wr_fit, xw_val, y_val, cnn_cfg, seed
            )
            pred_raw = predict_cnn1d(model_raw, xw_te)
            raw_metrics["cnn1d"] = {
                **hazard_metrics(y_te, pred_raw),
                **class_recall(y_te, pred_raw),
            }
            params["cnn1d"] = int(model_raw.count_params())
            print(
                f"  params {params['cnn1d']}  "
                f"hazard recall {100*raw_metrics['cnn1d']['hazard_recall']:.2f}%  "
                f"precision {100*raw_metrics['cnn1d']['hazard_precision']:.2f}%"
            )

            print("=== cnn1d  regime=poisson ===")
            model_poi = train_cnn1d(
                xw_fit_poi, y_fit, wp_fit, xw_val, y_val, cnn_cfg, seed
            )
            pred_poi = predict_cnn1d(model_poi, xw_te)
            poi_metrics["cnn1d"] = {
                **hazard_metrics(y_te, pred_poi),
                **class_recall(y_te, pred_poi),
            }
            extra["cnn1d"] = {"n_params": params["cnn1d"]}
            print(
                f"  params {params['cnn1d']}  "
                f"hazard recall {100*poi_metrics['cnn1d']['hazard_recall']:.2f}%  "
                f"precision {100*poi_metrics['cnn1d']['hazard_precision']:.2f}%"
            )

    rows = [
        _pack_row(family, raw_metrics[family], poi_metrics[family], params[family])
        for family in wanted
        if family in raw_metrics and family in poi_metrics
    ]
    table = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / ("table2_classification_smoke.csv" if args.smoke else "table2_classification.csv")
    out_json = RESULTS_DIR / ("table2_classification_smoke.json" if args.smoke else "table2_classification.json")
    scaler_json = RESULTS_DIR / "scaler_train.json"

    table.to_csv(out_csv, index=False)
    payload = {
        "protocol": {
            "split": "temporal_per_station",
            "test_fraction": test_fraction,
            "window_size": window,
            "features": features,
            "y": (
                "3-class perception (0 none, 1 flood, 2 landslide). "
                "Physical multi-hazard maps to flood. Metrics are binary disaster."
            ),
            "poisson": (
                f"sample_weight = 1 + {poi_scale} * poisson_p_risk; never used as y"
            ),
            "scaler": "StandardScaler fit on train hours only",
            "n_rows_used": int(len(df)),
            "n_stations": int(df["station_id"].nunique()),
            "n_train_windows": int(len(y_tr)),
            "n_test_windows": int(len(y_te)),
            "n_test_hazard": int((y_te > 0).sum()),
            "smoke": bool(args.smoke),
            "cnn_is_keras_conv1d": True,
        },
        "scaler": {
            "features": features,
            "mean": [float(v) for v in scaler.mean_],
            "scale": [float(v) for v in scaler.scale_],
        },
        "by_family": {
            family: {
                "raw": raw_metrics[family],
                "poisson": poi_metrics[family],
                **extra.get(family, {}),
            }
            for family in wanted
            if family in raw_metrics
        },
        "table": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    scaler_json.write_text(json.dumps(payload["scaler"], indent=2), encoding="utf-8")

    print()
    print(table.to_string(index=False))
    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_json}")
    if args.smoke:
        print("Smoke run: manuscript Table 2 was not overwritten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
