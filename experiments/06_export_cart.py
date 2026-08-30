#!/usr/bin/env python3
"""Fit the unscaled on-device CART and write firmware/src/cart_model.h.

Usage (after experiment 01):

    python experiments/06_export_cart.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ews_ai.cart_export import (  # noqa: E402
    DEFAULT_HEADER,
    FEATURE_ORDER,
    fit_unscaled_cart,
    load_xy,
    write_header,
)
from ews_ai.paths import PROCESSED_DIR, RESULTS_DIR, load_experiment_config  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Export unscaled CART to C.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--header", type=Path, default=DEFAULT_HEADER)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    dataset = args.dataset or (PROCESSED_DIR / "hourly_dataset.csv")
    if not dataset.is_file():
        print(f"Missing {dataset}. Run experiments/01_build_dataset.py first.")
        return 2

    usecols = ["station_id", "datetime", "risk_label", *FEATURE_ORDER]
    df = pd.read_csv(dataset, usecols=usecols)
    df = df.dropna(subset=["risk_label", *FEATURE_ORDER]).sort_values(
        ["station_id", "datetime"]
    )
    x, y, stations = load_xy(df)
    tree, metrics = fit_unscaled_cart(x, y, stations, cfg)
    dest = write_header(tree, args.header)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": list(FEATURE_ORDER),
        "scaled": False,
        "header": str(dest.relative_to(ROOT)) if dest.is_relative_to(ROOT) else str(dest),
        "metrics": metrics,
        "flash_bytes_est": int(metrics["n_nodes"]) * 12,
        "note": (
            "On-device perception artifact. Thresholds are in physical units. "
            "Table 2 CART used a scaler; this tree is the firmware model."
        ),
    }
    out_json = RESULTS_DIR / "cart_export.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"CART nodes {metrics['n_nodes']}  depth {metrics['max_depth']}  "
        f"test hazard recall {100*metrics['hazard_recall']:.2f}%  "
        f"precision {100*metrics['hazard_precision']:.2f}%"
    )
    print(f"Wrote {dest}")
    print(f"Wrote {out_json}")
    print(f"Estimated tree flash {payload['flash_bytes_est']} bytes (n_nodes * 12)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
