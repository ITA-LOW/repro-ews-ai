#!/usr/bin/env python3
"""Simulated 10-year autonomy: FSM-LoRa vs BDI, same CART, same hourly wake.

always_on_not_fsm is a no-duty-cycle strawman. It is not an FSM.

Usage (after experiment 01):

    python experiments/04_energy.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ews_ai.cart_export import FEATURE_ORDER, fit_unscaled_cart, load_xy, predict_cart  # noqa: E402
from ews_ai.energy import (  # noqa: E402
    POLICY_ALWAYS_ON,
    POLICY_BDI,
    POLICY_FSM_LORA,
    simulate_dataset,
)
from ews_ai.paths import PROCESSED_DIR, RESULTS_DIR, load_experiment_config  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Simulated FSM vs BDI energy.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--max-stations", type=int, default=None)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    dataset = args.dataset or (PROCESSED_DIR / "hourly_dataset.csv")
    if not dataset.is_file():
        print(f"Missing {dataset}. Run experiments/01_build_dataset.py first.")
        return 2

    usecols = ["station_id", "datetime", "risk_label", "solar_power_mw", *FEATURE_ORDER]
    # FEATURE_ORDER already includes solar_power_mw
    usecols = list(dict.fromkeys(usecols))
    df = pd.read_csv(dataset, usecols=usecols)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "risk_label", *FEATURE_ORDER]).sort_values(
        ["station_id", "datetime"]
    )
    if args.max_stations is not None:
        keep = sorted(df["station_id"].unique())[: int(args.max_stations)]
        df = df[df["station_id"].isin(keep)].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    x, y, stations = load_xy(df)
    tree, cart_metrics = fit_unscaled_cart(x, y, stations, cfg)
    hazard = predict_cart(tree, x) > 0
    print(
        f"CART detector on all hours: {int(hazard.sum()):,} / {len(hazard):,} "
        f"({100.0 * float(hazard.mean()):.2f}%)  "
        f"(test recall {100*cart_metrics['hazard_recall']:.2f}%)"
    )

    detail, summary = simulate_dataset(
        df,
        hazard,
        cfg,
        policies=(POLICY_FSM_LORA, POLICY_BDI, POLICY_ALWAYS_ON),
    )

    tight_cfg = {
        **cfg,
        "energy": {**cfg["energy"], "lifepo4_capacity_mwh": float(cfg["battery"]["capacity_mwh"])},
    }
    _, tight = simulate_dataset(
        df,
        hazard,
        tight_cfg,
        policies=(POLICY_FSM_LORA, POLICY_BDI, POLICY_ALWAYS_ON),
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = RESULTS_DIR / "energy_uptime_by_station.csv"
    summary_path = RESULTS_DIR / "energy_uptime.csv"
    tight_path = RESULTS_DIR / "energy_uptime_tight_cell.csv"
    json_path = RESULTS_DIR / "energy_uptime.json"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    tight.to_csv(tight_path, index=False)
    payload = {
        "protocol": {
            "same_detector": "unscaled CART (firmware tree)",
            "same_wake": "1 hour (INMET row)",
            "fsm_lora": "hazard → always LoRa; else deep sleep. No SoC plan.",
            "bdi": (
                "hazard + SoC>=low → LoRa; hazard + SoC<low → ESP-NOW "
                "(neighbor assumed present); else deep sleep."
            ),
            "always_on_not_fsm": "Continuous 120 mW load. Not an FSM.",
            "years": cfg["energy"]["simulation_years"],
            "capacity_mwh": cfg["energy"]["lifepo4_capacity_mwh"],
            "tight_capacity_mwh": cfg["battery"]["capacity_mwh"],
            "p_deep_sleep_mw": cfg["energy"]["p_deep_sleep_mw"],
            "p_espnow_mw": cfg["energy"]["p_espnow_mw"],
            "p_lora_mw": cfg["energy"]["p_lora_mw"],
            "p_continuous_tx_mw": cfg["energy"]["p_continuous_tx_mw"],
            "simulated": True,
        },
        "cart_test": cart_metrics,
        "summary": summary.to_dict(orient="records"),
        "tight_cell": tight.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print("Main cell (9620 mWh LiFePO4):")
    print(summary.to_string(index=False))
    print("\nTight cell (1500 mWh, sensitivity — SoC fallback can fire):")
    print(tight.to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {detail_path}")
    print(f"Wrote {tight_path}")
    print(f"Wrote {json_path}")
    print("These are simulations, not field measurements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
