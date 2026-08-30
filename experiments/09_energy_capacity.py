#!/usr/bin/env python3
"""Capacity sweep: FSM-LoRa vs BDI vs BDI-thrifty. Same CART, same hourly wake.

Not an M1–M4 ablation. Always-on is not in this table.

    python experiments/09_energy_capacity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ews_ai.cart_export import FEATURE_ORDER, fit_unscaled_cart, load_xy, predict_cart  # noqa: E402
from ews_ai.energy import (  # noqa: E402
    POLICY_BDI,
    POLICY_BDI_THRIFTY,
    POLICY_FSM_LORA,
    simulate_dataset,
)
from ews_ai.paths import PROCESSED_DIR, RESULTS_DIR, load_experiment_config  # noqa: E402

DEFAULT_CAPS = (9620.0, 1500.0, 500.0, 200.0)

# (policy, assume_neighbor, label)
VARIANTS = (
    (POLICY_FSM_LORA, True, "FSM-LoRa"),
    (POLICY_BDI, True, "BDI (ESP-NOW if neighbor)"),
    (POLICY_BDI_THRIFTY, True, "BDI-thrifty (neighbor)"),
    (POLICY_BDI_THRIFTY, False, "BDI-thrifty (no neighbor)"),
)

COLORS = {
    "FSM-LoRa": "#4C72B0",
    "BDI (ESP-NOW if neighbor)": "#55A868",
    "BDI-thrifty (neighbor)": "#C44E52",
    "BDI-thrifty (no neighbor)": "#8172B3",
}


def plot_sweep(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for label in [v[2] for v in VARIANTS]:
        sub = summary[summary["label"] == label].sort_values("capacity_mwh")
        if sub.empty:
            continue
        axes[0].plot(
            sub["capacity_mwh"],
            sub["mean_uptime_pct"],
            marker="o",
            color=COLORS[label],
            label=label,
        )
        axes[1].plot(
            sub["capacity_mwh"],
            sub["mean_missed_hazard_while_dead"],
            marker="o",
            color=COLORS[label],
            label=label,
        )
    axes[0].set_xlabel("Cell capacity (mWh)")
    axes[0].set_ylabel("Mean 10y uptime (%)")
    axes[0].set_title("Survival")
    axes[1].set_xlabel("Cell capacity (mWh)")
    axes[1].set_ylabel("Mean missed hazards while dead")
    axes[1].set_title("Service lost while dead")
    axes[0].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(sorted(summary["capacity_mwh"].unique()))
        ax.set_xticklabels([str(int(c)) for c in sorted(summary["capacity_mwh"].unique())])
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        "Simulated FSM vs BDI — same CART, same hourly wake (datasheet mW)",
        y=1.02,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="FSM vs BDI capacity sweep.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--max-stations", type=int, default=None)
    parser.add_argument(
        "--capacities",
        type=float,
        nargs="+",
        default=None,
        help="Override capacity_sweep_mwh (mWh).",
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    dataset = args.dataset or (PROCESSED_DIR / "hourly_dataset.csv")
    if not dataset.is_file():
        print(f"Missing {dataset}. Run experiments/01_build_dataset.py first.")
        return 2

    caps = args.capacities or cfg["energy"].get("capacity_sweep_mwh", DEFAULT_CAPS)
    caps = [float(c) for c in caps]

    usecols = ["station_id", "datetime", "risk_label", "solar_power_mw", *FEATURE_ORDER]
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
    print(f"Capacities (mWh): {caps}")

    summaries = []
    details = []
    for cap in caps:
        run_cfg = {
            **cfg,
            "energy": {**cfg["energy"], "lifepo4_capacity_mwh": cap},
        }
        for policy, neighbor, label in VARIANTS:
            print(f"\n--- {label} @ {cap:.0f} mWh ---")
            detail, summary = simulate_dataset(
                df,
                hazard,
                run_cfg,
                policies=(policy,),
                assume_neighbor=neighbor,
            )
            summary = summary.copy()
            summary["label"] = label
            detail = detail.copy()
            detail["label"] = label
            summaries.append(summary)
            details.append(detail)
            print(summary.to_string(index=False))

    summary = pd.concat(summaries, ignore_index=True)
    detail = pd.concat(details, ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sum_path = RESULTS_DIR / "energy_capacity_sweep.csv"
    det_path = RESULTS_DIR / "energy_capacity_sweep_by_station.csv"
    json_path = RESULTS_DIR / "energy_capacity_sweep.json"
    fig_path = fig_dir / "energy_capacity_sweep.png"
    summary.to_csv(sum_path, index=False)
    detail.to_csv(det_path, index=False)
    plot_sweep(summary, fig_path)
    ms_img = ROOT.parent / "artigo_2_ews-ai" / "manuscrito" / "img"
    if ms_img.is_dir():
        plot_sweep(summary, ms_img / "energy_capacity_sweep.png")
    payload = {
        "protocol": {
            "same_detector": "unscaled CART (firmware tree)",
            "same_wake": "1 hour (INMET row)",
            "fsm_lora": "hazard → always LoRa; else deep sleep. No SoC plan.",
            "bdi": (
                "hazard + SoC>=low → LoRa; hazard + SoC<low → ESP-NOW "
                "if a neighbor is assumed; else LoRa (collapses to FSM)."
            ),
            "bdi_thrifty": (
                "hazard + SoC>=low → LoRa; hazard + SoC<low → ESP-NOW "
                "if neighbor, else store/skip TX (sleep). Refuses LoRa."
            ),
            "always_on": "not in this table",
            "years": cfg["energy"]["simulation_years"],
            "capacities_mwh": caps,
            "p_deep_sleep_mw": cfg["energy"]["p_deep_sleep_mw"],
            "p_espnow_mw": cfg["energy"]["p_espnow_mw"],
            "p_lora_mw": cfg["energy"]["p_lora_mw"],
            "simulated": True,
        },
        "cart_test": cart_metrics,
        "summary": summary.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Capacity sweep (simulated, not a field measurement) ===")
    print(summary.to_string(index=False))
    print(f"\nWrote {sum_path}")
    print(f"Wrote {det_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {fig_path}")
    if ms_img.is_dir():
        print(f"Wrote {ms_img / 'energy_capacity_sweep.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
