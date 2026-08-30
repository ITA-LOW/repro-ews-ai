#!/usr/bin/env python3
"""Duty cycle at wake resolution on a harsh window (RS Apr–May 2024).

INMET weather is hourly. BDI shortens T_sleep to 120 s / 300 s; FSM stays
at 3600 s. Extra wakes are sampling density, not extra hourly recall.

Usage (after experiment 01; needs INMET for A801/A803/A827):

    python experiments/07_storm_duty.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ews_ai.cart_export import FEATURE_ORDER, fit_unscaled_cart, load_xy, predict_cart  # noqa: E402
from ews_ai.duty_cycle import (  # noqa: E402
    POLICY_BDI_RATE,
    POLICY_BDI_TX,
    POLICY_FSM_FIXED,
    simulate_window,
    summarize,
)
from ews_ai.event_audit import load_processed_table, load_stations  # noqa: E402
from ews_ai.paths import RESULTS_DIR, load_experiment_config  # noqa: E402

STATIONS = ("A801", "A803", "A827")
PAD_DAYS = 7
COLORS = {
    POLICY_FSM_FIXED: "#4C72B0",
    POLICY_BDI_TX: "#C44E52",
    POLICY_BDI_RATE: "#55A868",
}
LABELS = {
    POLICY_FSM_FIXED: "FSM 3600 s",
    POLICY_BDI_TX: "BDI adaptive (TX each wake)",
    POLICY_BDI_RATE: "BDI adaptive (1 TX / hour)",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_a803(trace: pd.DataFrame, event_start, event_end, out: Path) -> None:
    g = trace.loc[trace["station_id"] == "A803"].copy()
    g["datetime"] = pd.to_datetime(g["datetime"])
    weather = g.loc[g["policy"] == POLICY_FSM_FIXED]
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 6.2), sharex=True)

    ax = axes[0]
    ax.fill_between(weather["datetime"], weather["precip_6h"], color="#7EA6C8", alpha=0.7)
    ax.scatter(
        weather.loc[weather["hazard"] == 1, "datetime"],
        weather.loc[weather["hazard"] == 1, "precip_6h"],
        s=8,
        c="#C44E52",
        zorder=3,
        label="CART hazard hour",
    )
    ax.set_ylabel("P$_{6h}$ (mm)")
    ax.set_title("A803 — RS floods 2024, duty cycle at wake resolution")
    ax.legend(loc="upper left", frameon=False)
    ax.axvspan(event_start, event_end, color="k", alpha=0.06)

    ax = axes[1]
    for policy in (POLICY_FSM_FIXED, POLICY_BDI_TX, POLICY_BDI_RATE):
        p = g.loc[g["policy"] == policy]
        ax.step(p["datetime"], p["t_sleep_s"], where="post", color=COLORS[policy], label=LABELS[policy])
    ax.set_ylabel("T$_{sleep}$ (s)")
    ax.set_yscale("log")
    ax.set_yticks([120, 300, 3600])
    ax.set_yticklabels(["120", "300", "3600"])
    ax.legend(loc="upper right", frameon=False)
    ax.axvspan(event_start, event_end, color="k", alpha=0.06)

    ax = axes[2]
    for policy in (POLICY_FSM_FIXED, POLICY_BDI_TX, POLICY_BDI_RATE):
        p = g.loc[g["policy"] == policy]
        ax.plot(p["datetime"], p["soc_pct"], color=COLORS[policy], label=LABELS[policy], lw=1.4)
    ax.set_ylabel("SoC (%)")
    ax.set_xlabel("UTC")
    ax.set_ylim(0, 105)
    ax.axvspan(event_start, event_end, color="k", alpha=0.06)
    ax.legend(loc="lower left", frameon=False)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_soc_grid(trace: pd.DataFrame, event_start, event_end, out: Path) -> None:
    trace = trace.copy()
    trace["datetime"] = pd.to_datetime(trace["datetime"])
    stations = [s for s in STATIONS if s in set(trace["station_id"])]
    fig, axes = plt.subplots(len(stations), 1, figsize=(8.4, 2.15 * len(stations)), sharex=True)
    if len(stations) == 1:
        axes = [axes]
    for ax, sid in zip(axes, stations):
        for policy in (POLICY_FSM_FIXED, POLICY_BDI_TX, POLICY_BDI_RATE):
            p = trace.loc[(trace["station_id"] == sid) & (trace["policy"] == policy)]
            ax.plot(p["datetime"], p["soc_pct"], color=COLORS[policy], label=LABELS[policy], lw=1.3)
        ax.axvspan(event_start, event_end, color="k", alpha=0.06)
        ax.set_ylabel("SoC (%)")
        ax.set_ylim(0, 105)
        ax.set_title(sid)
    axes[0].legend(loc="lower left", frameon=False, ncol=3)
    axes[-1].set_xlabel("UTC")
    fig.suptitle("SoC during the RS Apr–May 2024 window (simulated)", y=1.01)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_bars(summary: pd.DataFrame, out: Path) -> None:
    policies = [POLICY_FSM_FIXED, POLICY_BDI_RATE, POLICY_BDI_TX]
    stations = [s for s in STATIONS if s in set(summary["station_id"])]
    x = np.arange(len(stations))
    width = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for i, (col, title, ylab) in enumerate(
        (
            ("n_wakes", "Wakes in the window", "wakes"),
            ("energy_mwh", "Energy in the window", "mWh"),
        )
    ):
        ax = axes[i]
        for j, policy in enumerate(policies):
            vals = [
                float(summary.loc[(summary.station_id == s) & (summary.policy == policy), col].iloc[0])
                for s in stations
            ]
            ax.bar(x + (j - 1) * width, vals, width, color=COLORS[policy], label=LABELS[policy])
        ax.set_xticks(x)
        ax.set_xticklabels(stations)
        ax.set_title(title)
        ax.set_ylabel(ylab)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="RS 2024 duty-cycle storm window.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)

    event = next(e for e in cfg["event_audit"]["events"] if e["id"] == "rs_floods_2024")
    start = pd.Timestamp(event["start"])
    end = pd.Timestamp(event["end"]) + pd.Timedelta(hours=23)
    win_start = start - pd.Timedelta(days=PAD_DAYS)
    win_end = end + pd.Timedelta(days=PAD_DAYS)

    print("Fitting unscaled CART on the 100-station table...")
    processed = load_processed_table()
    x, y, stations = load_xy(processed)
    tree, cart_m = fit_unscaled_cart(x, y, stations, cfg)
    print(f"CART test recall {100*cart_m['hazard_recall']:.2f}%")

    print(f"Loading INMET stations {STATIONS}...")
    storm = load_stations(cfg, list(STATIONS))
    if storm.empty:
        print("No RS station files. Set EWS_AI_INMET_DIR.")
        return 2
    storm = storm.loc[(storm["datetime"] >= win_start) & (storm["datetime"] <= win_end)]
    print(f"Window {win_start.date()} … {win_end.date()}  rows {len(storm)}")

    traces = []
    for sid, g in storm.groupby("station_id"):
        g = g.sort_values("datetime").reset_index(drop=True)
        x_s = g[list(FEATURE_ORDER)].to_numpy(dtype=np.float64)
        haz = predict_cart(tree, x_s) > 0
        traces.append(simulate_window(g, haz, cfg))
        print(f"  {sid}: {int(haz.sum())} CART hazard hours / {len(g)}")

    trace = pd.concat(traces, ignore_index=True)
    summary = summarize(trace)

    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _style()

    p_soc = fig_dir / "rs2024_soc.png"
    p_a803 = fig_dir / "rs2024_a803_duty.png"
    p_bars = fig_dir / "rs2024_wakes_energy.png"
    plot_soc_grid(trace, start, end, p_soc)
    if "A803" in set(trace["station_id"]):
        plot_a803(trace, start, end, p_a803)
    plot_bars(summary, p_bars)

    paper_img = ROOT.parent / "artigo_2_ews-ai" / "manuscrito" / "img"
    copied = []
    paper_img.mkdir(parents=True, exist_ok=True)
    if paper_img.is_dir():
        for src in (p_soc, p_a803, p_bars):
            if src.is_file():
                dest = paper_img / src.name
                dest.write_bytes(src.read_bytes())
                copied.append(str(dest))

    trace_path = RESULTS_DIR / "storm_duty_trace.csv"
    sum_path = RESULTS_DIR / "storm_duty_summary.csv"
    json_path = RESULTS_DIR / "storm_duty_summary.json"
    trace.to_csv(trace_path, index=False)
    summary.to_csv(sum_path, index=False)
    payload = {
        "event": event["id"],
        "window": {"start": str(win_start), "end": str(win_end)},
        "official": {"start": event["start"], "end": event["end"]},
        "stations": list(STATIONS),
        "cart_test_recall": cart_m["hazard_recall"],
        "protocol": (
            "INMET hourly weather. FSM wakes once per hour. BDI T_sleep is "
            "120 s on CART hazard, 300 s if precip_1h>5 mm, else 3600 s. "
            "bdi_adaptive_tx_per_wake transmits on every wake during hazard. "
            "bdi_adaptive_tx_hourly keeps the same wakes but one LoRa per hour. "
            "Extra wakes are sampling density, not extra hourly-label recall. "
            "Simulated datasheet energy, not a current-shunt measurement."
        ),
        "summary": summary.to_dict(orient="records"),
        "figures": [str(p_soc), str(p_a803), str(p_bars)],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(summary.to_string(index=False))
    print(f"\nWrote {sum_path}")
    print(f"Wrote {p_soc}")
    print(f"Wrote {p_a803}")
    print(f"Wrote {p_bars}")
    for c in copied:
        print(f"Copied {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
