#!/usr/bin/env python3
"""CART vs Poisson accumulation on real RS Apr–May 2024 hours.

Question: does P_risk >= 0.20 fire before CART class != 0, relative to
peak 6 h rain? Hourly INMET only. Not a forecast of future rain.

    python experiments/08_cart_poisson_lead.py
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
from ews_ai.event_audit import load_processed_table, load_stations  # noqa: E402
from ews_ai.paths import RESULTS_DIR, load_experiment_config  # noqa: E402

STATIONS = ("A801", "A803", "A827")
PAD_DAYS = 7
RULES = ("cart", "poisson", "or", "and")
RULE_LABEL = {
    "cart": "CART only",
    "poisson": r"$P_{risk}\geq 0.20$",
    "or": "OR (earliest)",
    "and": "AND (both)",
}
RULE_COLOR = {
    "cart": "#C44E52",
    "poisson": "#9467BD",
    "or": "#DD8452",
    "and": "#4C72B0",
}


def _first_true(times: pd.Series, mask: np.ndarray):
    hit = np.flatnonzero(mask)
    if len(hit) == 0:
        return pd.NaT
    return pd.Timestamp(times.iloc[int(hit[0])])


def _hours_between(a, b) -> float | None:
    if pd.isna(a) or pd.isna(b):
        return None
    return round((pd.Timestamp(b) - pd.Timestamp(a)).total_seconds() / 3600.0, 1)


def score_station(
    g: pd.DataFrame,
    threshold: float,
    official_start: pd.Timestamp,
    official_end: pd.Timestamp,
) -> tuple[dict, pd.DataFrame]:
    g = g.sort_values("datetime").reset_index(drop=True)
    t = g["datetime"]
    cart = g["cart_hazard"].to_numpy(dtype=bool)
    poi = g["poisson_p_risk"].to_numpy(dtype=np.float64) >= float(threshold)
    official = (t >= official_start) & (t <= official_end)
    pre = t < official_start

    if not official.any():
        raise RuntimeError(f"No official-window hours for {g['station_id'].iloc[0]}")

    peak_idx = int(g.loc[official, "precip_6h"].idxmax())
    peak_time = pd.Timestamp(g.loc[peak_idx, "datetime"])
    peak_p6 = float(g.loc[peak_idx, "precip_6h"])

    masks = {
        "cart": cart,
        "poisson": poi,
        "or": cart | poi,
        "and": cart & poi,
    }
    # First fire from the start of the plotted series (includes pre-event).
    first = {name: _first_true(t, mask) for name, mask in masks.items()}

    n_poi_before_cart = int(np.sum(poi & ~cart & (t < (first["cart"] if pd.notna(first["cart"]) else t.iloc[-1] + pd.Timedelta(hours=1)))))

    row = {
        "station_id": str(g["station_id"].iloc[0]),
        "n_hours": int(len(g)),
        "peak_p6h_mm": round(peak_p6, 2),
        "peak_p6h_utc": peak_time.isoformat(),
        "n_cart_hours": int(cart.sum()),
        "n_poisson_hours": int(poi.sum()),
        "n_or_hours": int((cart | poi).sum()),
        "n_and_hours": int((cart & poi).sum()),
        "n_poisson_only_hours": int((poi & ~cart).sum()),
        "n_poisson_only_before_first_cart": n_poi_before_cart,
        "false_cart_before_official": int((cart & pre).sum()),
        "false_poisson_before_official": int((poi & pre).sum()),
        "false_or_before_official": int(((cart | poi) & pre).sum()),
        "false_and_before_official": int(((cart & poi) & pre).sum()),
    }
    for name in RULES:
        row[f"first_{name}_utc"] = first[name].isoformat() if pd.notna(first[name]) else ""
        row[f"lead_{name}_h"] = _hours_between(first[name], peak_time)

    cart_after_poi = None
    if pd.notna(first["poisson"]) and pd.notna(first["cart"]):
        cart_after_poi = _hours_between(first["poisson"], first["cart"])
    row["cart_lag_after_poisson_h"] = cart_after_poi
    row["poisson_earlier_than_cart"] = bool(
        pd.notna(first["poisson"])
        and pd.notna(first["cart"])
        and first["poisson"] < first["cart"]
    )
    return row, g


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


def plot_station(
    g: pd.DataFrame,
    row: dict,
    threshold: float,
    official_start,
    official_end,
    out: Path,
) -> None:
    g = g.sort_values("datetime")
    t = pd.to_datetime(g["datetime"])
    peak = pd.Timestamp(row["peak_p6h_utc"])
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.6), sharex=True)

    ax = axes[0]
    ax.fill_between(t, g["precip_6h"], color="#7EA6C8", alpha=0.65, label=r"$P_{6h}$")
    ax.axhline(30.0, color="#1f77b4", ls=":", lw=1, label=r"flood $P_{6h}=30$ mm")
    ax.axvline(peak, color="k", ls="--", lw=1, label="peak $P_{6h}$")
    ax.axvspan(official_start, official_end, color="k", alpha=0.05)
    ax.set_ylabel("mm")
    ax.set_title(f"{row['station_id']} — CART vs Poisson accumulation (RS 2024, INMET hours)")
    ax.legend(loc="upper left", frameon=False, ncol=3)

    ax = axes[1]
    ax.plot(t, g["poisson_p_risk"], color=RULE_COLOR["poisson"], lw=1.6, label=r"$P_{risk}$")
    ax.axhline(threshold, color=RULE_COLOR["poisson"], ls=":", lw=1)
    cart_t = t[g["cart_hazard"] == 1]
    if len(cart_t):
        ax.scatter(
            cart_t,
            np.full(len(cart_t), -0.04),
            marker="|",
            s=80,
            c=RULE_COLOR["cart"],
            label="CART hazard",
        )
    ax.axvline(peak, color="k", ls="--", lw=1)
    ax.axvspan(official_start, official_end, color="k", alpha=0.05)
    ymax = 1.08
    for name, y in (("poisson", 0.20), ("cart", 0.42), ("or", 0.64), ("and", 0.86)):
        key = f"first_{name}_utc"
        if not row[key]:
            continue
        ts = pd.Timestamp(row[key])
        lead = row[f"lead_{name}_h"]
        ax.axvline(ts, color=RULE_COLOR[name], lw=1.1, alpha=0.85)
        ax.annotate(
            f"{RULE_LABEL[name]}\nlead {lead} h" if lead is not None else RULE_LABEL[name],
            xy=(ts, y),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=7,
            color=RULE_COLOR[name],
        )
    ax.set_ylim(-0.12, ymax)
    ax.set_ylabel(r"$P_{risk}$")
    ax.set_xlabel("UTC")
    ax.legend(loc="upper left", frameon=False, ncol=2)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_lead_bars(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(summary))
    width = 0.2
    for i, name in enumerate(RULES):
        vals = [float(v) if pd.notna(v) else 0.0 for v in summary[f"lead_{name}_h"]]
        ax.bar(x + (i - 1.5) * width, vals, width, color=RULE_COLOR[name], label=RULE_LABEL[name])
    ax.set_xticks(x)
    ax.set_xticklabels(summary["station_id"])
    ax.set_ylabel("Lead vs peak $P_{6h}$ (h)")
    ax.set_title("Earlier trigger = larger lead (real RS hours)")
    ax.legend(frameon=False, ncol=2)
    ax.axhline(0, color="k", lw=0.6)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="CART vs Poisson lead on RS 2024.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    threshold = float(cfg["event_audit"]["p_risk_threshold"])
    event = next(e for e in cfg["event_audit"]["events"] if e["id"] == "rs_floods_2024")
    official_start = pd.Timestamp(event["start"])
    official_end = pd.Timestamp(event["end"]) + pd.Timedelta(hours=23)
    win_start = official_start - pd.Timedelta(days=PAD_DAYS)
    win_end = official_end + pd.Timedelta(days=PAD_DAYS)

    print("Fitting unscaled CART on the 100-station table...")
    processed = load_processed_table()
    x, y, stations = load_xy(processed)
    tree, cart_m = fit_unscaled_cart(x, y, stations, cfg)
    print(f"CART test recall {100 * cart_m['hazard_recall']:.2f}%")

    print(f"Loading INMET {STATIONS}...")
    storm = load_stations(cfg, list(STATIONS))
    if storm.empty:
        print("No RS station files. Set EWS_AI_INMET_DIR.")
        return 2
    storm = storm.loc[(storm["datetime"] >= win_start) & (storm["datetime"] <= win_end)].copy()

    rows = []
    frames = []
    for sid, g in storm.groupby("station_id"):
        g = g.sort_values("datetime").reset_index(drop=True)
        x_s = g[list(FEATURE_ORDER)].to_numpy(dtype=np.float64)
        g = g.copy()
        g["cart_class"] = predict_cart(tree, x_s)
        g["cart_hazard"] = (g["cart_class"] > 0).astype(np.int32)
        row, scored = score_station(g, threshold, official_start, official_end)
        rows.append(row)
        frames.append(scored)
        lag = row["cart_lag_after_poisson_h"]
        print(
            f"  {sid}: peak P6={row['peak_p6h_mm']} mm @ {row['peak_p6h_utc']}  "
            f"lead CART {row['lead_cart_h']} h  Poisson {row['lead_poisson_h']} h  "
            f"CART after Poisson {lag} h  "
            f"Poisson-only hours {row['n_poisson_only_hours']}"
        )

    summary = pd.DataFrame(rows)
    hourly = pd.concat(frames, ignore_index=True)

    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _style()
    figures = []
    for sid, g in hourly.groupby("station_id"):
        row = summary.loc[summary["station_id"] == sid].iloc[0].to_dict()
        dest = fig_dir / f"rs2024_{sid.lower()}_cart_poisson.png"
        plot_station(g, row, threshold, official_start, official_end, dest)
        figures.append(dest)
        print(f"Wrote {dest}")
    bar_path = fig_dir / "rs2024_lead_bars.png"
    plot_lead_bars(summary, bar_path)
    figures.append(bar_path)
    print(f"Wrote {bar_path}")

    paper_img = ROOT.parent / "artigo_2_ews-ai" / "manuscrito" / "img"
    paper_img.mkdir(parents=True, exist_ok=True)
    for src in figures:
        (paper_img / src.name).write_bytes(src.read_bytes())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sum_path = RESULTS_DIR / "cart_poisson_lead.csv"
    hour_path = RESULTS_DIR / "cart_poisson_lead_hourly.csv"
    json_path = RESULTS_DIR / "cart_poisson_lead.json"
    summary.to_csv(sum_path, index=False)
    hourly.to_csv(hour_path, index=False)
    n_earlier = int(summary["poisson_earlier_than_cart"].sum())
    payload = {
        "question": (
            "On real RS hourly INMET, does P_risk >= 0.20 fire before CART "
            "class != 0, relative to peak 6 h rain?"
        ),
        "event": event["id"],
        "official": {"start": event["start"], "end": event["end"]},
        "window": {"start": str(win_start), "end": str(win_end)},
        "p_risk_threshold": threshold,
        "cart_test_recall": cart_m["hazard_recall"],
        "stations_where_poisson_earlier": n_earlier,
        "n_stations": int(len(summary)),
        "note": (
            "Not a forecast. Poisson integrates past/present rain and soil. "
            "Lead is vs peak P_6h and vs first CART hour. "
            "false_*_before_official counts fires before 2024-04-26."
        ),
        "summary": summary.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(summary.to_string(index=False))
    print(f"\nPoisson earlier than CART on {n_earlier}/{len(summary)} stations.")
    print(f"Wrote {sum_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
