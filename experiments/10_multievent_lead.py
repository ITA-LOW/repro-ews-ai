#!/usr/bin/env python3
"""WATCH versus WARNING timing across all eligible stations and audit events.

The analysis uses every usable INMET station in an event's state, excluding
the 100 stations used to fit CART. Events are the frozen convenience sample
in configs/experiment.yaml. Results are descriptive: stations within a storm
are spatially correlated, so uncertainty is bootstrapped by event.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ews_ai.cart_export import FEATURE_ORDER, fit_unscaled_cart, load_xy, predict_cart  # noqa: E402
from ews_ai.dataset import list_inmet_files, parse_inmet_file, process_station  # noqa: E402
from ews_ai.event_audit import load_processed_table  # noqa: E402
from ews_ai.paths import RESULTS_DIR, load_experiment_config, resolve_inmet_dir  # noqa: E402

PAD_DAYS = 7
MIN_COVERAGE = 0.80
BOOTSTRAP_REPS = 10_000
SENSITIVITY_THRESHOLDS = (0.10, 0.20, 0.30, 0.50)
FILE_RE = re.compile(r"_([A-Z]{2})_([AB]\d{3})_", re.IGNORECASE)

EVENT_LABELS = {
    "rs_floods_2024": "RS Apr 24",
    "rs_flood_2023_sep": "RS Sep 23",
    "rs_flood_2023_nov": "RS Nov 23",
    "sp_landslide_2023_feb": "SP Feb 23",
    "rj_flood_2024_jan": "RJ Jan 24",
    "rj_landslide_2024_mar": "RJ Mar 24",
    "pe_multiple_2023_may": "PE May 23",
    "pe_flood_2024_may": "PE May 24",
    "mg_landslide_2023_jan": "MG Jan 23",
    "ba_flood_2023_jan": "BA Jan 23",
}


def station_files_by_state(
    inmet_dir: Path,
    states: set[str],
    years: set[int],
) -> dict[str, dict[str, list[Path]]]:
    out: dict[str, dict[str, list[Path]]] = {uf: {} for uf in states}
    year_tokens = {str(y) for y in years}
    for path in list_inmet_files(inmet_dir):
        match = FILE_RE.search(path.name)
        if not match:
            continue
        uf, sid = match.group(1).upper(), match.group(2).upper()
        if uf not in states or not any(token in path.name for token in year_tokens):
            continue
        out[uf].setdefault(sid, []).append(path)
    return out


def load_state(
    uf: str,
    files: dict[str, list[Path]],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sid, paths in sorted(files.items()):
        parts = [parsed for path in paths if (parsed := parse_inmet_file(path)) is not None]
        if not parts:
            continue
        station = (
            pd.concat(parts, ignore_index=True)
            .sort_values("datetime")
            .drop_duplicates(subset=["datetime"])
        )
        processed = process_station(station, cfg)
        processed["station_id"] = sid
        processed["uf"] = uf
        frames.append(processed)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["station_id", "datetime"])


def first_edge_between(
    times: pd.Series,
    active: np.ndarray,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Timestamp | pd.NaT:
    previous = np.r_[False, active[:-1]]
    rising = active & ~previous
    candidates = times.loc[rising & (times >= start) & (times <= end)]
    return pd.Timestamp(candidates.iloc[0]) if len(candidates) else pd.NaT


def hours_before(trigger: pd.Timestamp | pd.NaT, peak: pd.Timestamp) -> float | None:
    if pd.isna(trigger):
        return None
    return round((peak - pd.Timestamp(trigger)).total_seconds() / 3600.0, 1)


def score_event_station(
    g: pd.DataFrame,
    event: dict[str, Any],
    threshold: float,
) -> dict[str, Any] | None:
    g = g.sort_values("datetime").reset_index(drop=True)
    start = pd.Timestamp(event["start"])
    end = pd.Timestamp(event["end"]) + pd.Timedelta(hours=23)
    official = g.loc[(g["datetime"] >= start) & (g["datetime"] <= end)]
    expected = int((end - start).total_seconds() / 3600.0) + 1
    coverage = len(official) / expected
    if official.empty or coverage < MIN_COVERAGE:
        return None

    peak_idx = int(official["precip_6h"].idxmax())
    peak_time = pd.Timestamp(g.loc[peak_idx, "datetime"])
    analysis_start = start - pd.Timedelta(days=PAD_DAYS)
    times = g["datetime"]
    watch = g["poisson_p_risk"].to_numpy(dtype=float) >= threshold
    warning = g["cart_hazard"].to_numpy(dtype=bool)
    first_watch = first_edge_between(times, watch, analysis_start, peak_time)
    first_warning = first_edge_between(times, warning, analysis_start, peak_time)
    start_positions = np.flatnonzero((times >= analysis_start).to_numpy())
    start_pos = int(start_positions[0]) if len(start_positions) else 0
    left_censored = bool(watch[start_pos] or warning[start_pos])

    if left_censored:
        delta = None
        category = "left_censored"
    elif pd.notna(first_watch) and pd.notna(first_warning):
        delta = round(
            (pd.Timestamp(first_warning) - pd.Timestamp(first_watch)).total_seconds() / 3600.0,
            1,
        )
        category = "watch_earlier" if delta > 0 else ("simultaneous" if delta == 0 else "warning_earlier")
    elif pd.notna(first_watch):
        delta = None
        category = "watch_only"
    elif pd.notna(first_warning):
        delta = None
        category = "warning_only"
    else:
        delta = None
        category = "neither"

    pre = (times >= analysis_start) & (times < start)
    previous_watch = np.r_[False, watch[:-1]]
    previous_warning = np.r_[False, warning[:-1]]
    return {
        "event_id": event["id"],
        "event_label": EVENT_LABELS.get(event["id"], event["id"]),
        "uf": event["uf"],
        "station_id": str(g["station_id"].iloc[0]),
        "coverage_fraction": round(coverage, 4),
        "peak_p6h_mm": round(float(g.loc[peak_idx, "precip_6h"]), 2),
        "peak_p6h_utc": peak_time.isoformat(),
        "first_watch_utc": first_watch.isoformat() if pd.notna(first_watch) else "",
        "first_warning_utc": first_warning.isoformat() if pd.notna(first_warning) else "",
        "lead_watch_h": hours_before(first_watch, peak_time),
        "lead_warning_h": hours_before(first_warning, peak_time),
        "watch_advantage_h": delta,
        "category": category,
        "watch_active_at_window_start": bool(watch[start_pos]),
        "warning_active_at_window_start": bool(warning[start_pos]),
        "pre_watch_edges": int(np.sum(watch & ~previous_watch & pre)),
        "pre_warning_edges": int(np.sum(warning & ~previous_warning & pre)),
    }


def summarize_events(detail: pd.DataFrame, events: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events:
        sub = detail.loc[detail["event_id"] == event["id"]]
        both = sub.loc[sub["category"].isin(["watch_earlier", "simultaneous", "warning_earlier"])]
        delta = both["watch_advantage_h"].dropna()
        rows.append(
            {
                "event_id": event["id"],
                "event_label": EVENT_LABELS.get(event["id"], event["id"]),
                "uf": event["uf"],
                "n_stations": int(len(sub)),
                "n_both": int(len(both)),
                "n_watch_earlier": int((sub["category"] == "watch_earlier").sum()),
                "n_simultaneous": int((sub["category"] == "simultaneous").sum()),
                "n_warning_earlier": int((sub["category"] == "warning_earlier").sum()),
                "n_watch_only": int((sub["category"] == "watch_only").sum()),
                "n_warning_only": int((sub["category"] == "warning_only").sum()),
                "n_neither": int((sub["category"] == "neither").sum()),
                "n_left_censored": int((sub["category"] == "left_censored").sum()),
                "median_advantage_h": round(float(delta.median()), 1) if len(delta) else None,
                "q1_advantage_h": round(float(delta.quantile(0.25)), 1) if len(delta) else None,
                "q3_advantage_h": round(float(delta.quantile(0.75)), 1) if len(delta) else None,
            }
        )
    return pd.DataFrame(rows)


def clustered_bootstrap(
    event_summary: pd.DataFrame,
    detail: pd.DataFrame,
    seed: int,
) -> dict[str, Any]:
    clusters = event_summary.reset_index(drop=True)
    total_both = int(clusters["n_both"].sum())
    total_earlier = int(clusters["n_watch_earlier"].sum())
    observed = total_earlier / total_both if total_both else np.nan
    all_deltas = detail["watch_advantage_h"].dropna()
    positive_deltas = detail.loc[
        detail["category"] == "watch_earlier", "watch_advantage_h"
    ].dropna()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(BOOTSTRAP_REPS):
        sampled = clusters.iloc[
            rng.integers(0, len(clusters), size=len(clusters))
        ]
        denominator = int(sampled["n_both"].sum())
        if denominator:
            draws.append(float(sampled["n_watch_earlier"].sum()) / denominator)
    low, high = np.quantile(draws, [0.025, 0.975]) if draws else (np.nan, np.nan)
    return {
        "n_event_clusters": int(len(clusters)),
        "n_station_event_pairs_both": total_both,
        "watch_earlier_fraction_among_both": round(float(observed), 4),
        "event_cluster_bootstrap_95ci": [round(float(low), 4), round(float(high), 4)],
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "median_advantage_all_dual_h": round(float(all_deltas.median()), 1),
        "iqr_advantage_all_dual_h": [
            round(float(all_deltas.quantile(0.25)), 1),
            round(float(all_deltas.quantile(0.75)), 1),
        ],
        "median_advantage_when_earlier_h": round(float(positive_deltas.median()), 1),
        "iqr_advantage_when_earlier_h": [
            round(float(positive_deltas.quantile(0.25)), 1),
            round(float(positive_deltas.quantile(0.75)), 1),
        ],
    }


def plot_event_summary(event_summary: pd.DataFrame, aggregate: dict[str, Any], out: Path) -> None:
    shown = event_summary.loc[event_summary["n_both"] > 0].copy()
    shown["fraction"] = shown["n_watch_earlier"] / shown["n_both"]
    shown = shown.iloc[::-1]
    fig, ax = plt.subplots(figsize=(3.0, 4.25), layout="constrained")
    y = np.arange(len(shown))
    ax.barh(y, 100 * shown["fraction"], color="#31577D")
    ax.set_yticks(y)
    ax.set_yticklabels(shown["event_label"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("WATCH earlier (%)")
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.6)
    ax.set_axisbelow(True)
    for yi, (_, row) in zip(y, shown.iterrows()):
        ax.text(
            min(98, 100 * row["fraction"] + 2),
            yi,
            f"{int(row['n_watch_earlier'])}/{int(row['n_both'])}",
            va="center",
            ha="right" if row["fraction"] > 0.88 else "left",
            fontsize=7,
        )
    ax.set_title("WATCH earlier by event")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-event WATCH/WARNING lead analysis.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    events = list((cfg.get("event_audit") or {}).get("events") or [])
    threshold = float(cfg["event_audit"]["p_risk_threshold"])
    seed = int(cfg.get("seed", 42))

    processed = load_processed_table()
    train_stations = set(processed["station_id"].astype(str).str.upper())
    x, y, station_ids = load_xy(processed)
    tree, cart_metrics = fit_unscaled_cart(x, y, station_ids, cfg)

    states = {str(event["uf"]).upper() for event in events}
    years = {pd.Timestamp(event["start"]).year for event in events}
    file_map = station_files_by_state(resolve_inmet_dir(), states, years)
    state_frames: dict[str, pd.DataFrame] = {}
    for uf in sorted(states):
        print(f"Loading {uf}: {len(file_map.get(uf, {}))} station ids")
        frame = load_state(uf, file_map.get(uf, {}), cfg)
        if not frame.empty:
            frame = frame.loc[~frame["station_id"].isin(train_stations)].copy()
            frame["cart_class"] = predict_cart(
                tree, frame[list(FEATURE_ORDER)].to_numpy(dtype=np.float64)
            )
            frame["cart_hazard"] = (frame["cart_class"] > 0).astype(np.int8)
        state_frames[uf] = frame

    rows: list[dict[str, Any]] = []
    for event in events:
        state = state_frames.get(str(event["uf"]).upper(), pd.DataFrame())
        if state.empty:
            continue
        for _, station in state.groupby("station_id"):
            row = score_event_station(station, event, threshold)
            if row is not None:
                rows.append(row)

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise RuntimeError("No eligible event-station pairs.")
    event_summary = summarize_events(detail, events)
    aggregate = clustered_bootstrap(event_summary, detail, seed)
    aggregate.update(
        {
            "n_events": int(len(event_summary)),
            "n_station_event_pairs": int(len(detail)),
            "category_counts": {
                key: int(value)
                for key, value in detail["category"].value_counts().sort_index().items()
            },
            "pad_days": PAD_DAYS,
            "minimum_official_window_coverage": MIN_COVERAGE,
            "training_stations_excluded": len(train_stations),
            "cart_holdout_recall": cart_metrics["hazard_recall"],
            "protocol": (
                "All usable stations in each event UF; CART training stations excluded. "
                "First rising edge in [event start - 7 days, station peak P6h]. "
                "Pairs active at the seven-day boundary are left-censored and excluded "
                "from timing categories. "
                "Lead is relative to station peak P6h, not disaster onset. "
                "The estimand pools uncensored dual-edge station-event pairs. "
                "All ten convenience-sampled events are resampled for a descriptive "
                "percentile interval; this does not imply population-level forecast skill."
            ),
        }
    )

    sensitivity_rows: list[dict[str, Any]] = []
    for candidate in SENSITIVITY_THRESHOLDS:
        if np.isclose(candidate, threshold):
            candidate_detail = detail
        else:
            candidate_rows: list[dict[str, Any]] = []
            for event in events:
                state = state_frames.get(str(event["uf"]).upper(), pd.DataFrame())
                if state.empty:
                    continue
                for _, station in state.groupby("station_id"):
                    row = score_event_station(station, event, candidate)
                    if row is not None:
                        candidate_rows.append(row)
            candidate_detail = pd.DataFrame(candidate_rows)
        candidate_events = summarize_events(candidate_detail, events)
        candidate_aggregate = clustered_bootstrap(
            candidate_events, candidate_detail, seed
        )
        sensitivity_rows.append(
            {
                "watch_threshold": candidate,
                "n_pairs": int(len(candidate_detail)),
                "n_left_censored": int(
                    (candidate_detail["category"] == "left_censored").sum()
                ),
                "n_dual": candidate_aggregate["n_station_event_pairs_both"],
                "watch_earlier_fraction": candidate_aggregate[
                    "watch_earlier_fraction_among_both"
                ],
                "bootstrap_95ci_low": candidate_aggregate[
                    "event_cluster_bootstrap_95ci"
                ][0],
                "bootstrap_95ci_high": candidate_aggregate[
                    "event_cluster_bootstrap_95ci"
                ][1],
                "median_positive_advantage_h": candidate_aggregate[
                    "median_advantage_when_earlier_h"
                ],
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(RESULTS_DIR / "multievent_lead_by_station.csv", index=False)
    event_summary.to_csv(RESULTS_DIR / "multievent_lead_by_event.csv", index=False)
    sensitivity.to_csv(
        RESULTS_DIR / "multievent_threshold_sensitivity.csv", index=False
    )
    (RESULTS_DIR / "multievent_lead_summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )

    figure_dir = RESULTS_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "multievent_watch_warning.png"
    plot_event_summary(event_summary, aggregate, figure_path)
    paper_path = ROOT.parent / "artigo_2_ews-ai" / "manuscrito" / "img" / figure_path.name
    paper_path.write_bytes(figure_path.read_bytes())

    print(event_summary.to_string(index=False))
    print(sensitivity.to_string(index=False))
    print(json.dumps(aggregate, indent=2))
    print(f"Wrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
