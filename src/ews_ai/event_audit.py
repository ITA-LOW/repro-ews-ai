"""Held-out audit of publicly reported hazard episodes.

The event list is never written into λ(t) or risk_label. This module only asks
whether Poisson P_risk, built from INMET rain and the soil bucket, crosses the
BDI threshold during those windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ews_ai.dataset import (
    list_inmet_files,
    parse_inmet_file,
    process_station,
    station_id_from_filename,
)
from ews_ai.paths import PROCESSED_DIR, resolve_inmet_dir


def load_processed_table(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or (PROCESSED_DIR / "hourly_dataset.csv")
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run experiments/01_build_dataset.py first."
        )
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    if "poisson_p_risk" not in df.columns:
        raise ValueError("hourly_dataset.csv has no poisson_p_risk column.")
    return df


def find_station_files(inmet_dir: Path, station_id: str, years: list[int]) -> list[Path]:
    year_tokens = {str(y) for y in years}
    sid = station_id.upper()
    found = []
    for path in list_inmet_files(inmet_dir):
        if station_id_from_filename(path) != sid:
            continue
        if years and not any(tok in path.name for tok in year_tokens):
            continue
        found.append(path)
    return sorted(found, key=lambda p: p.name)


def load_stations(cfg: dict[str, Any], station_ids: list[str]) -> pd.DataFrame:
    """Parse and process named stations with the same physics as experiment 01."""
    inmet_dir = resolve_inmet_dir()
    years = list(cfg.get("years") or [])
    frames: list[pd.DataFrame] = []
    for sid in station_ids:
        parts = []
        for path in find_station_files(inmet_dir, sid, years):
            parsed = parse_inmet_file(path)
            if parsed is not None:
                parts.append(parsed)
        if not parts:
            continue
        station_df = pd.concat(parts, ignore_index=True)
        station_df = station_df.sort_values("datetime").drop_duplicates(subset=["datetime"])
        frames.append(process_station(station_df, cfg))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def event_coverage(
    processed: pd.DataFrame,
    event: dict[str, Any],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Hours for one event: processed UF rows plus any missing preferred stations."""
    uf = event["uf"]
    preferred = [s.upper() for s in (event.get("preferred_stations") or [])]
    local = processed.loc[processed["uf"] == uf].copy()
    have = set(local["station_id"].astype(str).str.upper()) if len(local) else set()
    missing = [s for s in preferred if s not in have]
    extra = load_stations(cfg, missing) if missing else pd.DataFrame()
    if extra.empty:
        return local
    extra = extra.loc[extra["uf"] == uf]
    return pd.concat([local, extra], ignore_index=True)


def score_event(
    hours: pd.DataFrame,
    event: dict[str, Any],
    threshold: float,
    pre_window_hours: int,
) -> dict[str, Any]:
    start = pd.Timestamp(event["start"])
    end = pd.Timestamp(event["end"]) + pd.Timedelta(hours=23)
    watch_start = start - pd.Timedelta(hours=int(pre_window_hours))

    if hours.empty:
        return _empty_score(event, "no_station_in_uf")

    window = hours.loc[(hours["datetime"] >= watch_start) & (hours["datetime"] <= end)]
    official = hours.loc[(hours["datetime"] >= start) & (hours["datetime"] <= end)]
    if official.empty:
        return _empty_score(event, "no_hours_in_window")

    stations = sorted(official["station_id"].astype(str).unique())
    max_p_risk = float(official["poisson_p_risk"].max())
    max_p6h = float(official["precip_6h"].max())
    peak_idx = official["precip_6h"].idxmax()
    peak_time = official.loc[peak_idx, "datetime"]
    peak_station = str(official.loc[peak_idx, "station_id"])

    alerts = window.loc[window["poisson_p_risk"] >= threshold]
    detected = not alerts.empty
    first_alert = alerts["datetime"].min() if detected else pd.NaT
    first_station = (
        str(alerts.loc[alerts["datetime"].idxmin(), "station_id"]) if detected else ""
    )
    lead_hours = None
    if detected and pd.notna(first_alert):
        lead_hours = round((peak_time - first_alert).total_seconds() / 3600.0, 1)

    preferred = [s.upper() for s in (event.get("preferred_stations") or [])]
    pref_hit = False
    if preferred:
        pref_hours = official.loc[official["station_id"].astype(str).str.upper().isin(preferred)]
        pref_hit = (not pref_hours.empty) and float(pref_hours["poisson_p_risk"].max()) >= threshold

    return {
        "event_id": event["id"],
        "name": event.get("name", event["id"]),
        "uf": event["uf"],
        "type": event.get("type", ""),
        "start": event["start"],
        "end": event["end"],
        "status": "scored",
        "n_stations": len(stations),
        "stations": ",".join(stations),
        "max_poisson_p_risk": round(max_p_risk, 4),
        "max_precip_6h_mm": round(max_p6h, 2),
        "peak_precip_6h_utc": peak_time.isoformat(),
        "peak_station": peak_station,
        "detected": bool(detected),
        "first_alert_utc": first_alert.isoformat() if detected else "",
        "first_alert_station": first_station,
        "lead_hours_vs_peak_p6h": lead_hours,
        "preferred_station_detected": bool(pref_hit),
        "source": event.get("source", ""),
        "note": (
            "P_risk from rain/soil only. Event dates are not inputs to λ(t)."
        ),
    }


def _empty_score(event: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "event_id": event["id"],
        "name": event.get("name", event["id"]),
        "uf": event["uf"],
        "type": event.get("type", ""),
        "start": event["start"],
        "end": event["end"],
        "status": status,
        "n_stations": 0,
        "stations": "",
        "max_poisson_p_risk": None,
        "max_precip_6h_mm": None,
        "peak_precip_6h_utc": "",
        "peak_station": "",
        "detected": False,
        "first_alert_utc": "",
        "first_alert_station": "",
        "lead_hours_vs_peak_p6h": None,
        "preferred_station_detected": False,
        "source": event.get("source", ""),
        "note": "No INMET hours available for this UF/window in the processed or preferred set.",
    }


def run_audit(cfg: dict[str, Any], processed: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    block = cfg.get("event_audit") or {}
    events = block.get("events") or []
    threshold = float(block.get("p_risk_threshold", cfg.get("bdi", {}).get("hazard_p_risk", 0.20)))
    pre_hours = int(block.get("pre_window_hours", 24))

    rows = []
    for event in events:
        hours = event_coverage(processed, event, cfg)
        rows.append(score_event(hours, event, threshold, pre_hours))

    table = pd.DataFrame(rows)
    scored = table.loc[table["status"] == "scored"]
    detected = scored.loc[scored["detected"] == True]  # noqa: E712
    leads = detected["lead_hours_vs_peak_p6h"].dropna()
    summary = {
        "n_events": int(len(table)),
        "n_scored": int(len(scored)),
        "n_uncovered": int((table["status"] != "scored").sum()),
        "n_detected": int(len(detected)),
        "event_hit_rate": (
            round(len(detected) / len(scored), 4) if len(scored) else None
        ),
        "median_lead_hours_vs_peak_p6h": (
            round(float(leads.median()), 1) if len(leads) else None
        ),
        "p_risk_threshold": threshold,
        "pre_window_hours": pre_hours,
        "protocol": (
            "Convenience sample of public episodes. Dates are not used in λ(t) "
            "or risk_label. Detection = max Poisson P_risk >= threshold at any "
            "station in the UF during [start - pre_window, end]. Matching is "
            "UF-scale, not a complete CEMADEN inventory."
        ),
    }
    return table, summary
