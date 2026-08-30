"""Build the hourly table from official INMET CSVs.

Invariants
----------
* Rolling rain, soil moisture, Poisson intensity, and 6-hour windows never
  cross a station boundary.
* ``risk_label`` uses physical rain/soil thresholds only. Poisson ``p_risk``
  is stored as a belief/feature and is not written into y.
* Station id, timestamp, and UF are kept so temporal and station hold-outs
  are possible.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ews_ai.paths import PROCESSED_DIR, STATIONS_JSON, resolve_inmet_dir

STATION_CODE_RE = re.compile(r"([AB]\d{3})", re.IGNORECASE)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _clean_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", ".", regex=False).str.strip()
    num = pd.to_numeric(s, errors="coerce")
    return num.replace(-9999.0, np.nan)


def parse_inmet_file(filepath: Path) -> pd.DataFrame | None:
    """Parse one official INMET hourly CSV. Returns None if unusable."""
    try:
        header_line = 8
        meta_uf = None
        meta_code = None
        meta_name = None
        with filepath.open("r", encoding="latin1") as f:
            for idx, line in enumerate(f):
                raw = line.strip()
                if ":" in raw:
                    key, _, val = raw.partition(":")
                    key_u = key.strip().upper()
                    val = val.replace(";", "").strip()
                    if "ESTADO" in key_u or key_u == "UF":
                        meta_uf = val
                    elif "CODIGO" in key_u or "CÓDIGO" in key_u:
                        meta_code = val
                    elif "ESTACAO" in key_u or "ESTAÇÃO" in key_u:
                        meta_name = val
                if raw.startswith("Data") or "Data;" in raw or "DATA;" in raw:
                    header_line = idx
                    break

        df = pd.read_csv(
            filepath,
            sep=";",
            skiprows=header_line,
            encoding="latin1",
            on_bad_lines="skip",
        )
        df.columns = [str(c).strip().upper() for c in df.columns]

        col_date = next(c for c in df.columns if "DATA" in c)
        col_time = next(c for c in df.columns if "HORA" in c)
        col_precip = next(c for c in df.columns if "PRECIP" in c)

        temp_candidates = [
            c
            for c in df.columns
            if "TEMPERATURA" in c and ("SECO" in c or "BULBO" in c or "AR" in c)
        ]
        col_temp = (
            temp_candidates[0]
            if temp_candidates
            else next(c for c in df.columns if "TEMPERATURA" in c)
        )
        hum_candidates = [
            c
            for c in df.columns
            if "UMIDADE" in c and ("RELATIVA" in c or "HORARIA" in c or "AR" in c)
        ]
        col_hum = (
            hum_candidates[0]
            if hum_candidates
            else next(c for c in df.columns if "UMIDADE" in c)
        )
        wind_candidates = [c for c in df.columns if "VENTO" in c and "VELOCIDADE" in c]
        if not wind_candidates:
            wind_candidates = [c for c in df.columns if "VELOCIDADE" in c]
        col_wind = wind_candidates[0] if wind_candidates else None
        solar_candidates = [c for c in df.columns if "RADIACAO" in c or "RADIAÇÃO" in c]
        col_solar = solar_candidates[0] if solar_candidates else None

        date_series = df[col_date].astype(str).str.strip().str.replace("/", "-", regex=False)
        time_series = (
            df[col_time]
            .astype(str)
            .str.replace("UTC", "", regex=False)
            .str.replace(":", "", regex=False)
            .str.strip()
        )
        time_series = time_series.str.zfill(4).str[:2] + ":00"

        out = pd.DataFrame()
        out["datetime"] = pd.to_datetime(date_series + " " + time_series, errors="coerce")
        out["precip_1h"] = _clean_numeric(df[col_precip]).fillna(0.0)
        out["temperature_c"] = _clean_numeric(df[col_temp])
        out["humidity_pct"] = _clean_numeric(df[col_hum])
        if col_wind:
            out["wind_speed_ms"] = _clean_numeric(df[col_wind])
        else:
            out["wind_speed_ms"] = np.nan
        if col_solar:
            out["solar_rad_kj"] = _clean_numeric(df[col_solar])
        else:
            out["solar_rad_kj"] = np.nan

        filename = filepath.name
        match = STATION_CODE_RE.search(filename)
        station_id = (meta_code or (match.group(1) if match else "")).strip().upper()
        uf = (meta_uf or "").strip().upper()
        if len(uf) != 2:
            parts = filename.split("_")
            uf = next((p for p in parts if len(p) == 2 and p.isupper()), "BR")

        out["station_id"] = station_id
        out["uf"] = uf
        out["station_name"] = (meta_name or "").strip()
        out["source_file"] = filename

        out["temperature_c"] = (
            out["temperature_c"].interpolate(method="linear").ffill().bfill().fillna(24.0)
        )
        out["humidity_pct"] = (
            out["humidity_pct"].interpolate(method="linear").ffill().bfill().fillna(70.0)
        )
        out["wind_speed_ms"] = (
            out["wind_speed_ms"]
            .interpolate(method="linear")
            .ffill()
            .bfill()
            .fillna(2.0)
            .clip(0.0, 50.0)
        )
        out["solar_rad_kj"] = (
            out["solar_rad_kj"].interpolate(method="linear").ffill().bfill().fillna(0.0).clip(0.0, 5000.0)
        )
        out["precip_1h"] = out["precip_1h"].clip(0.0, 300.0)

        out = out.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates(
            subset=["datetime"]
        )
        if len(out) < 10 or not station_id:
            return None
        return out.reset_index(drop=True)
    except Exception:
        return None


def list_inmet_files(inmet_dir: Path) -> list[Path]:
    files = list(inmet_dir.glob("*.csv")) + list(inmet_dir.glob("*.CSV"))
    return sorted({p.resolve() for p in files}, key=lambda p: p.name)


def station_id_from_filename(path: Path) -> str | None:
    match = STATION_CODE_RE.search(path.name)
    return match.group(1).upper() if match else None


def select_station_files(
    files: list[Path],
    n_stations: int,
    years: list[int],
) -> dict[str, list[Path]]:
    """Deterministic map station_id -> files (sorted WMO code, first n_stations)."""
    year_tokens = {str(y) for y in years}
    by_station: dict[str, list[Path]] = {}
    for path in files:
        if years and not any(tok in path.name for tok in year_tokens):
            continue
        sid = station_id_from_filename(path)
        if not sid:
            continue
        by_station.setdefault(sid, []).append(path)

    ordered = sorted(by_station.keys())[:n_stations]
    return {sid: sorted(by_station[sid], key=lambda p: p.name) for sid in ordered}


def _event_bonus_mask(
    dates: pd.Series,
    ufs: pd.Series,
    events: list[dict[str, str]],
    bonus: float,
) -> np.ndarray:
    out = np.zeros(len(dates), dtype=np.float64)
    if not events:
        return out
    date_str = dates.dt.strftime("%Y-%m-%d").to_numpy()
    uf_arr = ufs.to_numpy()
    for event in events:
        hit = (uf_arr == event["uf"]) & (date_str >= event["start"]) & (date_str <= event["end"])
        out = np.where(hit, bonus, out)
    return out


def process_station(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Apply hydrology, Poisson belief, and physical labels inside one station."""
    df = df.sort_values("datetime").reset_index(drop=True)
    soil_cfg = cfg["soil"]
    solar_cfg = cfg["solar"]
    bat_cfg = cfg["battery"]
    poi = cfg["poisson"]
    lab = cfg["labels"]
    bdi = cfg["bdi"]

    precip = df["precip_1h"].to_numpy(dtype=np.float64)
    temp = df["temperature_c"].to_numpy(dtype=np.float64)
    n = len(df)

    precip_6h = pd.Series(precip).rolling(window=6, min_periods=1).sum().to_numpy()
    precip_24h = pd.Series(precip).rolling(window=24, min_periods=1).sum().to_numpy()

    sm = np.empty(n, dtype=np.float64)
    current = float(soil_cfg["initial_pct"])
    for t in range(n):
        wetting = precip[t] * float(soil_cfg["wet_gain"])
        drying = float(soil_cfg["evap_base"]) + float(soil_cfg["evap_temp_coef"]) * (
            temp[t] - float(soil_cfg["evap_temp_ref_c"])
        )
        current = float(
            np.clip(
                current + wetting - drying,
                float(soil_cfg["clip_min"]),
                float(soil_cfg["clip_max"]),
            )
        )
        sm[t] = current

    solar_mw = np.clip(
        df["solar_rad_kj"].to_numpy(dtype=np.float64) / float(solar_cfg["kj_to_wm2"]),
        0.0,
        float(solar_cfg["panel_peak_mw"]),
    )

    soc = np.empty(n, dtype=np.float64)
    current_soc = float(bat_cfg["initial_soc_pct"])
    cap = float(bat_cfg["capacity_mwh"])
    for t in range(n):
        p_out = float(bat_cfg["rain_mw"]) if precip[t] > 0 else float(bat_cfg["idle_mw"])
        current_soc = float(
            np.clip(
                current_soc + ((solar_mw[t] - p_out) / cap) * 100.0,
                float(bat_cfg["soc_clip_min"]),
                float(bat_cfg["soc_clip_max"]),
            )
        )
        soc[t] = current_soc

    bonus = np.zeros(n, dtype=np.float64)
    if poi.get("use_event_prior", False):
        events = cfg.get("event_prior") or (cfg.get("event_audit") or {}).get("events") or []
        bonus = _event_bonus_mask(
            df["datetime"],
            df["uf"],
            events,
            float(poi["event_bonus"]),
        )

    term_6h = float(poi["alpha"]) * (precip_6h / float(poi["p_crit_6h_mm"]))
    term_24h = float(poi["beta"]) * (precip_24h / float(poi["p_crit_24h_mm"]))
    term_sm = float(poi["gamma"]) * (sm / float(poi["sm_sat_pct"]))
    exponent = np.minimum(float(poi["exponent_clip"]), term_6h + term_24h + term_sm + bonus)
    lambda_t = np.maximum(0.0, float(poi["lambda_0"]) * (np.exp(exponent) - 1.0))
    lambda_cum = (
        pd.Series(lambda_t)
        .rolling(window=int(poi["horizon_hours"]), min_periods=1)
        .sum()
        .to_numpy()
    )
    p_risk = 1.0 - np.exp(-lambda_cum)

    landslide = (precip_24h >= float(lab["landslide"]["precip_24h_mm"])) & (
        sm >= float(lab["landslide"]["soil_moisture_pct"])
    )
    flood = (precip_6h >= float(lab["flood"]["precip_6h_mm"])) | (
        precip >= float(lab["flood"]["precip_1h_mm"])
    )
    risk_label = np.zeros(n, dtype=np.int32)
    risk_label[landslide] = 1
    risk_label[flood] = 2
    risk_label[landslide & flood] = 3

    # BDI policy replay from Poisson belief + SoC. Not the classification target.
    hazard_thr = float(bdi["hazard_p_risk"])
    low_soc = float(bat_cfg["low_soc_pct"])
    bdi_action = np.zeros(n, dtype=np.int32)
    alert = p_risk >= hazard_thr
    bdi_action[alert] = 2
    bdi_action[alert & (soc < low_soc)] = 1

    out = df.copy()
    out["precip_6h"] = np.round(precip_6h, 2)
    out["precip_24h"] = np.round(precip_24h, 2)
    out["soil_moisture"] = np.round(sm, 2)
    out["solar_power_mw"] = np.round(solar_mw, 2)
    out["battery_soc_percent"] = np.round(soc, 2)
    out["poisson_lambda"] = np.round(lambda_t, 6)
    out["poisson_p_risk"] = np.round(p_risk, 4)
    out["risk_label"] = risk_label
    out["bdi_action"] = bdi_action
    return out


def build_dataset(
    cfg: dict[str, Any],
    inmet_dir: Path | None = None,
    max_stations: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse INMET, process per station, concatenate. Returns (table, manifest)."""
    inmet_dir = inmet_dir or resolve_inmet_dir()
    n_stations = int(max_stations if max_stations is not None else cfg["n_stations"])
    files = list_inmet_files(inmet_dir)
    if not files:
        raise RuntimeError(f"No INMET CSV files in {inmet_dir}")

    selected = select_station_files(
        files,
        n_stations=10_000,
        years=list(cfg.get("years") or []),
    )
    if not selected:
        raise RuntimeError("Station selection produced an empty set.")

    frames: list[pd.DataFrame] = []
    manifest_files: list[dict[str, str]] = []
    for station_id, paths in selected.items():
        if len(frames) >= n_stations:
            break
        parts = []
        pending_manifest: list[dict[str, str]] = []
        for path in paths:
            parsed = parse_inmet_file(path)
            if parsed is None:
                continue
            parts.append(parsed)
            pending_manifest.append(
                {
                    "station_id": station_id,
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "n_rows": str(len(parsed)),
                }
            )
        if not parts:
            continue
        station_df = pd.concat(parts, ignore_index=True)
        station_df = station_df.sort_values("datetime").drop_duplicates(subset=["datetime"])
        if len(station_df) < int(cfg["min_hours_per_station"]):
            continue
        frames.append(process_station(station_df, cfg))
        manifest_files.extend(pending_manifest)

    if not frames:
        raise RuntimeError("No station passed the minimum-hour filter.")

    table = pd.concat(frames, ignore_index=True)
    table = table.sort_values(["station_id", "datetime"]).reset_index(drop=True)

    counts = table["risk_label"].value_counts().sort_index()
    summary = {
        "n_rows": int(len(table)),
        "n_stations": int(table["station_id"].nunique()),
        "label_counts": {int(k): int(v) for k, v in counts.items()},
        "label_share_pct": {
            int(k): round(100.0 * v / len(table), 4) for k, v in counts.items()
        },
        "note": (
            "risk_label is physical-threshold only. poisson_p_risk is a belief "
            "feature and is not used as ground truth."
        ),
    }
    manifest = {
        "n_stations": summary["n_stations"],
        "n_rows": summary["n_rows"],
        "inmet_dir": str(inmet_dir),
        "files": manifest_files,
        "summary": summary,
    }
    return table, manifest


def save_dataset(
    table: pd.DataFrame,
    manifest: dict[str, Any],
    processed_dir: Path | None = None,
    stations_json: Path | None = None,
) -> Path:
    processed_dir = processed_dir or PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_csv = processed_dir / "hourly_dataset.csv"
    table.to_csv(out_csv, index=False)
    summary_path = processed_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(manifest["summary"], indent=2), encoding="utf-8")
    dest = stations_json or STATIONS_JSON
    dest.write_text(json.dumps({k: manifest[k] for k in ("n_stations", "n_rows", "files")}, indent=2), encoding="utf-8")
    return out_csv
