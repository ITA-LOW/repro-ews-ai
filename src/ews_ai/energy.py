"""Trace-driven 10-year energy simulation. Datasheet mW, not a field test.

FSM-LoRa and BDI share the same CART detector and the same hourly wake.
The only difference is the radio plan.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

POLICY_FSM_LORA = "fsm_lora"
POLICY_BDI = "bdi"
POLICY_BDI_THRIFTY = "bdi_thrifty"
POLICY_ALWAYS_ON = "always_on_not_fsm"

ACTION_SLEEP = 0
ACTION_ESPNOW = 1
ACTION_LORA = 2


def tile_hours(values: np.ndarray, n_hours: int) -> np.ndarray:
    if len(values) == 0:
        raise ValueError("empty station trace")
    reps = int(np.ceil(n_hours / len(values)))
    return np.tile(values, reps)[:n_hours]


def policy_action(
    name: str,
    hazard: bool,
    soc: float,
    low_soc_pct: float,
    assume_neighbor: bool,
) -> int:
    if name == POLICY_ALWAYS_ON:
        return ACTION_LORA
    if not hazard:
        return ACTION_SLEEP
    if name == POLICY_FSM_LORA:
        return ACTION_LORA
    if name == POLICY_BDI:
        if soc < low_soc_pct and assume_neighbor:
            return ACTION_ESPNOW
        return ACTION_LORA
    if name == POLICY_BDI_THRIFTY:
        # Refuse LoRa when the buffer cannot pay. Neighbor → ESP-NOW; else store.
        if soc < low_soc_pct:
            if assume_neighbor:
                return ACTION_ESPNOW
            return ACTION_SLEEP
        return ACTION_LORA
    raise ValueError(f"unknown policy {name}")


def power_mw(action: int, energy_cfg: dict[str, Any], policy: str | None = None) -> float:
    if policy == POLICY_ALWAYS_ON:
        return float(energy_cfg["p_continuous_tx_mw"])
    if action == ACTION_SLEEP:
        return float(energy_cfg["p_deep_sleep_mw"])
    if action == ACTION_ESPNOW:
        return float(energy_cfg["p_espnow_mw"])
    if action == ACTION_LORA:
        return float(energy_cfg["p_lora_mw"])
    raise ValueError(action)


def simulate_trace(
    solar_mw: np.ndarray,
    hazard: np.ndarray,
    energy_cfg: dict[str, Any],
    battery_cfg: dict[str, Any],
    policy: str,
    years: int | None = None,
    assume_neighbor: bool | None = None,
) -> dict[str, float | int | str]:
    """Replay one station. Solar/hazard are hourly. Results are simulated."""
    years = int(years if years is not None else energy_cfg["simulation_years"])
    n_hours = years * 8760
    solar = tile_hours(np.asarray(solar_mw, dtype=np.float64), n_hours)
    haz = tile_hours(np.asarray(hazard, dtype=bool), n_hours)

    cap0 = float(energy_cfg["lifepo4_capacity_mwh"])
    soc = float(battery_cfg["initial_soc_pct"])
    soc_min = float(battery_cfg["soc_clip_min"])
    soc_max = float(battery_cfg["soc_clip_max"])
    low_soc = float(battery_cfg["low_soc_pct"])
    assume_nb = (
        bool(energy_cfg.get("assume_neighbor", True))
        if assume_neighbor is None
        else bool(assume_neighbor)
    )
    solar_deg = float(energy_cfg["solar_degradation_per_year"])
    bat_deg = float(energy_cfg["battery_degradation_per_year"])
    p_sleep = float(energy_cfg["p_deep_sleep_mw"])
    p_esp = float(energy_cfg["p_espnow_mw"])
    p_lora = float(energy_cfg["p_lora_mw"])
    p_always = float(energy_cfg["p_continuous_tx_mw"])
    year = np.arange(n_hours, dtype=np.int32) // 8760
    harvest_all = solar * ((1.0 - solar_deg) ** year)
    cap_all = cap0 * ((1.0 - bat_deg) ** year)

    energy_out = 0.0
    hours_alive = 0
    hours_dead = 0
    missed = 0
    n_lora = 0
    n_esp = 0
    n_sleep = 0
    n_hazard = int(haz.sum())
    n_skipped = 0
    hours_low_soc = 0

    for t in range(n_hours):
        is_haz = bool(haz[t])
        dead = soc <= soc_min
        if dead:
            action = ACTION_SLEEP
            hours_dead += 1
            if is_haz:
                missed += 1
        else:
            action = policy_action(policy, is_haz, soc, low_soc, assume_nb)
            hours_alive += 1
            if soc < low_soc:
                hours_low_soc += 1
            if is_haz and action == ACTION_SLEEP:
                n_skipped += 1

        if action == ACTION_LORA:
            n_lora += 1
        elif action == ACTION_ESPNOW:
            n_esp += 1
        else:
            n_sleep += 1

        if policy == POLICY_ALWAYS_ON:
            p_load = p_always
        elif action == ACTION_SLEEP:
            p_load = p_sleep
        elif action == ACTION_ESPNOW:
            p_load = p_esp
        else:
            p_load = p_lora
        energy_out += p_load
        soc += ((harvest_all[t] - p_load) / cap_all[t]) * 100.0
        if soc < soc_min:
            soc = soc_min
        elif soc > soc_max:
            soc = soc_max

    return {
        "policy": policy,
        "assume_neighbor": assume_nb,
        "capacity_mwh": cap0,
        "n_hours": n_hours,
        "years": years,
        "uptime_pct": round(100.0 * hours_alive / n_hours, 4),
        "mean_power_mw": round(energy_out / n_hours, 4),
        "hours_dead": hours_dead,
        "n_hazard": n_hazard,
        "missed_hazard_while_dead": missed,
        "n_skipped_warning": n_skipped,
        "n_lora": n_lora,
        "n_espnow": n_esp,
        "n_sleep": n_sleep,
        "hours_low_soc": hours_low_soc,
        "final_soc_pct": round(soc, 2),
        "note": "Simulated from INMET irradiance and datasheet-mean mW. Not a field measurement.",
    }


def simulate_dataset(
    df: pd.DataFrame,
    hazard: np.ndarray,
    cfg: dict[str, Any],
    policies: tuple[str, ...] = (POLICY_FSM_LORA, POLICY_BDI, POLICY_ALWAYS_ON),
    assume_neighbor: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-station then per-policy means."""
    energy_cfg = cfg["energy"]
    battery_cfg = cfg["battery"]
    years = int(energy_cfg["simulation_years"])
    rows: list[dict[str, Any]] = []
    solar = df["solar_power_mw"].to_numpy(dtype=np.float64)
    stations = df["station_id"].to_numpy()

    for sid in sorted(df["station_id"].unique()):
        mask = stations == sid
        for policy in policies:
            rec = simulate_trace(
                solar[mask],
                hazard[mask],
                energy_cfg,
                battery_cfg,
                policy,
                years=years,
                assume_neighbor=assume_neighbor,
            )
            rec["station_id"] = sid
            rows.append(rec)

    detail = pd.DataFrame(rows)
    summary_rows = []
    for policy in policies:
        sub = detail[detail["policy"] == policy]
        assume_val = bool(sub["assume_neighbor"].iloc[0]) if len(sub) else False
        cap_val = float(sub["capacity_mwh"].iloc[0]) if len(sub) else 0.0
        summary_rows.append(
            {
                "policy": policy,
                "assume_neighbor": assume_val,
                "capacity_mwh": cap_val,
                "n_stations": int(sub["station_id"].nunique()),
                "mean_uptime_pct": round(float(sub["uptime_pct"].mean()), 4),
                "min_uptime_pct": round(float(sub["uptime_pct"].min()), 4),
                "mean_power_mw": round(float(sub["mean_power_mw"].mean()), 4),
                "mean_missed_hazard_while_dead": round(
                    float(sub["missed_hazard_while_dead"].mean()), 2
                ),
                "mean_n_skipped_warning": round(
                    float(sub["n_skipped_warning"].mean()), 2
                ),
                "mean_n_lora": round(float(sub["n_lora"].mean()), 1),
                "mean_n_espnow": round(float(sub["n_espnow"].mean()), 1),
                "mean_hours_low_soc": round(float(sub["hours_low_soc"].mean()), 1),
                "same_cart_detector": True,
                "same_hourly_wake": True,
                "simulated": True,
            }
        )
    return detail, pd.DataFrame(summary_rows)
