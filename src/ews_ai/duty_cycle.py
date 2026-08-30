"""Sub-hour duty-cycle energy. Weather stays hourly (INMET); wakes do not.

FSM: one wake per hour.
BDI: T_sleep from the firmware rule (120 / 300 / 3600 s).
More wakes are sampling density, not extra hourly recall.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

POLICY_FSM_FIXED = "fsm_fixed_3600s"
POLICY_BDI_TX = "bdi_adaptive_tx_per_wake"
POLICY_BDI_RATE = "bdi_adaptive_tx_hourly"


def sleep_seconds(hazard: bool, precip_1h: float, energy_cfg: dict[str, Any]) -> int:
    if hazard:
        return int(energy_cfg.get("t_sleep_hazard_s", 120))
    if precip_1h > float(energy_cfg.get("rain_wake_mm", 5.0)):
        return int(energy_cfg.get("t_sleep_rain_s", 300))
    return int(energy_cfg.get("t_sleep_calm_s", 3600))


def wakes_per_hour(t_sleep_s: int) -> int:
    return max(1, int(round(3600.0 / float(t_sleep_s))))


def hour_energy_mwh(
    n_wakes: int,
    n_tx: int,
    radio: str,
    energy_cfg: dict[str, Any],
) -> float:
    """Energy of one clock hour. Returns mWh (= mean mW over that hour)."""
    t_awake = n_wakes * float(energy_cfg.get("t_awake_s", 2.0))
    if radio == "espnow":
        t_tx = n_tx * float(energy_cfg.get("t_espnow_tx_s", 0.05))
        p_tx = float(energy_cfg.get("p_espnow_tx_mw", 180.0))
    else:
        t_tx = n_tx * float(energy_cfg.get("t_lora_tx_s", 1.0))
        p_tx = float(energy_cfg.get("p_lora_tx_mw", 400.0))
    t_sleep = max(0.0, 3600.0 - t_awake - t_tx)
    e = (
        float(energy_cfg.get("p_awake_mw", 80.0)) * t_awake
        + p_tx * t_tx
        + float(energy_cfg["p_deep_sleep_mw"]) * t_sleep
    ) / 3600.0
    return float(e)


def plan_hour(
    policy: str,
    hazard: bool,
    precip_1h: float,
    soc: float,
    energy_cfg: dict[str, Any],
    battery_cfg: dict[str, Any],
) -> tuple[int, int, int, str]:
    """Return (t_sleep_s, n_wakes, n_tx, radio)."""
    low = soc < float(battery_cfg["low_soc_pct"])
    assume_nb = bool(energy_cfg.get("assume_neighbor", True))
    radio = "espnow" if (hazard and low and assume_nb and policy != POLICY_FSM_FIXED) else "lora"

    if policy == POLICY_FSM_FIXED:
        t_sleep = int(energy_cfg.get("t_sleep_calm_s", 3600))
        n_wakes = 1
        n_tx = 1 if hazard else 0
        return t_sleep, n_wakes, n_tx, radio

    t_sleep = sleep_seconds(hazard, precip_1h, energy_cfg)
    n_wakes = wakes_per_hour(t_sleep)
    if not hazard:
        n_tx = 0
    elif policy == POLICY_BDI_RATE:
        n_tx = 1
    else:
        n_tx = n_wakes
    return t_sleep, n_wakes, n_tx, radio


def simulate_window(
    hours: pd.DataFrame,
    hazard: np.ndarray,
    cfg: dict[str, Any],
    policies: tuple[str, ...] = (POLICY_FSM_FIXED, POLICY_BDI_TX, POLICY_BDI_RATE),
    initial_soc: float | None = None,
) -> pd.DataFrame:
    """Hourly SoC / power / wake counts for each policy on one station window."""
    energy_cfg = cfg["energy"]
    battery_cfg = cfg["battery"]
    cap = float(energy_cfg["lifepo4_capacity_mwh"])
    soc0 = float(initial_soc if initial_soc is not None else battery_cfg["initial_soc_pct"])
    soc_min = float(battery_cfg["soc_clip_min"])
    soc_max = float(battery_cfg["soc_clip_max"])
    solar = hours["solar_power_mw"].to_numpy(dtype=np.float64)
    precip = hours["precip_1h"].to_numpy(dtype=np.float64)
    times = hours["datetime"].to_numpy()
    sid = str(hours["station_id"].iloc[0])

    rows: list[dict[str, Any]] = []
    for policy in policies:
        soc = soc0
        for t in range(len(hours)):
            dead = soc <= soc_min
            haz = bool(hazard[t]) and not dead
            t_sleep, n_wakes, n_tx, radio = plan_hour(
                policy, haz, float(precip[t]), soc, energy_cfg, battery_cfg
            )
            if dead:
                n_wakes, n_tx, t_sleep = 0, 0, 3600
            e_mwh = hour_energy_mwh(max(n_wakes, 1) if not dead else 0, n_tx, radio, energy_cfg)
            if dead:
                e_mwh = float(energy_cfg["p_deep_sleep_mw"])
            soc = float(
                np.clip(soc + ((solar[t] - e_mwh) / cap) * 100.0, soc_min, soc_max)
            )
            rows.append(
                {
                    "datetime": times[t],
                    "station_id": sid,
                    "policy": policy,
                    "hazard": int(hazard[t]),
                    "dead": int(dead),
                    "t_sleep_s": t_sleep,
                    "n_wakes": n_wakes,
                    "n_tx": n_tx,
                    "radio": radio,
                    "power_mw": round(e_mwh, 4),
                    "soc_pct": round(soc, 3),
                    "solar_mw": round(float(solar[t]), 2),
                    "precip_1h": round(float(precip[t]), 2),
                    "precip_6h": round(float(hours["precip_6h"].iloc[t]), 2),
                }
            )
    return pd.DataFrame(rows)


def summarize(trace: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sid, policy), g in trace.groupby(["station_id", "policy"], sort=True):
        rows.append(
            {
                "station_id": sid,
                "policy": policy,
                "n_hours": int(len(g)),
                "n_hazard_hours": int(g["hazard"].sum()),
                "n_wakes": int(g["n_wakes"].sum()),
                "n_tx": int(g["n_tx"].sum()),
                "mean_power_mw": round(float(g["power_mw"].mean()), 4),
                "energy_mwh": round(float(g["power_mw"].sum()), 2),
                "min_soc_pct": round(float(g["soc_pct"].min()), 2),
                "final_soc_pct": round(float(g["soc_pct"].iloc[-1]), 2),
                "hours_dead": int(g["dead"].sum()),
                "wakes_per_hazard_hour": (
                    round(float(g.loc[g["hazard"] == 1, "n_wakes"].mean()), 2)
                    if (g["hazard"] == 1).any()
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)
