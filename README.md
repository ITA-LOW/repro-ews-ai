# EWS-AI

This repository is the reviewer-facing artifact for the ICAART manuscript
**EWS-AI** (energy-aware embedded BDI + TinyML for disaster early warning).
It is written so a single command sequence rebuilds the processed dataset
from official INMET hourly files. Experiment 02 writes Table 2.
Experiment 04 writes the simulated FSM-vs-BDI energy table. Experiment 06
exports the on-device CART.

Author identities are withheld for double-blind review.

## What a reviewer runs

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Official INMET hourly CSVs: set the directory, or copy them into data/inmet/
export EWS_AI_INMET_DIR=/path/to/inmet_csv_folder

python experiments/01_build_dataset.py
python experiments/05_event_audit.py      # held-out public episodes
python experiments/02_classification.py   # Table 2 (needs TensorFlow for the 1D-CNN)
python experiments/06_export_cart.py      # unscaled CART → firmware/src/cart_model.h
python experiments/04_energy.py           # simulated FSM-LoRa vs BDI (same CART)
python experiments/09_energy_capacity.py  # capacity sweep: FSM vs BDI vs BDI-thrifty
python experiments/07_storm_duty.py       # RS 2024 wake-resolution duty cycle + plots
python experiments/08_cart_poisson_lead.py  # CART vs P_risk lead on real RS hours
python experiments/10_multievent_lead.py  # WATCH vs WARNING across all event-state stations
python experiments/03_ablation.py         # stub — next
# optional: cd firmware && pio run -e heltec_wifi_lora_32_V3
```

Experiment 01 writes:

| File | Role |
|---|---|
| `data/processed/hourly_dataset.csv` | Per-station hourly table |
| `data/processed/dataset_summary.json` | Row / station / label counts |
| `configs/stations.json` | Filenames + SHA-256 of every ingested CSV |
| `results/dataset_summary.json` | Same summary, for the paper trail |

Experiment 02 writes:

| File | Role |
|---|---|
| `results/table2_classification.csv` | Manuscript Table 2 (disaster recall / precision) |
| `results/table2_classification.json` | Per-family confusion counts and protocol |
| `results/scaler_train.json` | Train-only StandardScaler mean / scale |

Experiment 06 writes `firmware/src/cart_model.h` and `results/cart_export.json`.

Experiment 04 writes:

| File | Role |
|---|---|
| `results/energy_uptime.csv` | Mean simulated 10-year FSM vs BDI |
| `results/energy_uptime_by_station.csv` | Per-station traces |
| `results/energy_uptime.json` | Protocol + CART test metrics |

Experiment 09 writes a **capacity sweep** (9620 / 1500 / 500 / 200 mWh):
same CART, same hourly wake. BDI-thrifty refuses LoRa when SoC < 20%.
Always-on is not in that table.

| File | Role |
|---|---|
| `results/energy_capacity_sweep.csv` | Mean uptime / missed hazards / packets |
| `results/energy_capacity_sweep_by_station.csv` | Per-station traces |
| `results/energy_capacity_sweep.json` | Protocol |
| `results/figures/energy_capacity_sweep.png` | Uptime and missed-hazard curves |

Experiment 10 evaluates WATCH/WARNING rising-edge timing for every usable
station in each configured event state, excluding CART training stations.
Pairs already active at the seven-day analysis boundary are marked as
left-censored. Uncertainty is bootstrapped by event because stations within
one storm are spatially correlated; the analysis is also repeated at WATCH
thresholds 0.10, 0.20, 0.30, and 0.50.

| File | Role |
|---|---|
| `results/multievent_lead_by_station.csv` | Station--event timing and activation category |
| `results/multievent_lead_by_event.csv` | Event-level counts and lead summaries |
| `results/multievent_lead_summary.json` | Aggregate statistics and clustered interval |
| `results/multievent_threshold_sensitivity.csv` | Threshold, censoring, lead, and interval sensitivity |
| `results/figures/multievent_watch_warning.png` | Event-level WATCH-earlier proportions |

## Protocol (frozen in `configs/experiment.yaml`)

1. Stations are chosen by **sorted WMO code**, never by filesystem glob order.
2. Rain accumulation, soil moisture, Poisson intensity, and 6-hour tensors
   are computed **inside each station**. Series are never concatenated first.
3. `risk_label` uses **physical thresholds only** (24 h rain + soil for
   landslide; 6 h / 1 h rain for flood). `poisson_p_risk` is a belief
   variable. It is neither a classifier feature nor written into `y`.
4. `station_id`, `datetime`, and `uf` are kept so temporal and station
   hold-outs are possible.
5. Dated public episodes in `event_audit` are a **held-out checklist**.
   They are not written into λ(t) and not used as `risk_label`. Experiment
   05 reports event-level hit rate and lead time vs peak 6 h rain (UF-scale
   convenience sample, not a CEMADEN inventory).
6. Energy numbers (experiment 04) are **simulations** from datasheet-mean
   milliwatts and INMET irradiance, not benchtop or field measurements.
   FSM-LoRa and BDI share the same CART detector and the same hourly wake.
   They do not model reasoning-engine overhead. `always_on_not_fsm` is a
   no-duty-cycle reference, not an FSM.
7. Table 2 (experiment 02) uses a **temporal per-station** split, a scaler
   fit on train hours only, and binary disaster recall (any non-zero class).
   The 1D-CNN is a Keras comparison head. The **on-device** model is the
   unscaled CART from experiment 06.
8. On-device perception (experiment 06) is an unscaled CART. Thresholds
   stay in physical units. Its five environmental inputs exclude solar
   power and battery SoC. Flash/RAM come from `pio run`, not from a TFLite
   arena.

Label encoding: `0` none, `1` landslide, `2` flood, `3` multi-hazard.

`bdi_action` is the deterministic BDI policy replay
(`poisson_p_risk ≥ 0.20` → LoRa, or ESP-NOW if SoC < 20 %). It is for
energy experiments, not for Table 2.

## Layout

```
configs/experiment.yaml     frozen constants (seed, Poisson, mW, split)
configs/stations.json       written by experiment 01
src/ews_ai/dataset.py       INMET parse + per-station physics
experiments/01_…            dataset
experiments/05_…            held-out public-event audit
experiments/02_…            Table 2 (classification)
experiments/06_…            unscaled CART → firmware header
experiments/03_…            Table 4 (ablation)
experiments/04_…            simulated FSM vs BDI energy
experiments/09_…            capacity sweep (FSM vs BDI-thrifty)
experiments/10_…            all-station multi-event WATCH/WARNING timing
firmware/                   Heltec V3 + CART + embedded-bdi (optional `pio run`)
```

## Firmware (optional)

```bash
cd firmware && pio run -e heltec_wifi_lora_32_V3
```

This only proves the agent firmware **builds**. Current image: 740 993 B
flash, 45 340 B RAM; the 109-node CART table is estimated at 1 308 B.
WATCH and WARNING are both
in `main.cpp` (Poisson rising/falling \(P_{risk}\ge 0.20\); CART class
\(\neq 0\)). It does not reproduce classification recall or 10-year
uptime. LoRa TX and FLoRaSat contact windows in the current binary are
stubs; Deep Sleep is disabled on the bench OLED path. See
`firmware/README.md`.

## Data

INMET hourly files are public but are not redistributed here. See
`data/README.md`.
# repro-ews-ai
