# Data

## INMET hourly files (required for experiment 01)

This package does not ship the raw meteorological CSVs.

1. Download the official yearly archives from the Brazilian National Institute
   of Meteorology (INMET) BDMEP / dados históricos portal
   (`https://portal.inmet.gov.br/dadoshistoricos`).
2. Extract the automatic-station CSVs for 2023 and 2024 into `data/inmet/`,
   **or** point the environment variable at an existing folder:

   ```bash
   export EWS_AI_INMET_DIR=/absolute/path/to/inmet_data
   ```

3. Expected filename pattern (INMET default):

   `INMET_<region>_<UF>_<WMO>_<NAME>_01-01-YYYY_A_31-12-YYYY.CSV`

Experiment 01 selects the first `n_stations` WMO codes in sorted order
(see `configs/experiment.yaml`) and writes SHA-256 hashes of every file
it actually read into `configs/stations.json`.

If this checkout sits next to the private lab tree, experiment 01 also
looks at `../dataset/inmet_data` so local development does not need a copy.

## Processed table (generated)

`data/processed/hourly_dataset.csv` is produced by experiment 01. It is
gitignored because of size. Columns include:

`station_id`, `datetime`, `uf`, `precip_1h`, `precip_6h`, `precip_24h`,
`temperature_c`, `humidity_pct`, `soil_moisture`, `solar_power_mw`,
`battery_soc_percent`, `poisson_lambda`, `poisson_p_risk`, `risk_label`,
`bdi_action`.
