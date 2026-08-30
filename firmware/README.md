# Firmware

Bare-metal EWS-AI node for the Heltec WiFi LoRa 32 **V3** (ESP32-S3):
unscaled CART perception artifact, `embedded-bdi` reasoning cycle,
ESP-NOW cognitive network artifact, PSO-MPPT on Core 0.

Generate the tree header first (from the package root):

```bash
python experiments/06_export_cart.py
cd firmware && pio run -e heltec_wifi_lora_32_V3
```

`pio run` prints Flash and RAM. Last measured build
(`results/firmware_footprint.json`):

- Image: **741 089 B flash**, **45 300 B RAM** (Heltec V3, Arduino + WiFi + OLED + BDI + Poisson watch)
- CART table: **1 812 B** (`CART_NODES`, 151 nodes). `cart_predict` is 68 B of text.
- TFLite 64 KB arena is **not** allocated.

## Beliefs on the device (matches the manuscript)

- **WATCH** (`+poisson_watch` / `-poisson_watch`): rising/falling edge of
  \(P_{\mathrm{risk}}\ge 0.20\). Same \(\lambda(t)\) constants as
  `configs/experiment.yaml` / `src/ews_ai/dataset.py`. No event-date bonus.
  One placeholder packet each (`action_code` 5 / 6).
- **WARNING** (`+tinyml_hazard`): CART class \(\neq 0\). Unchanged plans
  (ESP-NOW / LoRa / DART placeholder).
- Duty cycle: CART hazard 120 s, watch 300 s, else 3600 s.

`src/poisson_belief.h` is the on-device \(\lambda(t)\) table. Combining
Poisson with CART does **not** change the CART class.

## What this build does and does not prove

- **Does:** the agent compiles with static allocation, the BDI plan library
  is the same structure described in the manuscript, CART nodes are a
  static table in `src/cart_model.h`.
- **Does not:** reproduce Table 2 recall, 10-year uptime, or a measured
  current waveform. Those come from `experiments/01`–`08`.
- LoRa TX is a serial placeholder. Satellite contact windows are three
  hardcoded intervals, not a live FLoRaSat run.
- `action_deep_sleep` does not call `esp_deep_sleep_start()` on the bench
  OLED path (commented, so the display stays on).

Libraries live in `firmware/lib/` (no path back to the private monorepo).
