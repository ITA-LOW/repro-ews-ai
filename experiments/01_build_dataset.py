#!/usr/bin/env python3
"""Build the per-station hourly table used by every later experiment.

Usage (from repository root):

    python experiments/01_build_dataset.py
    python experiments/01_build_dataset.py --max-stations 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ews_ai.dataset import build_dataset, save_dataset  # noqa: E402
from ews_ai.paths import RESULTS_DIR, load_experiment_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the EWS-AI hourly dataset.")
    parser.add_argument(
        "--max-stations",
        type=int,
        default=None,
        help="Override n_stations from configs/experiment.yaml (smoke tests).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to a YAML config (defaults to configs/experiment.yaml).",
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    table, manifest = build_dataset(cfg, max_stations=args.max_stations)
    out_csv = save_dataset(table, manifest)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "dataset_summary.json"
    summary_path.write_text(json.dumps(manifest["summary"], indent=2), encoding="utf-8")

    print(f"Wrote {out_csv}  ({manifest['n_rows']} rows, {manifest['n_stations']} stations)")
    print(f"Label counts: {manifest['summary']['label_counts']}")
    print(f"Manifest: configs/stations.json")
    print(f"Summary:  {summary_path}")
    print(manifest["summary"]["note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
