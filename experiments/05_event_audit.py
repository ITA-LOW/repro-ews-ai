#!/usr/bin/env python3
"""Retrospective audit of publicly reported hazard episodes.

Does not change labels or λ(t). Reads hourly_dataset.csv, pulls preferred
INMET stations if the 100-station training set misses that UF, and asks
whether Poisson P_risk crossed 0.20 during each documented window.

Usage (from repository root, after experiment 01):

    python experiments/05_event_audit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ews_ai.event_audit import load_processed_table, run_audit  # noqa: E402
from ews_ai.paths import RESULTS_DIR, load_experiment_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Held-out public-event audit.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    if cfg.get("poisson", {}).get("use_event_prior"):
        print(
            "WARNING: poisson.use_event_prior is true. "
            "This audit is only valid when dates are not injected into λ(t)."
        )

    processed = load_processed_table(args.dataset)
    table, summary = run_audit(cfg, processed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "event_audit.csv"
    json_path = RESULTS_DIR / "event_audit_summary.json"
    table.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(table.to_string(index=False))
    print()
    print(
        f"Detected {summary['n_detected']}/{summary['n_scored']} scored events "
        f"(uncovered={summary['n_uncovered']}). "
        f"Median lead vs peak P_6h: {summary['median_lead_hours_vs_peak_p6h']} h"
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(summary["protocol"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
