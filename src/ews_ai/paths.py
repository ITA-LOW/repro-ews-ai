"""Repository paths and frozen YAML config."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"
EXPERIMENT_YAML = CONFIGS_DIR / "experiment.yaml"
STATIONS_JSON = CONFIGS_DIR / "stations.json"


def load_experiment_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or EXPERIMENT_YAML
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_inmet_dir() -> Path:
    """INMET CSVs: env, data/inmet, or the lab checkout next to this package."""
    env = os.environ.get("EWS_AI_INMET_DIR")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(DATA_DIR / "inmet")
    candidates.append(REPO_ROOT.parent / "dataset" / "inmet_data")
    for c in candidates:
        if not c.is_dir():
            continue
        if any(p.suffix.lower() == ".csv" for p in c.iterdir()):
            return c
    raise FileNotFoundError(
        "No INMET CSV directory found. Set EWS_AI_INMET_DIR or copy official "
        "INMET hourly files into data/inmet/. See data/README.md."
    )
