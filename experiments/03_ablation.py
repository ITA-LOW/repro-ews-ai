#!/usr/bin/env python3
"""Table 4 — component ablation (window, Poisson weighting, BDI policy).

Not implemented yet. Recall will be measured, not hardcoded. Energy of each
variant will come from the controller actually being ablated, not from a
constant milliwatt assigned after the fact.

Usage (after experiment 01):

    python experiments/03_ablation.py
"""

from __future__ import annotations


def main() -> int:
    print(
        "experiments/03_ablation.py is a stub.\n"
        "It will emit results/table4_ablation.csv matching the manuscript.\n"
        "Run experiments/01_build_dataset.py first."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
