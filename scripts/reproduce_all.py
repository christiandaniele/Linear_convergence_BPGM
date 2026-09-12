#!/usr/bin/env python3
"""Generate the principal experiment families from Figures 3--6."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_experiment.py"

EXPERIMENTS = (
    ("figure_3_rank_deficient_unregularized", 400, 0.0, "boundary"),
    ("figure_4a_full_rank_unregularized_interior", 500, 0.0, "interior"),
    ("figure_4b_full_rank_unregularized_boundary", 500, 0.0, "boundary"),
    ("figure_5_rank_deficient_regularized", 400, 0.1, "boundary"),
    ("figure_6a_full_rank_regularized_interior", 500, 0.1, "interior"),
    ("figure_6b_full_rank_regularized_boundary", 500, 0.1, "boundary"),
)


def main():
    for name, measurements, lam, location in EXPERIMENTS:
        subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--M",
                str(measurements),
                "--lambda",
                str(lam),
                "--solution",
                location,
                "--name",
                name,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()

