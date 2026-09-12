#!/usr/bin/env python3
"""Run one of the four main experimental settings."""

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.experiments import ExperimentConfig, run_experiment, save_result
from src.plotting import plot_objective_gaps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--lambda", dest="lam", type=float, required=True)
    parser.add_argument("--solution", choices=("boundary", "interior"), default="boundary")
    parser.add_argument("--xi", type=float, default=0.8)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200_000)
    parser.add_argument("--plot-iterations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig(
        M=args.M,
        lam=args.lam,
        xi=args.xi,
        n_trials=args.trials,
        n_iter=args.iterations,
        plot_iterations=args.plot_iterations,
        seed=args.seed,
        solution_location=args.solution,
    )
    result = run_experiment(config)
    save_result(result, REPOSITORY_ROOT / "results/data" / f"{args.name}.npz")
    plot_objective_gaps(
        result["samples"],
        REPOSITORY_ROOT / "results/figures" / f"{args.name}.pdf",
        result["reference_value"],
    )


if __name__ == "__main__":
    main()

