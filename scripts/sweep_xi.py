#!/usr/bin/env python3
"""Reproduce the smoothed-Burg parameter study in Figure 2."""

import argparse
import sys
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.algorithms import BPGM
from src.data import generate_matrix, generate_standard_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=500)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", default="results/figures/figure_2_xi_sweep.pdf")
    return parser.parse_args()


def main():
    args = parse_args()
    N = 500
    np.random.seed(args.seed)
    A = generate_matrix(args.M, N)
    _, y = generate_standard_data(A, b=1e-3, Q=100)
    x0 = np.random.rand(N, 1) * 1e4

    # The paper selects xi among 30 candidate values.
    xi_values = np.geomspace(0.1, 2.0, num=30)
    histories = []
    for xi in xi_values:
        _, values, _, _ = BPGM(
            x_0=x0,
            y=y,
            n_iter=args.iterations,
            A=A,
            b_mat=1e-3,
            xi=xi,
            phi="burg",
            reg=args.lam > 0,
            lam=args.lam,
        )
        histories.append(values)

    reference_value = min(values[-1] for values in histories)
    fig, ax = plt.subplots(figsize=(11, 6))
    norm = colors.LogNorm(vmin=xi_values.min(), vmax=xi_values.max())
    cmap = plt.colormaps["jet"]
    for xi, values in zip(xi_values, histories):
        gap = np.maximum(values - reference_value, 1e-15)
        ax.plot(gap, color=cmap(norm(xi)), lw=2)

    ax.set_xlabel("Iterations", fontsize=20)
    ax.set_ylabel(r"$J(\mathbf{x}^k)-\bar{J}$", fontsize=20)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-14)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    colorbar = fig.colorbar(scalar_map, ax=ax, pad=0.02, aspect=30)
    colorbar.set_label(r"$\xi$", fontsize=20, rotation=0, labelpad=15)
    colorbar.locator = ticker.LogLocator(base=10.0)
    colorbar.update_ticks()
    fig.tight_layout()

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        ROOT / "results/data/figure_2_xi_sweep.npz",
        xi_values=xi_values,
        histories=np.asarray(histories),
        reference_value=reference_value,
    )


if __name__ == "__main__":
    main()

