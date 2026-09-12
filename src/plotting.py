"""Plotting utilities matching the style of the paper figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


METHOD_STYLE = {
    "burg": (r"$\varphi_{\mathrm{B}}$", "crimson"),
    "smoothed_burg": (r"$\varphi_{\mathrm{sB}}$", "orange"),
    "l2": (r"$\varphi_{\ell_2}$", "royalblue"),
    "rl": ("RL", "green"),
}


def plot_objective_gaps(samples, output_file, reference_value=None):
    """Plot medians and 1st--99th percentile bands over trials."""
    if reference_value is None:
        reference_value = min(values[-1] for runs in samples.values() for values in runs)

    fig, ax = plt.subplots(figsize=(12, 7))
    for method, runs in samples.items():
        label, color = METHOD_STYLE[method]
        gaps = np.asarray(runs) - reference_value
        gaps = np.maximum(gaps, 1e-15)
        iterations = np.arange(gaps.shape[1])
        median = np.percentile(gaps, 50, axis=0)
        low = np.percentile(gaps, 1, axis=0)
        high = np.percentile(gaps, 99, axis=0)
        ax.semilogy(iterations, median, color=color, lw=2, label=label)
        ax.fill_between(iterations, low, high, color=color, alpha=0.15)

    ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.set_xlabel("Iterations", fontsize=28)
    ax.set_ylabel(r"$J(\mathbf{x}^k)-\bar{J}$", fontsize=28)
    ax.set_ylim(bottom=1e-14)
    ax.tick_params(axis="both", which="major", labelsize=24)
    ax.grid(True, which="both", ls="-", alpha=0.1)
    ax.legend(loc="best", fontsize=22)
    fig.tight_layout()

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)

