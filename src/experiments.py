"""High-level experiment runner for Figures 2--6."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .algorithms import BPGM
from .data import generate_interior_data, generate_matrix, generate_standard_data


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration of one experiment family.

    ``M < N`` selects the rank-deficient regime and ``lam > 0`` selects the
    regularized problem. Thus, M and lam are the only inputs needed to choose
    among the four main settings of the paper.
    """

    M: int
    lam: float
    N: int = 500
    Q: int = 100
    b: float = 1e-3
    cond_target: float = 50
    xi: float = 0.8
    n_iter: int = 200_000
    plot_iterations: int = 100_000
    n_trials: int = 30
    seed: int = 13
    solution_location: str = "boundary"

    @property
    def regularized(self):
        return self.lam > 0

    @property
    def rank_deficient(self):
        return self.M < self.N

    def validate(self):
        if self.M > self.N:
            raise ValueError("The paper assumes M <= N.")
        if self.lam < 0:
            raise ValueError("lam must be nonnegative.")
        if self.solution_location not in {"boundary", "interior"}:
            raise ValueError("solution_location must be 'boundary' or 'interior'.")
        if self.rank_deficient and self.solution_location == "interior":
            raise ValueError(
                "The paper does not use the interior construction in the "
                "rank-deficient experiments; see Remark 3.9."
            )


def run_experiment(config):
    """Run all four algorithms for one paper configuration."""
    config.validate()
    np.random.seed(config.seed)
    A = generate_matrix(config.M, config.N, config.cond_target)

    if config.solution_location == "interior":
        if config.regularized:
            xt, y = generate_interior_data(
                A, config.b, config.lam, Q=1, noise=False
            )
        else:
            # Appendix B.2.1.
            xt = np.ones((config.N, 1), dtype=np.float64)
            y = np.dot(A, xt) + config.b
    else:
        xt, y = generate_standard_data(A, config.b, config.Q)

    samples = {"burg": [], "smoothed_burg": [], "l2": [], "rl": []}
    final_iterates = {key: [] for key in samples}

    methods = {
        "burg": ("burg", 0.0),
        "smoothed_burg": ("burg", config.xi),
        "l2": ("l2", 0.0),
        "rl": ("RL", 0.0),
    }

    for _ in range(config.n_trials):
        x0 = np.random.uniform(1e-3, 1e4, size=(config.N, 1))
        for name, (phi, xi) in methods.items():
            x, values, _, _ = BPGM(
                x_0=x0,
                y=y,
                n_iter=config.n_iter,
                A=A,
                b_mat=config.b,
                reg=config.regularized,
                xi=xi,
                phi=phi,
                lam=config.lam,
            )
            samples[name].append(values[: config.plot_iterations])
            final_iterates[name].append(x)

    reference_value = min(
        values[-1] for method_runs in samples.values() for values in method_runs
    )
    return {
        "config": asdict(config),
        "A": A,
        "x_star": xt,
        "y": y,
        "samples": samples,
        "final_iterates": final_iterates,
        "reference_value": reference_value,
    }


def save_result(result, output_file):
    """Save a complete experiment result as a compressed NumPy archive."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, result=np.array([result], dtype=object))
