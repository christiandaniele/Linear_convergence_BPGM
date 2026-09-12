"""Fast structural tests; these do not reproduce the long paper runs."""

import numpy as np

from src.algorithms import BPGM, F, gF
from src.data import generate_matrix
from src.experiments import ExperimentConfig


def test_matrix_shapes_and_rank_regimes():
    np.random.seed(13)
    square = generate_matrix(8, 8)
    rectangular = generate_matrix(6, 8)
    assert square.shape == (8, 8)
    assert rectangular.shape == (6, 8)
    assert np.linalg.matrix_rank(square) == 8
    assert np.linalg.matrix_rank(rectangular) == 6
    assert np.all(square >= 0)
    assert np.all(rectangular >= 0)


def test_configuration_is_derived_from_M_and_lambda():
    config = ExperimentConfig(M=400, lam=0.1)
    assert config.rank_deficient
    assert config.regularized


def test_all_methods_run():
    np.random.seed(13)
    A = generate_matrix(5, 5)
    x_star = np.ones((5, 1))
    b = 1e-3
    y = A @ x_star + b
    x0 = np.ones_like(x_star) * 2

    assert np.isfinite(F(x0, A, b, y))
    assert np.isfinite(gF(x0, A, b, y)).all()
    for phi, xi in (("burg", 0), ("burg", 0.8), ("l2", 0), ("RL", 0)):
        x, values, _, _ = BPGM(
            x0, y, 10, A, b, xi=xi, phi=phi, lam=0, reg=False
        )
        assert x.shape == x0.shape
        assert np.isfinite(values).all()

