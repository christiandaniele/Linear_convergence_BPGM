"""Synthetic data generation used in Section 3.2 and Appendix B."""

import numpy as np
from scipy.stats import poisson


def generate_matrix(M, N=500, cond_target=50, dtype=np.float64):
    """Generate the nonnegative matrix described in Appendix B.1.
    """
    A = np.random.rand(M, N).astype(dtype)
    norm = np.linalg.norm(A)
    A = A / norm

    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    s_max = S[0]
    s_min = S[-1]
    S_new = s_max - (s_max - s_max / cond_target) * (
        (s_max - S) / (s_max - s_min)
    )
    A_cond = np.dot(U * S_new, Vt)
    norm = np.linalg.norm(A_cond)
    A_final = A_cond / norm

    # Step 3 of Appendix B.1: enforce nonnegative entries.
    return abs(A_final)


def generate_standard_data(A, b, Q=100, P=None, dtype=np.float64):
    """Generate the noisy data used by the original notebook.

     ``P`` controls
    the number of nonzero ground-truth entries and defaults to ``N``.
    """
    _, N = A.shape
    P = N if P is None else P
    xt = np.zeros((N, 1), dtype=dtype)
    indices = np.random.permutation(N)[:P]
    xt[indices] = Q * np.random.rand(P, 1).astype(dtype)
    y_noiseless = np.dot(A, xt) + b
    y = poisson.rvs(y_noiseless).astype(dtype) / Q
    return xt, y


def generate_interior_data(A, b, lam, Q=1, noise=True, dtype=np.float64):
    """Generate data whose target minimizer is strictly positive.
    """
    M, N = A.shape
    ones_N = np.ones((N, 1), dtype=dtype)

    # Choose a strictly positive target point, away from the boundary.
    x_star = (0.2 + 1.0 * np.random.rand(N, 1)).astype(dtype) * Q

    # Enforce the interior KKT condition.
    rhs_kkt = np.dot(A.T, ones_N) + lam * ones_N
    w = np.linalg.solve(A.T, rhs_kkt)

    # Keep the Poisson rate positive.
    w = np.maximum(w, 1e-5)

    # Construct a signal producing the required data profile.
    target_mu = (np.dot(A, x_star) + b) * w
    x_tilde = np.linalg.solve(A, target_mu - b)
    y_noiseless = np.dot(A, x_tilde) + b
    y_noiseless = np.maximum(y_noiseless, 1e-15)

    if noise:
        y = np.random.poisson(y_noiseless * Q).astype(dtype) / Q
    else:
        y = y_noiseless.copy()

    xt = x_star
    return xt, y

