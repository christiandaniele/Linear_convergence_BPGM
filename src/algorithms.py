"""Algorithms used in the numerical section of the paper.

The implementations in this module preserve the numerical behavior of the
original experimental notebook. Only comments, docstrings, and organization
have been changed.
"""

import numpy as np


def F(x, A, b, y, reg=False, lam=0):
    """Evaluate the KL objective, optionally including the l1 penalty."""
    arg = np.dot(A, x) + b
    eps = np.finfo(x.dtype).eps if x.dtype == np.float32 else 1e-15
    val = np.sum(arg - y + y * np.log(y / np.maximum(arg, eps)))
    if reg:
        val += lam * np.sum(np.abs(x))
    return val


def gF(x, A, b, y):
    """Evaluate the gradient of the smooth KL data-fidelity term."""
    arg = np.dot(A, x) + b
    eps = np.finfo(x.dtype).eps if x.dtype == np.float32 else 1e-15
    grad = np.dot(A.T, (1 - y / np.maximum(arg, eps)))
    return grad


def burg_divergence(x, y, xi=0):
    """Evaluate the Burg or smoothed-Burg Bregman divergence."""
    return np.sum(np.log1p((y - x) / (x + xi))) + np.sum(
        (x - y) / (y + xi)
    )


def BPGM(
    x_0,
    y,
    n_iter,
    A,
    b_mat,
    reg=False,
    xi=0,
    phi="burg",
    tau=1,
    Beta=0.9,
    lam=0,
    gt=None,
    bt=True,
):
    """Run BPGM, Euclidean PGM, or Richardson--Lucy.

    Parameters follow the original notebook. In particular, ``phi='burg'``
    gives Burg entropy for ``xi=0`` and smoothed Burg entropy for ``xi>0``;
    ``phi='l2'`` gives Euclidean proximal gradient; and ``phi='RL'`` gives
    Richardson--Lucy. The update and backtracking rules are unchanged.
    """
    current_dtype = x_0.dtype
    x = x_0.copy()

    F_vals = np.zeros(n_iter, dtype=current_dtype)
    diff_iter = np.zeros(n_iter, dtype=current_dtype)
    stepsize_evo = np.zeros(n_iter, dtype=current_dtype)

    F_vals[0] = F(x, A, b_mat, y, reg, lam)

    if phi == "burg":
        ones_N = np.ones_like(x_0)
        max_term = np.maximum((np.dot(A, ones_N) * xi) / b_mat, 1.0)
        tauth = 1.0 / np.sum((max_term**2) * y)

        stepsize_evo[0] = tau
        diff_iter[0] = burg_divergence(gt, x, xi) if gt is not None else 0

        for k in range(1, n_iter):
            grad = gF(x, A, b_mat, y)
            if reg:
                grad += lam

            x_xi = x + xi
            eps = np.finfo(current_dtype).eps

            denom = 1.0 + tau * grad * x_xi
            c = x_xi / np.maximum(denom, eps) - xi
            x_new = np.maximum(c, 0) if xi > 0 else c

            # Apply backtracking only when requested.
            if bt:
                if xi == 0:
                    while np.any(x_new < 0):
                        tau *= Beta
                        denom = 1.0 + tau * grad * x_xi
                        c = x_xi / np.maximum(denom, eps) - xi
                        x_new = c

                Axnew = np.dot(A, x_new)
                Ax = np.dot(A, x)

                while True:
                    lhs = np.sum(
                        y
                        * (
                            (Axnew - Ax) / (Ax + b_mat)
                            - np.log1p((Axnew - Ax) / (Ax + b_mat))
                        )
                    )
                    rhs = burg_divergence(x_new, x, xi) / tau

                    if lhs <= rhs:
                        break
                    else:
                        tau *= Beta

                        denom = 1.0 + tau * grad * x_xi
                        c = x_xi / np.maximum(denom, eps) - xi
                        x_new = np.maximum(c, 0) if xi > 0 else c
                        Axnew = np.dot(A, x_new)

                        if tau < tauth:
                            break

            x = x_new.copy()
            F_vals[k] = F(x, A, b_mat, y, reg, lam)
            stepsize_evo[k] = tau
            diff_iter[k] = (
                burg_divergence(gt, x, xi) if gt is not None else 0
            )

    elif phi == "l2":
        tauth = 1e-10

        stepsize_evo[0] = tau
        diff_iter[0] = (
            0.5 * np.sum((x - gt) ** 2) if gt is not None else 0
        )

        for k in range(1, n_iter):
            grad = gF(x, A, b_mat, y)

            x_new = x - tau * grad
            if lam > 0:
                x_new = np.sign(x_new) * np.maximum(
                    np.abs(x_new) - tau * lam, 0
                )
            x_new = np.maximum(x_new, 0)

            if bt:
                # Backtracking line search.
                Axnew = np.dot(A, x_new)
                Ax = np.dot(A, x)

                while True:
                    ratio = ((Axnew - Ax) + 1e-15) / (Ax + b_mat)

                    if (ratio <= -1).any() or np.isnan(ratio).any():
                        valid_condition = False
                    else:
                        lhs = np.sum(
                            y
                            * (
                                (Axnew - Ax) / (Ax + b_mat)
                                - np.log1p(ratio)
                            )
                        )
                        rhs = (0.5 * np.sum((x - x_new) ** 2)) / tau
                        valid_condition = lhs <= rhs

                    if valid_condition:
                        break
                    else:
                        tau *= Beta

                        x_new = x - tau * grad
                        if lam > 0:
                            x_new = np.sign(x_new) * np.maximum(
                                np.abs(x_new) - tau * lam, 0
                            )
                        x_new = np.maximum(x_new, 0)
                        Axnew = np.dot(A, x_new)

                        if tau < tauth:
                            break

            x = x_new
            F_vals[k] = F(x, A, b_mat, y, reg, lam)
            stepsize_evo[k] = tau
            diff_iter[k] = (
                0.5 * np.sum((x - gt) ** 2) if gt is not None else 0
            )

    elif phi == "RL":
        normalization_factor = np.dot(A.T, np.ones_like(y, dtype=x.dtype))
        for k in range(1, n_iter):
            Ax = np.dot(A, x)
            eps = np.finfo(x.dtype).eps if x.dtype == np.float32 else 1e-15
            x = (x / (normalization_factor + lam)) * np.dot(
                A.T, (y / np.maximum(Ax + b_mat, eps))
            )
            F_vals[k] = F(x, A, b_mat, y, reg, lam)
            diff_iter[k] = (
                burg_divergence(gt, x, xi) if gt is not None else 0
            )

    return x, F_vals, diff_iter, stepsize_evo

