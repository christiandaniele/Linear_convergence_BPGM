"""Numerical experiments for BPGM applied to KL regression."""

from .algorithms import BPGM, F, burg_divergence, gF
from .data import generate_interior_data, generate_matrix, generate_standard_data

__all__ = [
    "BPGM",
    "F",
    "gF",
    "burg_divergence",
    "generate_interior_data",
    "generate_matrix",
    "generate_standard_data",
]

