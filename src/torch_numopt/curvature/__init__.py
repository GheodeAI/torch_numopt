"""
Curvature estimators for second-order optimization.

This package provides classes that approximate or compute the Hessian matrix
(and its products) in various ways:

- Exact Hessian (full or block-diagonal)
- Gauss-Newton approximation (full or block)
- Hutchinson diagonal approximation (via random projections)
- Identity (no curvature)

All estimators inherit from :class:`CurvatureEstimator` and implement the
``scaling_matrix``, ``hvp``, and ``quadratic_form`` methods.
"""

from .naive_identity import NaiveIdentityCalculator
from .exact_hessian import ExactHessianCalculator
from .exact_block_hessian import ExactBlockHessianCalculator
from .gauss_newton_approximation import GaussNewtonApproximation
from .gauss_newton_block_approximation import GaussNewtonBlockApproximation
from .hutchinson_diagonal_approximation import HutchinsonDiagonalApproximation

__all__ = [
    "NaiveIdentityCalculator",
    "ExactHessianCalculator",
    "ExactBlockHessianCalculator",
    "GaussNewtonApproximation",
    "GaussNewtonBlockApproximation",
    "HutchinsonDiagonalApproximation",
]
