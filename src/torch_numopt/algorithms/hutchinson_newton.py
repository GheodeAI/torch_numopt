"""
Newton methods with a diagonal Hutchinson diagonal approximation.
"""

from __future__ import annotations
from ..utils import Params
from ..line_search import create_line_search_solver
from ..numerical_optimizer import NumericalOptimizer, LineSearchOptimizer
from ..curvature import HutchinsonDiagonalApproximation


class DiagonalNewton(NumericalOptimizer):
    """
    Diagonal Newton approximation (using Hutchinson's method).

    Parameters
    ----------
    params : Params
        Parameter tensors.
    lr_init : float, default=1
        Initial learning rate.
    lr_method : str or None, default=None
        Learning-rate initialization method.
    n_samples, skip_iters: Hutchinson approximation parameters.
    """

    def __init__(
        self,
        params: Params,
        lr_init: float = 1e-3,
        lr_method: str | None = None,
        n_samples: int = 10,
        skip_iters: int = 0,
    ):
        super().__init__(
            params,
            curvature_estimator=HutchinsonDiagonalApproximation(n_samples=n_samples, skip_iters=skip_iters),
            lr_init=lr_init,
            lr_method=lr_method,
        )


class DiagonalNewtonLS(LineSearchOptimizer):
    """
    Diagonal Newton approximation (using Hutchinson's method) with line search.

    Parameters
    ----------
    params : Params
        Parameter tensors.
    lr_init : float, default=1
        Initial learning rate.
    lr_method : str or None, default=None
        Learning-rate initialization method.
    n_samples, skip_iters: Hutchinson approximation parameters.
    c1, c2, tau, max_iter, tol : line-search parameters.
    line_search_method : str, default="backtrack"
        Line-search method.
    line_search_cond : str, default="armijo"
        Line-search condition.
    """

    def __init__(
        self,
        params: Params,
        lr_init: float = 1,
        lr_method: str | None = None,
        n_samples: int = 10,
        skip_iters: int = 0,
        c1: float = 1e-4,
        c2: float = 0.9,
        tau: float = 0.1,
        max_iter: int = 20,
        tol: float = 1e-8,
        line_search_method: str = "backtrack",
        line_search_cond: str = "armijo",
    ):
        super().__init__(
            params,
            curvature_estimator=HutchinsonDiagonalApproximation(n_samples=n_samples, skip_iters=skip_iters),
            lr_init=lr_init,
            lr_method=lr_method,
            line_search=create_line_search_solver(
                method=line_search_method, condition=line_search_cond, c1=c1, c2=c2, tau=tau, max_iter=max_iter, tol=tol
            ),
        )
