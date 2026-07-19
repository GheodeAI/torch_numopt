"""
Gradient descent optimizers (first-order).

These optimizers use only the gradient (identity curvature). They are the
simplest methods and serve as a baseline.
"""

from __future__ import annotations
from typing import Iterable
import torch
from ..line_search import create_line_search_solver
from ..trust_region import create_trust_region_solver
from ..numerical_optimizer import NumericalOptimizer, LineSearchOptimizer, TrustRegionOptimizer
from ..curvature import NaiveIdentityCalculator
from ..utils import Params


class GradientDescent(NumericalOptimizer):
    """
    Vanilla gradient descent with a fixed or adaptively initialized learning rate.

    Parameters
    ----------
    params : Params
        Parameter tensors.
    lr_init : float, default=1e-3
        Initial learning rate.
    lr_method : str, optional
        Learning-rate initialization method (see :class:`NumericalOptimizer`).
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr_init: float = 1e-3,
        lr_method: str = None,
    ):

        super().__init__(
            params=params,
            curvature_estimator=NaiveIdentityCalculator(),
            lr_init=lr_init,
            lr_method=lr_method,
        )


class GradientDescentLS(LineSearchOptimizer):
    """
    Gradient descent with a line search to determine the step length.

    This optimizer computes the steepest descent direction and then performs a
    line search (backtracking, interpolation, or bisection) to find a step size
    that satisfies the chosen condition (Armijo, Wolfe, Goldstein, etc.).

    Parameters
    ----------
    params : Params
        Parameter tensors to optimize.
    lr_init : float, default=1
        Initial guess for the learning rate (used as the starting point for the
        line search).
    lr_method : str or None, default=None
        Method for initializing the learning rate before the line search
        (see :class:`NumericalOptimizer` for options). If None, uses `lr_init`
        directly.
    c1 : float, default=1e-4
        Sufficient decrease parameter (Armijo condition).
    c2 : float, default=0.9
        Curvature condition parameter (Wolfe conditions).
    tau : float, default=0.1
        Step-size reduction factor for backtracking.
    max_iter : int, default=20
        Maximum number of line-search iterations.
    tol : float, default=1e-8
        Tolerance for stopping (e.g., minimum step size).
    line_search_method : str, default="backtrack"
        Line-search algorithm. Options: "backtrack", "interpolate", "bisect".
    line_search_cond : str, default="armijo"
        Stopping condition. Options: "greedy", "armijo", "wolfe",
        "strong-wolfe", "goldstein".
    """

    def __init__(
        self,
        params: Params,
        lr_init: float = 1,
        lr_method: str = None,
        c1: float = 1e-4,
        c2: float = 0.9,
        tau: float = 0.1,
        max_iter: int = 20,
        tol: float = 1e-8,
        line_search_method: str = "backtrack",
        line_search_cond: str = "armijo",
    ):

        super().__init__(
            params=params,
            curvature_estimator=NaiveIdentityCalculator(),
            lr_init=lr_init,
            lr_method=lr_method,
            line_search=create_line_search_solver(
                method=line_search_method, condition=line_search_cond, c1=c1, c2=c2, tau=tau, max_iter=max_iter, tol=tol
            ),
        )


class GradientDescentTR(TrustRegionOptimizer):
    """
    Gradient descent with a trust-region (Cauchy point) step.

    This optimizer builds a quadratic model using the gradient (identity
    curvature) and solves the trust-region subproblem using the Cauchy point,
    which is the minimizer of the model along the steepest descent direction
    within the trust region.

    Parameters
    ----------
    params : Params
        Parameter tensors to optimize.
    radius_init : float, default=1.0
        Initial trust-region radius.
    trust_region_method : str, default="cauchy"
        Trust-region solver method. For gradient descent, only "cauchy" is
        recommended, but other methods (e.g., "dogleg", "exact") can be used
        if a curvature estimator is provided (here it uses identity, so they
        degenerate to Cauchy point anyway).
    accept_tol : float, default=0.1
        Threshold for the ratio rho (actual vs. predicted reduction) above
        which the step is accepted.
    curvature_estimator : CurvatureEstimator, optional
        Not used directly (identity is forced), kept for compatibility with
        the base class.
    """

    def __init__(
        self,
        params: Params,
        lr_init: float = 1.0,
        trust_region_method: str = "cauchy",
        *,
        accept_tol: float = 0.1,
        contract_tol: float = 0.25,
        expand_tol: float = 0.75,
        growth_factor: float = 2,
        shrink_factor: float = 0.25,
        radius_max: float = 1e3,
    ):
        super().__init__(
            params,
            trust_region=create_trust_region_solver(method=trust_region_method, curvature_estimator=NaiveIdentityCalculator()),
            lr_init=lr_init,
            accept_tol=accept_tol,
            contract_tol=contract_tol,
            expand_tol=expand_tol,
            growth_factor=growth_factor,
            shrink_factor=shrink_factor,
            radius_max=radius_max,
        )


class GradientDescentLipschitz(NumericalOptimizer):
    """
    Gradient descent with a learning rate estimated from the Lipschitz constant.

    This optimizer uses a heuristic (the "lipschitz" method) to estimate the
    learning rate at each step based on the change in gradient and parameters
    from the previous iteration. It avoids the need for manual tuning and often
    converges faster than fixed-rate gradient descent.

    Parameters
    ----------
    params : Params
        Parameter tensors to optimize.
    lr_init : float, default=1e-3
        Initial learning rate (used only for the first iteration).
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr_init: float = 1e-3,
    ):
        super().__init__(
            params=params,
            curvature_estimator=NaiveIdentityCalculator(),
            lr_init=lr_init,
            lr_method="lipschitz",
        )
