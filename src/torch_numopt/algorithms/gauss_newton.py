"""
Gauss-Newton optimization algorithms for least-squares problems.

The Gauss-Newton method approximates the Hessian as JᵀJ, where J is the Jacobian
of the residual vector. This module provides three variants: a vanilla version
with a fixed learning rate, a line-search version, and a trust-region version.
The block-diagonal approximation is available for memory efficiency.
"""

from __future__ import annotations
from ..line_search import create_line_search_solver
from ..trust_region import create_trust_region_solver
from ..numerical_optimizer import NumericalOptimizer, LineSearchOptimizer, TrustRegionOptimizer
from ..curvature import GaussNewtonBlockApproximation, GaussNewtonApproximation
from ..utils import Params


class GaussNewton(NumericalOptimizer):
    """
    Gauss-Newton optimizer (no line search or trust region).

    Uses the Gauss-Newton approximation of the Hessian and solves the linear
    system to obtain the step direction, then applies a fixed learning rate.

    Parameters
    ----------
    params : Params
        Parameter tensors.
    lr_init : float, default=1e-3
        Initial learning rate.
    lr_method : str or None, default=None
        Learning rate initialization method.
    solver : str, default="solve"
        Linear solver for the system.
    damping : str or None, default=None
        Damping strategy.
    mu : float, default=1
        Damping coefficient.
    block_hessian : bool, default=True
        If True, use block-diagonal Gauss-Newton; else full.
    """

    def __init__(
        self,
        params: Params, 
        lr_init: float = 1e-3,
        lr_method: str | None = None,
        solver: str = "solve",
        damping: str | None = None,
        mu: float = 1,
        block_hessian: bool = True,
    ):
        if block_hessian:
            curvature_estimator = GaussNewtonBlockApproximation(damping=damping, mu=mu)
        else:
            curvature_estimator = GaussNewtonApproximation(damping=damping, mu=mu)

        super().__init__(
            params,
            curvature_estimator=curvature_estimator,
            lr_init=lr_init,
            lr_method=lr_method,
            solver=solver,
        )


class GaussNewtonLS(LineSearchOptimizer):
    """
    Gauss-Newton optimizer with line search for step-length selection.

    This method computes the step direction by solving the Gauss-Newton system
    (JᵀJ) p = -g, where J is the Jacobian of the residuals and g is the gradient.
    The resulting direction is then scaled by a step length determined by a
    line search (backtracking, interpolation, or bisection) that satisfies a
    chosen condition (Armijo, Wolfe, etc.).

    This is particularly useful for non-linear least-squares problems where the
    residual vector is known and the Hessian can be approximated as JᵀJ.

    Parameters
    ----------
    params : Params
        Parameter tensors to optimize.
    lr_init : float, default=1
        Initial guess for the learning rate (starting point for the line search).
    lr_method : str or None, default=None
        Learning-rate initialization method (see :class:`NumericalOptimizer`).
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
    solver : str, default="solve"
        Linear solver used to invert the Gauss-Newton system.
    damping : str or None, default=None
        Damping strategy ("identity" or "fletcher") to improve conditioning.
    mu : float, default=1
        Damping coefficient.
    block_hessian : bool, default=True
        If True, use block-diagonal Gauss-Newton (each parameter group forms
        its own block) to save memory; otherwise, compute the full matrix.
    """

    def __init__(
        self,
        params: Params,
        lr_init: float = 1,
        lr_method: str | None = None,
        c1: float = 1e-4,
        c2: float = 0.9,
        tau: float = 0.1,
        max_iter: int = 20,
        tol: float = 1e-8,
        line_search_method: str = "backtrack",
        line_search_cond: str = "armijo",
        solver: str = "solve",
        damping: str | None = None,
        mu: float = 1,
        block_hessian: bool = True,
    ):
        if block_hessian:
            curvature_estimator = GaussNewtonBlockApproximation(damping=damping, mu=mu)
        else:
            curvature_estimator = GaussNewtonApproximation(damping=damping, mu=mu)

        super().__init__(
            params,
            curvature_estimator=curvature_estimator,
            lr_init=lr_init,
            lr_method=lr_method,
            line_search=create_line_search_solver(
                method=line_search_method, condition=line_search_cond, c1=c1, c2=c2, tau=tau, max_iter=max_iter, tol=tol
            ),
            solver=solver,
        )


class GaussNewtonTR(TrustRegionOptimizer):
    """
    Gauss-Newton optimizer with a trust-region subproblem solver.

    This method builds a quadratic model using the Gauss-Newton approximation
    of the Hessian (JᵀJ) and solves the trust-region subproblem
        minimize  m(p) = f + gᵀp + ½ pᵀ(JᵀJ)p   subject to  ||p|| ≤ Δ.
    The step is computed by a trust-region solver (e.g., Cauchy point, dogleg,
    exact, or Steihaug-Toint) that respects the trust-region radius.

    Trust-region Gauss-Newton is robust and often converges faster than the
    line-search variant, especially in regions where the quadratic model is
    not accurate.

    Parameters
    ----------
    params : Params
        Parameter tensors to optimize.
    radius_init : float, default=1.0
        Initial trust-region radius.
    trust_region_method : str, default="exact"
        Trust-region solver method (see `create_trust_region_solver`).
        Common choices: "cauchy", "dogleg", "exact", "steihaug-toint".
    solver : str, default="solve"
        Linear solver used internally (for methods that require solving a system).
    damping : str or None, default=None
        Damping strategy to improve conditioning.
    mu : float, default=1
        Damping coefficient.
    block_hessian : bool, default=False
        If True, use block-diagonal Gauss-Newton; otherwise, compute the full
        matrix. Note: block-diagonal is often sufficient and saves memory.
    accept_tol : float, default=0.1
        Threshold for the ratio rho (actual vs. predicted reduction) above
        which the step is accepted.
    """

    def __init__(
        self,
        params: Params,
        radius_init: float = 1.0,
        trust_region_method: str = "exact",
        solver: str = "solve",
        damping: str | None = None,
        mu: float = 1,
        block_hessian: bool = False,
    ):
        if block_hessian:
            curvature_estimator = GaussNewtonBlockApproximation(damping=damping, mu=mu)
        else:
            curvature_estimator = GaussNewtonApproximation(damping=damping, mu=mu)

        super().__init__(
            params,
            trust_region=create_trust_region_solver(method=trust_region_method, curvature_estimator=curvature_estimator, solver=solver),
            curvature_estimator=curvature_estimator,
            radius_init=radius_init,
        )
