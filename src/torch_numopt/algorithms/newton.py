"""
Newton-type methods using exact Hessian (full or block).

These optimizers compute the exact second-order derivatives and use them to
form a quadratic model. They offer fast local convergence but may be expensive
for large models.
"""

from __future__ import annotations
import torch.nn as nn
from ..line_search import create_line_search_solver
from ..trust_region import SteihaugTointTRSolver, create_trust_region_solver
from ..numerical_optimizer import NumericalOptimizer, LineSearchOptimizer, TrustRegionOptimizer
from ..curvature import ExactBlockHessianCalculator, ExactHessianCalculator
from ..utils import Params
from ..solve_system import iterative_solver_set


class Newton(NumericalOptimizer):
    """
    Newton method with exact Hessian (full or block) and fixed learning rate.

    Uses the exact Hessian (or block-diagonal) to compute the Newton step.

    Parameters
    ----------
    params : Params
        Parameter tensors.
    lr_init : float, default=1
        Initial learning rate.
    lr_method : str or None, default=None
        Learning rate initialization method.
    damping : str or None, default=None
        Damping strategy.
    mu : float, default=1
        Damping coefficient.
    solver : str, default="solve"
        Linear solver for the system.
    block_hessian : bool, default=True
        If True, use block-diagonal Hessian.
    """

    def __init__(
        self,
        params: Params,
        lr_init: float = 1,
        lr_method: str | None = None,
        damping: str = None,
        mu: float = 1,
        solver: str = "solve",
        block_hessian: bool = True,
    ):
        if block_hessian:
            curvature_estimator = ExactBlockHessianCalculator(damping=damping, mu=mu)
        else:
            curvature_estimator = ExactHessianCalculator(damping=damping, mu=mu)

        super().__init__(
            params,
            curvature_estimator=curvature_estimator,
            lr_init=lr_init,
            lr_method=lr_method,
            solver=solver,
        )


class NewtonLS(LineSearchOptimizer):
    """
    Newton method with exact Hessian and line search.

    Parameters
    ----------
    params : Params
        Parameter tensors.
    lr_init : float, default=1
        Initial learning rate.
    lr_method : str or None, default=None
        Learning rate initialization method.
    c1, c2, tau, max_iter, tol : line search parameters.
    damping : str or None, default=None
        Damping strategy.
    mu : float, default=1
        Damping coefficient.
    line_search_method : str, default="backtrack"
        Line-search method.
    line_search_cond : str, default="armijo"
        Line-search condition.
    solver : str, default="solve"
        Linear solver.
    block_hessian : bool, default=True
        If True, use block-diagonal Hessian.
    """

    def __init__(
        self,
        params: nn.Module,
        lr_init: float = 1,
        lr_method: str | None = None,
        c1: float = 1e-4,
        c2: float = 0.9,
        tau: float = 0.1,
        max_iter: int = 20,
        tol: float = 1e-8,
        damping: str = None,
        mu: float = 1,
        line_search_method: str = "backtrack",
        line_search_cond: str = "armijo",
        solver: str = "solve",
        block_hessian: bool = True,
    ):
        if block_hessian:
            curvature_estimator = ExactBlockHessianCalculator(damping=damping, mu=mu)
        else:
            curvature_estimator = ExactHessianCalculator(damping=damping, mu=mu)

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


class NewtonTR(TrustRegionOptimizer):
    """
    Newton method with exact Hessian and trust region.

    Uses a trust-region solver (e.g., exact or Steihaug-Toint) to compute the step.

    Parameters
    ----------
    params : Params
        Parameter tensors.
    radius_init : float, default=1.0
        Initial trust-region radius.
    trust_region_method : str, default="exact"
        Trust-region solver method.
    damping : str or None, default=None
        Damping strategy.
    mu : float, default=1
        Damping coefficient.
    solver : str, default="solve"
        Linear solver for the system.
    block_hessian : bool, default=False
        If True, use block-diagonal Hessian.
    """

    def __init__(
        self,
        params: Params,
        lr_init: float = 1.0,
        trust_region_method: str = "exact",
        damping: str = None,
        mu: float = 1,
        solver: str = "solve",
        block_hessian: bool = False,
        *,
        accept_tol: float = 0.1,
        contract_tol: float = 0.25,
        expand_tol: float = 0.75,
        growth_factor: float = 2,
        shrink_factor: float = 0.25,
        radius_max: float = 1e3,
    ):
        if block_hessian:
            curvature_estimator = ExactBlockHessianCalculator(damping=damping, mu=mu)
        else:
            curvature_estimator = ExactHessianCalculator(damping=damping, mu=mu)

        super().__init__(
            params,
            trust_region=create_trust_region_solver(method=trust_region_method, curvature_estimator=curvature_estimator, solver=solver),
            curvature_estimator=curvature_estimator,
            lr_init=lr_init,
            accept_tol=accept_tol,
            contract_tol=contract_tol,
            expand_tol=expand_tol,
            growth_factor=growth_factor,
            shrink_factor=shrink_factor,
            radius_max=radius_max,
        )


class NewtonCG(NumericalOptimizer):
    """
    Newton-CG method (inexact Newton) using conjugate gradient to solve the linear system.

    Uses exact Hessian but solves the system iteratively with CG.

    Parameters
    ----------
    params : Params
        Parameter tensors.
    lr_init : float, default=1
        Initial learning rate.
    lr_method : str or None, default=None
        Learning rate initialization method.
    damping : str or None, default=None
        Damping strategy.
    mu : float, default=1
        Damping coefficient.
    solver : str, default="cg-trunc"
        Iterative solver (must be in iterative_solver_set).
    """

    def __init__(self, params: Params, lr_init: float = 1, lr_method: str | None = None, damping: str = None, mu: float = 1, solver="cg-trunc"):
        assert solver in iterative_solver_set, "``NewtonCG`` does not accept direct solvers. Consider using the ``Newton`` optimizer."

        super().__init__(
            params,
            curvature_estimator=ExactHessianCalculator(damping=damping, mu=mu),
            lr_init=lr_init,
            lr_method=lr_method,
            solver=solver,
        )


class NewtonCGLS(LineSearchOptimizer):
    """
    Newton-CG with line search.

    Combines the iterative CG solution of the Newton system with a line search
    to determine the step length.

    Parameters
    ----------
    params : Params
        Parameter tensors.
    lr_init : float, default=1
        Initial learning rate.
    lr_method : str or None, default=None
        Learning-rate initialization method.
    c1, c2, tau, max_iter, tol : line-search parameters.
    damping : str or None, default=None
        Damping strategy.
    mu : float, default=1
        Damping coefficient.
    line_search_method : str, default="backtrack"
        Line-search method.
    line_search_cond : str, default="armijo"
        Stopping condition.
    solver : str, default="cg-trunc"
        Iterative solver.
    """

    def __init__(
        self,
        params: nn.Module,
        lr_init: float = 1,
        lr_method: str | None = None,
        c1: float = 1e-4,
        c2: float = 0.9,
        tau: float = 0.1,
        max_iter: int = 20,
        tol: float = 1e-8,
        damping: str = None,
        mu: float = 1,
        line_search_method: str = "backtrack",
        line_search_cond: str = "armijo",
        solver="cg-trunc",
    ):
        assert solver in iterative_solver_set, "``NewtonCG`` does not accept direct solvers. Consider using the ``Newton`` optimizer."

        super().__init__(
            params,
            curvature_estimator=ExactHessianCalculator(damping=damping, mu=mu),
            lr_init=lr_init,
            lr_method=lr_method,
            line_search=create_line_search_solver(
                method=line_search_method, condition=line_search_cond, c1=c1, c2=c2, tau=tau, max_iter=max_iter, tol=tol
            ),
            solver=solver,
        )


class NewtonCGTR(TrustRegionOptimizer):
    """
    Newton-CG with trust region (Steihaug-Toint).

    Uses the Steihaug-Toint CG-trust-region method, which solves the trust-region
    subproblem iteratively with a CG approach that automatically handles negative
    curvature and the trust-region boundary.

    Parameters
    ----------
    params : Params
        Parameter tensors.
    radius_init : float, default=1.0
        Initial trust-region radius.
    damping : str or None, default=None
        Damping strategy.
    mu : float, default=1
        Damping coefficient.
    """

    def __init__(
        self,
        params: Params,
        lr_init: float = 1.0,
        damping: str = None,
        mu: float = 1,
        *,
        accept_tol: float = 0.1,
        contract_tol: float = 0.25,
        expand_tol: float = 0.75,
        growth_factor: float = 2,
        shrink_factor: float = 0.25,
        radius_max: float = 1e3,
    ):
        curvature_estimator = ExactHessianCalculator(damping=damping, mu=mu)

        super().__init__(
            params,
            trust_region=SteihaugTointTRSolver(curvature_estimator),
            curvature_estimator=curvature_estimator,
            lr_init=lr_init,
            accept_tol=accept_tol,
            contract_tol=contract_tol,
            expand_tol=expand_tol,
            growth_factor=growth_factor,
            shrink_factor=shrink_factor,
            radius_max=radius_max,
        )
