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
    Heavily inspired by https://github.com/hahnec/torchimize/blob/master/torchimize/optimizer/gna_opt.py

    Parameters
    ----------

    model: nn.Module
        The model to be optimized
    lr_init: float
        Maximum learning rate in backtracking line search, if the learning rate is set as constant, this will be the value used.
    lr_method: str
        Method to use to initialize the learning rate before applying line search.
    c1: float
        Coefficient of the sufficient increase condition in backtracking line search.
    c2: float
        Coefficient used in the second condition for wolfe conditions.
    tau: float
        Factor used to reduce the step size in each step of the backtracking line search.
    line_search_method: str
        Method used for line search, options are "backtrack" and "constant".
    line_search_cond: str
        Condition to be used in backtracking line search, options are "armijo", "wolfe", "strong-wolfe" and "goldstein".
    solver: str
        Method to use to invert the hessian.
    batch_size: int
        Size of the amount of data to use at a time to calculate the hessian matrix.
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
    Heavily inspired by https://github.com/hahnec/torchimize/blob/master/torchimize/optimizer/gna_opt.py

    Parameters
    ----------

    model: nn.Module
        The model to be optimized
    lr_init: float
        Maximum learning rate in backtracking line search, if the learning rate is set as constant, this will be the value used.
    lr_method: str
        Method to use to initialize the learning rate before applying line search.
    c1: float
        Coefficient of the sufficient increase condition in backtracking line search.
    c2: float
        Coefficient used in the second condition for wolfe conditions.
    tau: float
        Factor used to reduce the step size in each step of the backtracking line search.
    line_search_method: str
        Method used for line search, options are "backtrack" and "constant".
    line_search_cond: str
        Condition to be used in backtracking line search, options are "armijo", "wolfe", "strong-wolfe" and "goldstein".
    solver: str
        Method to use to invert the hessian.
    batch_size: int
        Size of the amount of data to use at a time to calculate the hessian matrix.
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
