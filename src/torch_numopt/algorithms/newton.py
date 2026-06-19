from __future__ import annotations
import torch.nn as nn
from ..line_search import create_line_search_solver
from ..trust_region import create_trust_region_solver
from ..numerical_optimizer import NumericalOptimizer, LineSearchOptimizer, TrustRegionOptimizer
from ..curvature import ExactBlockHessianCalculator


class Newton(NumericalOptimizer):
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
    damping: bool
        Whether to use the diagonal of the Hessian matrix instead of an identity matrix to adjust the Hessian matrix.
    mu: float
        Initial value for the coefficient used when adding a diagonal matrix to the Hessian matrix.
    mu_dec: float
        Factor with which to decrease the coefficient of the diagonal matrix if the previous iteration didn't improve the model.
    mu_max: float
        Factor with which to increase the coefficient of the diagonal matrix if the previous iteration improved the model.
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
        model: nn.Module,
        lr_init: float = 1,
        lr_method: str | None = None,
        damping: str = None,
        mu: float = 1,
        solver: str = "solve",
        batch_size: int | None = None,
    ):
        super().__init__(
            model,
            curvature_estimator=ExactBlockHessianCalculator(model=model, batch_size=batch_size, damping=damping, mu=mu),
            lr_init=lr_init,
            lr_method=lr_method,
            solver=solver,
        )


class NewtonLS(LineSearchOptimizer):
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
    damping: bool
        Whether to use the diagonal of the Hessian matrix instead of an identity matrix to adjust the Hessian matrix.
    mu: float
        Initial value for the coefficient used when adding a diagonal matrix to the Hessian matrix.
    mu_dec: float
        Factor with which to decrease the coefficient of the diagonal matrix if the previous iteration didn't improve the model.
    mu_max: float
        Factor with which to increase the coefficient of the diagonal matrix if the previous iteration improved the model.
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
        model: nn.Module,
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
        batch_size: int | None = None,
    ):
        super().__init__(
            model,
            curvature_estimator=ExactBlockHessianCalculator(model=model, batch_size=batch_size, damping=damping, mu=mu),
            lr_init=lr_init,
            lr_method=lr_method,
            line_search=create_line_search_solver(
                method=line_search_method, condition=line_search_cond, c1=c1, c2=c2, tau=tau, max_iter=max_iter, tol=tol
            ),
            solver=solver,
        )


class NewtonTR(TrustRegionOptimizer):
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
    damping: bool
        Whether to use the diagonal of the Hessian matrix instead of an identity matrix to adjust the Hessian matrix.
    mu: float
        Initial value for the coefficient used when adding a diagonal matrix to the Hessian matrix.
    mu_dec: float
        Factor with which to decrease the coefficient of the diagonal matrix if the previous iteration didn't improve the model.
    mu_max: float
        Factor with which to increase the coefficient of the diagonal matrix if the previous iteration improved the model.
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
        model: nn.Module,
        radius_init: float = 1.0,
        tr_method: str = "cauchy",
        damping: str = None,
        mu: float = 1,
        solver: str = "solve",
        batch_size: int | None = None,
    ):
        curvature_estimator = ExactBlockHessianCalculator(model=model, batch_size=batch_size, damping=damping, mu=mu)
        super().__init__(
            model,
            curvature_estimator=curvature_estimator,
            trust_region=create_trust_region_solver(method=tr_method, curvature_estimator=curvature_estimator, solver=solver),
            radius_init=radius_init,
            solver=solver,
        )
