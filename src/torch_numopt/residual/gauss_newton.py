from __future__ import annotations
import torch.nn as nn
from ..line_search import create_line_search_solver
from ..numerical_optimizer import NumericalOptimizer, LineSearchOptimizer
from ..scaling_matrix_calculator import GaussNewtonBlockApproximation


class GaussNewton(NumericalOptimizer):
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
        model: nn.Module,
        lr_init: float = 1,
        lr_method: str | None = None,
        solver: str = "solve",
        batch_size: int | None = None,
    ):
        super().__init__(
            model,
            scaling_matrix=GaussNewtonBlockApproximation(model=model, batch_size=batch_size, damping=None),
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
        model: nn.Module,
        lr_init: float = 1,
        lr_method: str | None = None,
        c1: float = 1e-4,
        c2: float = 0.9,
        tau: float = 0.1,
        line_search_method: str = "backtrack",
        line_search_cond: str = "armijo",
        solver: str = "solve",
        batch_size: int | None = None,
    ):
        super().__init__(
            model,
            scaling_matrix=GaussNewtonBlockApproximation(model=model, batch_size=batch_size, damping=None),
            lr_init=lr_init,
            lr_method=lr_method,
            line_search=create_line_search_solver(method=line_search_method, condition=line_search_cond, c1=c1, c2=c2, tau=tau),
            solver=solver,
        )
