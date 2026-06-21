from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn
from ..line_search import create_line_search_solver
from ..trust_region import create_trust_region_solver
from ..numerical_optimizer import NumericalOptimizer, LineSearchOptimizer, TrustRegionOptimizer
from ..curvature import NaiveIdentityCalculator
from ..utils import Params


class GradientDescent(NumericalOptimizer):
    """
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
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr_init: float = 1,
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
    """

    def __init__(
        self,
        params: Params,
        radius_init: float = 1.0,
        trust_region_method: str = "cauchy",
    ):
        curvature_estimator = NaiveIdentityCalculator()
        super().__init__(
            params,
            curvature_estimator=curvature_estimator,
            trust_region=create_trust_region_solver(method=trust_region_method, curvature_estimator=curvature_estimator),
            radius_init=radius_init,
        )
