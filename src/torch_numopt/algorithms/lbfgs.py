
from __future__ import annotations
from typing import Callable
import torch
import torch.nn as nn
from collections import deque
from ..line_search import create_line_search_solver
from ..trust_region import create_trust_region_solver
from ..numerical_optimizer import NumericalOptimizer, LineSearchOptimizer, TrustRegionOptimizer
from ..curvature import NaiveIdentityCalculator
from ..utils import (
    param_add,
    param_neg,
    param_sub,
    param_dot,
    param_scalar_prod,
    param_copy
)

class LBFGSMixin:
    def __init__(self, *args, memory_size: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.s = []
        self.y = []
        self.gamma = 1
        self.memory_size = memory_size

    def get_step_direction(self, params: tuple, d_p_list: tuple):
        if self.prev_params_ is None:
            return d_p_list
        
        # q = param_neg(d_p_list)
        q = d_p_list
        alphas = []
        for s, y in zip(reversed(self.s), reversed(self.y)):
            alpha = param_dot(s, q) / param_dot(s, y)
            q = param_sub(q, param_scalar_prod(alpha, y))
            alphas.append(alpha)
        
        r = param_scalar_prod(self.gamma, q)
        for s, y, alpha in zip(self.s, self.y, reversed(alphas)):
            beta = param_dot(y, r) / param_dot(s, y)
            r = param_add(r, param_scalar_prod(alpha - beta, s))

        if len(self.s) > 0:
            self.gamma = param_dot(self.s[-1], self.y[-1])/param_dot(self.y[-1], self.y[-1])

        return r
    
    def apply_gradients(self, eval_model: Callable, params: list, d_p_list: list):
        old_params = param_copy(params)

        super().apply_gradients(eval_model, params, d_p_list)

        with torch.enable_grad():
            new_loss = eval_model(*params)
            new_grad = torch.autograd.grad(new_loss, params, create_graph=False, retain_graph=False)
        
        new_s = param_sub(params, old_params)
        new_y = param_sub(new_grad, d_p_list)

        if param_dot(new_s, new_y) > 1e-12:
            self.s.append(new_s)
            self.y.append(new_y)
            
        
        if len(self.s) > self.memory_size:
            self.s.pop(0)
            self.y.pop(0)

        return params


class LBFGS(LBFGSMixin, NumericalOptimizer):
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
        memory_size: int = 10,
    ):
        super().__init__(
            model,
            curvature_estimator=NaiveIdentityCalculator(model=model),
            lr_init=lr_init,
            lr_method=lr_method,
            memory_size=memory_size
        )
    
class LBFGSLS(LBFGSMixin, LineSearchOptimizer):
    """
    Heavily inspired by https://github.com/hahnec/torchimize/blob/master/torchimize/optimizer/gna_opt.py
    TODO: configure line search factories

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
        lr_method: str = None,
        c1: float = 1e-4,
        c2: float = 0.9,
        tau: float = 0.1,
        max_iter: int = 20,
        tol: float = 1e-8,
        memory_size: int = 10,
        line_search_method: str = "backtrack",
        line_search_cond: str = "armijo",
    ):

        super().__init__(
            model,
            curvature_estimator=NaiveIdentityCalculator(model=model),
            lr_init=lr_init,
            lr_method=lr_method,
            line_search=create_line_search_solver(
                method=line_search_method, condition=line_search_cond, c1=c1, c2=c2, tau=tau, max_iter=max_iter, tol=tol
            ),
            memory_size = memory_size
        )
    