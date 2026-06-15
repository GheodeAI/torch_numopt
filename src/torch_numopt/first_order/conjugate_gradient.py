from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn
from ..line_search import create_line_search_solver
from ..numerical_optimizer import NumericalOptimizer, LineSearchOptimizer
from ..curvature import NaiveIdentityCalculator
from ..utils import param_reshape_like


class ConjugateGradient(NumericalOptimizer):
    """
    Heavily inspired by https://github.com/hahnec/torchimize/blob/master/torchimize/optimizer/gna_opt.py
    https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
    https://arxiv.org/abs/2201.08568

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
    cg_method: str
        Formula used to calculate the conjugate gradient, options are "FR", "PR" and "PRP+".
    """

    def __init__(
        self,
        model: nn.Module,
        lr_init: float = 1,
        lr_method: str | None = None,
        cg_method: str = "PRP+",
    ):
        super().__init__(
            model,
            scaling_matrix=NaiveIdentityCalculator(model=model),
            lr_init=lr_init,
            lr_method=lr_method,
        )

        # Conjugate gradient memory
        self.cg_method = cg_method

    def get_step_direction(self, d_p_list, _):
        """ """

        if self.prev_grad_ is None:
            return d_p_list

        grad = torch.hstack([i.ravel() for i in d_p_list])
        prev_grad = torch.hstack([i.ravel() for i in self.prev_grad_])
        prev_step = torch.hstack([i.ravel() for i in self.prev_step_dir_])

        res = -grad
        prev_res = -prev_grad

        eps = torch.finfo(res.dtype).eps
        match self.cg_method:
            case "FR":
                beta = torch.dot(res, res) / (torch.dot(prev_res, prev_res) + eps)
            case "PR":
                beta = torch.dot(res, res - prev_res) / (torch.dot(prev_res, prev_res) + eps)
            case "PRP+":
                beta = torch.dot(res, res - prev_res) / (torch.dot(prev_res, prev_res) + eps)
                beta = torch.relu(beta)
            case "HS":
                beta = torch.dot(res, res - prev_res) / (torch.dot(prev_step, res - prev_res) + eps)
            case "DY":
                beta = torch.dot(res, res) / (torch.dot(-prev_step, res - prev_res) + eps)
            case _:
                raise ValueError("Incorrect conjugate gradient method, try 'FR', 'PR' or 'PRP+', 'HS', 'DY'.")

        # Invert sign since we update the weights like x - lr*step
        next_dir = param_reshape_like(grad - beta * prev_step, d_p_list)
        return next_dir


class ConjugateGradientLS(LineSearchOptimizer):
    """
    Heavily inspired by https://github.com/hahnec/torchimize/blob/master/torchimize/optimizer/gna_opt.py
    https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
    https://arxiv.org/abs/2201.08568

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
    cg_method: str
        Formula used to calculate the conjugate gradient, options are "FR", "PR" and "PRP+".
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
        line_search_method: str = "backtrack",
        line_search_cond: str = "armijo",
        cg_method: str = "PRP+",
        **kwargs,
    ):
        super().__init__(
            model,
            scaling_matrix=NaiveIdentityCalculator(model=model),
            lr_init=lr_init,
            lr_method=lr_method,
            line_search=create_line_search_solver(
                method=line_search_method, condition=line_search_cond, c1=c1, c2=c2, tau=tau, max_iter=max_iter, tol=tol
            ),
        )

        # Conjugate gradient memory
        self.cg_method = cg_method

    def get_step_direction(self, d_p_list, _):
        """ """

        if self.prev_grad_ is None:
            return d_p_list

        grad = torch.hstack([i.ravel() for i in d_p_list])
        prev_grad = torch.hstack([i.ravel() for i in self.prev_grad_])
        prev_step = torch.hstack([i.ravel() for i in self.prev_step_dir_])

        res = -grad
        prev_res = -prev_grad

        eps = torch.finfo(res.dtype).eps
        match self.cg_method:
            case "FR":
                beta = torch.dot(res, res) / (torch.dot(prev_res, prev_res) + eps)
            case "PR":
                beta = torch.dot(res, res - prev_res) / (torch.dot(prev_res, prev_res) + eps)
            case "PRP+":
                beta = torch.dot(res, res - prev_res) / (torch.dot(prev_res, prev_res) + eps)
                beta = torch.relu(beta)
            case "HS":
                beta = torch.dot(res, res - prev_res) / (torch.dot(prev_step, res - prev_res) + eps)
            case "DY":
                beta = torch.dot(res, res) / (torch.dot(-prev_step, res - prev_res) + eps)
            case _:
                raise ValueError("Incorrect conjugate gradient method, try 'FR', 'PR' or 'PRP+', 'HS', 'DY'.")

        # Invert sign since we update the weights like x - lr*step
        next_dir = param_reshape_like(grad - beta * prev_step, d_p_list)
        return next_dir
