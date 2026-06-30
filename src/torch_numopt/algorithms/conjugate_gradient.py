from __future__ import annotations
import torch
import torch.nn as nn

from ..line_search import create_line_search_solver
from ..numerical_optimizer import NumericalOptimizer, LineSearchOptimizer
from ..curvature import NaiveIdentityCalculator
from ..utils import param_dot, param_neg, param_scaled_add, param_diff, param_numel, Params


class ConjugateGradientMixin:
    def __init__(self, *args, cg_method: str = "PRP+", **kwargs):
        super().__init__(*args, **kwargs)
        self.cg_method = cg_method
        self.iter_count = 0
        self.reset = False

    def get_step_direction(self, objective, grad_params):
        """ """

        self.reset = False
        step_dir = param_neg(grad_params)

        if self.prev_grad is None:
            return step_dir

        prev_grad = self.prev_grad
        prev_step = self.prev_step_dir

        eps = torch.finfo(grad_params[0].dtype).eps
        match self.cg_method:
            case "FR":
                beta = param_dot(grad_params, grad_params) / (param_dot(prev_grad, prev_grad) + eps)
            case "PR":
                beta = param_dot(grad_params, param_diff(grad_params, prev_grad)) / (param_dot(prev_grad, prev_grad) + eps)
                self.reset = param_dot(prev_grad, grad_params) >= 0.2 * param_dot(grad_params, grad_params)
            case "PRP+":
                beta = param_dot(grad_params, param_diff(grad_params, prev_grad)) / (param_dot(prev_grad, prev_grad) + eps)
                beta = torch.relu(beta)
                self.reset = param_dot(prev_grad, grad_params) >= 0.2 * param_dot(grad_params, grad_params)
            case "HS":
                beta = param_dot(grad_params, param_diff(grad_params, prev_grad)) / (-param_dot(prev_step, param_diff(grad_params, prev_grad)) + eps)
            case "DY":
                beta = param_dot(grad_params, grad_params) / (-param_dot(prev_step, param_diff(grad_params, prev_grad)) + eps)
            case _:
                raise ValueError("Incorrect conjugate gradient method, try 'FR', 'PR' or 'PRP+', 'HS', 'DY'.")

        cg_step = param_scaled_add(grad_params, prev_step, scale=beta)

        self.iter_count += 1
        if self.iter_count >= param_numel(grad_params) or param_dot(grad_params, cg_step) > 0:
            self.reset = True
            self.iter_count = 0

        if param_dot(grad_params, cg_step) > 0:
            return param_neg(grad_params)

        return param_neg(cg_step)


class ConjugateGradient(ConjugateGradientMixin, NumericalOptimizer):
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
        params: Params,
        lr_init: float = 1.0,
        lr_method: str | None = "lipschitz",
        cg_method: str = "PRP+",
    ):
        super().__init__(params, curvature_estimator=NaiveIdentityCalculator(), lr_init=lr_init, lr_method=lr_method, cg_method=cg_method)


class ConjugateGradientLS(ConjugateGradientMixin, LineSearchOptimizer):
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
        params: nn.Module,
        lr_init: float = 1.0,
        lr_method: str = None,
        c1: float = 1e-4,
        c2: float = 0.9,
        tau: float = 0.1,
        max_iter: int = 20,
        tol: float = 1e-8,
        line_search_method: str = "backtrack",
        line_search_cond: str = "armijo",
        cg_method: str = "PRP+",
    ):
        super().__init__(
            params,
            curvature_estimator=NaiveIdentityCalculator(),
            lr_init=lr_init,
            lr_method=lr_method,
            line_search=create_line_search_solver(
                method=line_search_method, condition=line_search_cond, c1=c1, c2=c2, tau=tau, max_iter=max_iter, tol=tol
            ),
            cg_method=cg_method,
        )
