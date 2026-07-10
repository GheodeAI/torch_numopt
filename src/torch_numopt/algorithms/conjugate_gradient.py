"""
Non-linear conjugate gradient methods.

These methods combine the gradient with previous search directions to achieve
faster convergence than gradient descent, without requiring explicit curvature.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from ..line_search import create_line_search_solver
from ..numerical_optimizer import NumericalOptimizer, LineSearchOptimizer
from ..curvature import NaiveIdentityCalculator
from ..utils import param_dot, param_neg, param_scaled_add, param_diff, param_numel, Params


class ConjugateGradientMixin:
    """
    Mixin that provides the conjugate gradient direction computation.

    Supports formulas: FR (Fletcher-Reeves), PR (Polak-Ribière), PRP+ (positive
    version), HS (Hestenes-Stiefel), DY (Dai-Yuan).

    Parameters
    ----------
    cg_method : str, default="PRP+"
        Formula name.
    """

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
    Non-linear conjugate gradient optimizer with fixed learning rate.

    Uses the Fletcher-Reeves, Polak-Ribière, etc. formulas to compute the
    search direction without explicit curvature.

    Parameters
    ----------
    params : Params
        Parameter tensors.
    lr_init : float, default=1.0
        Initial learning rate.
    lr_method : str or None, default="lipschitz"
        Learning rate initialization method.
    cg_method : str, default="PRP+"
        Conjugate gradient formula: "FR", "PR", "PRP+", "HS", "DY".
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
    Non-linear conjugate gradient with line search.

    Combines conjugate gradient direction with a line-search to determine
    the step length.

    Parameters
    ----------
    params : Params
        Parameter tensors.
    lr_init : float, default=1.0
        Initial learning rate.
    lr_method : str or None, default=None
        Learning rate initialization method.
    c1 : float, default=1e-4
        Sufficient decrease parameter (Armijo).
    c2 : float, default=0.9
        Curvature condition parameter (Wolfe).
    tau : float, default=0.1
        Step reduction factor for backtracking.
    max_iter : int, default=20
        Maximum iterations for line search.
    tol : float, default=1e-8
        Tolerance (e.g., minimum step).
    line_search_method : str, default="backtrack"
        Line-search method (see create_line_search_solver).
    line_search_cond : str, default="armijo"
        Line-search stopping condition.
    cg_method : str, default="PRP+"
        Conjugate gradient formula.
    """

    def __init__(
        self,
        params: Params,
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
