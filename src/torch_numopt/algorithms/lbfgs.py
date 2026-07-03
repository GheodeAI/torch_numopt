from __future__ import annotations
import torch
import torch.nn as nn
from ..line_search import create_line_search_solver
from ..numerical_optimizer import NumericalOptimizer, LineSearchOptimizer
from ..curvature import NaiveIdentityCalculator
from ..utils import param_add, param_diff, param_dot, param_scalar_prod, param_copy, param_neg, Params
from ..objective import ObjectiveFunction


class LBFGSMixin:
    def __init__(self, *args, memory_size: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.s = []
        self.y = []
        self.gamma = 1
        self.memory_size = memory_size

    def get_step_direction(self, objective: Params, grad_params: tuple):
        if self.prev_params is None:
            return param_neg(grad_params)

        eps = torch.finfo(grad_params[0].dtype).eps
        q = grad_params
        alphas = []
        for s, y in zip(reversed(self.s), reversed(self.y)):
            alpha = param_dot(s, q) / (param_dot(s, y) + eps)
            q = param_diff(q, param_scalar_prod(alpha, y))
            alphas.append(alpha)

        r = param_scalar_prod(self.gamma, q)
        for s, y, alpha in zip(self.s, self.y, reversed(alphas)):
            beta = param_dot(y, r) / (param_dot(s, y) + eps)
            r = param_add(r, param_scalar_prod(alpha - beta, s))

        if len(self.s) > 0:
            self.gamma = param_dot(self.s[-1], self.y[-1]) / (param_dot(self.y[-1], self.y[-1]) + eps)

        return param_neg(r)

    def apply_gradients(self, objective: ObjectiveFunction, params: Params, grad_params: Params):
        old_params = param_copy(params)

        super().apply_gradients(objective, params, grad_params)

        with torch.enable_grad():
            new_loss = objective.loss(*params)
            new_grad = torch.autograd.grad(new_loss, params, create_graph=False, retain_graph=False)

        new_s = param_diff(params, old_params)
        new_y = param_diff(new_grad, grad_params)

        if param_dot(new_s, new_y) > 1e-12:
            self.s.append(new_s)
            self.y.append(new_y)

        if len(self.s) > self.memory_size:
            self.s.pop(0)
            self.y.pop(0)

        return params


class LBFGS(LBFGSMixin, NumericalOptimizer):
    """
    Limited-memory BFGS optimizer with fixed learning rate.

    Maintains a history of past updates (s, y) to approximate the inverse Hessian.

    Parameters
    ----------
    params : Params
        Parameter tensors.
    lr_init : float, default=1.0
        Initial learning rate.
    lr_method : str or None, default=None
        Learning rate initialization method.
    memory_size : int, default=10
        Number of past updates to store.
    """

    def __init__(
        self,
        params: Params,
        lr_init: float = 1.0,
        lr_method: str | None = None,
        memory_size: int = 10,
    ):
        super().__init__(params, curvature_estimator=NaiveIdentityCalculator(), lr_init=lr_init, lr_method=lr_method, memory_size=memory_size)


class LBFGSLS(LBFGSMixin, LineSearchOptimizer):
    """
    L-BFGS with line search.

    After computing the L-BFGS direction, a line search is performed to find
    an appropriate step length. This is the recommended way to use L-BFGS.

    Parameters
    ----------
    params : Params
        Parameter tensors.
    lr_init : float, default=1
        Initial learning rate.
    lr_method : str or None, default=None
        Learning-rate initialization method.
    c1, c2, tau, max_iter, tol : line-search parameters.
    memory_size : int, default=10
        Number of stored (s, y) pairs.
    line_search_method : str, default="interpolate"
        Line-search method (interpolate is often good for L-BFGS).
    line_search_cond : str, default="wolfe"
        Condition (Wolfe conditions are typical for L-BFGS).
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
        memory_size: int = 10,
        line_search_method: str = "interpolate",
        line_search_cond: str = "wolfe",
    ):

        super().__init__(
            params,
            curvature_estimator=NaiveIdentityCalculator(),
            lr_init=lr_init,
            lr_method=lr_method,
            line_search=create_line_search_solver(
                method=line_search_method, condition=line_search_cond, c1=c1, c2=c2, tau=tau, max_iter=max_iter, tol=tol
            ),
            memory_size=memory_size,
        )
