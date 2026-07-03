"""
AdaHessian optimizer (diagonal Hessian with momentum).

This module implements the AdaHessian algorithm, which combines the adaptive
learning rate mechanism of Adam with a diagonal Hessian approximation computed
via Hutchinson's method. It maintains moving averages of both the gradient and
the Hessian diagonal, and uses them to compute a preconditioned step direction.
"""

from __future__ import annotations
import torch
from ..utils import param_reshape_like, Params
from ..line_search import create_line_search_solver
from ..numerical_optimizer import NumericalOptimizer, LineSearchOptimizer
from ..curvature import HutchinsonDiagonalApproximation


class AdaHessianMixin:
    """
    Mixin that implements the AdaHessian algorithm.

    AdaHessian uses a diagonal Hessian approximation (via Hutchinson's method)
    and maintains moving averages of the gradient and the squared diagonal Hessian.

    Parameters
    ----------
    beta1 : float, default=0.9
        Exponential decay rate for the gradient moment.
    beta2 : float, default=0.999
        Exponential decay rate for the Hessian diagonal moment.
    k : float, default=1
        Exponent used in the denominator; typically 0.5 for AdaHessian (root),
        but here set to 1 to allow flexibility.
    eps : float, default=1e-4
        Small constant for numerical stability in the division.
    """

    def __init__(self, *args, beta1=0.9, beta2=0.999, k: float = 1, eps: float = 1e-8, **kwargs):
        super().__init__(*args, **kwargs)

        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1_acc = beta1
        self.beta2_acc = beta2

        self.prev_first_moment = 0
        self.prev_hess_moment = 0
        self.k = k
        self.eps = eps

    def get_step_direction(self, objective, grad_params):
        """ """
        h_params = self.curvature_estimator.scaling_matrix(objective, objective.params)

        grad = torch.hstack([i.flatten() for i in grad_params])
        h_diag = torch.hstack([i.flatten() for i in h_params])
        eps = self.eps

        # Calculate first unbiased moment of the gradient
        first_moment = self.beta1 * self.prev_first_moment + (1 - self.beta1) * grad
        first_moment_unbias = first_moment / (1 - self.beta1_acc)
        self.prev_first_moment = first_moment
        self.beta1_acc *= self.beta1

        # Calculate second unbiased moment of the hessian diagonal
        hess_moment = self.beta2 * self.prev_hess_moment + (1 - self.beta2) * h_diag * h_diag
        hess_moment_unbias = hess_moment / (1 - self.beta2_acc)
        self.prev_hess_moment = hess_moment
        self.beta2_acc *= self.beta2

        # Calculate the next step direction
        next_dir_flat = -first_moment_unbias / (hess_moment_unbias ** (0.5 * self.k) + eps)

        return param_reshape_like(next_dir_flat, grad_params)


class AdaHessian(AdaHessianMixin, NumericalOptimizer):
    """
    AdaHessian optimizer (diagonal Hessian with momentum).

    Uses Hutchinson diagonal Hessian approximation and momentum for both
    gradient and Hessian diagonal, similar to Adam but using second-order
    information.

    Parameters
    ----------
    params : Params
        Parameter tensors.
    lr_init : float, default=1
        Initial learning rate.
    lr_method : str or None, default=None
        Learning rate initialization method.
    beta1 : float, default=0.9
        Exponential decay rate for the first moment estimate (gradient).
    beta2 : float, default=0.999
        Exponential decay rate for the second moment estimate (Hessian diagonal).
    k : float, default=1
        Exponent for the Hessian diagonal in the step calculation (0.5 for AdaHessian).
    eps : float, default=1e-4
        Small constant for numerical stability.
    n_samples : int, default=5
        Number of Hutchinson samples for diagonal estimation.
    """

    def __init__(
        self,
        params: Params,
        lr_init: float = 1e-2,
        lr_method: str | None = None,
        beta1=0.9,
        beta2=0.999,
        k: float = 1,
        eps: float = 1e-8,
        n_samples: int = 10,
    ):
        super().__init__(
            params,
            curvature_estimator=HutchinsonDiagonalApproximation(n_samples=n_samples),
            lr_init=lr_init,
            lr_method=lr_method,
            beta1=beta1,
            beta2=beta2,
            k=k,
            eps=eps,
            fix_ascent=False,
        )


class AdaHessianLS(AdaHessianMixin, LineSearchOptimizer):
    """
    AdaHessian with line search.

    Same as AdaHessian, but instead of a fixed learning rate it performs a
    line search to determine the step length.
    
    Works well in practice, but theoretically it's not well supported.

    Parameters
    ----------
    params : Params
        Parameter tensors.
    lr_init : float, default=1
        Initial learning rate.
    lr_method : str or None, default=None
        Learning-rate initialization method.
    beta1, beta2, k, eps, n_samples : same as in AdaHessian.
    c1, c2, tau, max_iter, tol : line-search parameters.
    line_search_method : str, default="backtrack"
        Line-search method.
    line_search_cond : str, default="armijo"
        Line-search condition.
    """

    def __init__(
        self,
        params: Params,
        lr_init: float = 1,
        lr_method: str | None = None,
        beta1=0.9,
        beta2=0.999,
        k: float = 1,
        eps: float = 1e-8,
        n_samples: int = 10,
        c1: float = 1e-4,
        c2: float = 0.9,
        tau: float = 0.1,
        max_iter: int = 20,
        tol: float = 1e-8,
        line_search_method: str = "backtrack",
        line_search_cond: str = "armijo",
    ):
        super().__init__(
            params,
            curvature_estimator=HutchinsonDiagonalApproximation(n_samples=n_samples),
            lr_init=lr_init,
            lr_method=lr_method,
            line_search=create_line_search_solver(
                method=line_search_method, condition=line_search_cond, c1=c1, c2=c2, tau=tau, max_iter=max_iter, tol=tol
            ),
            beta1=beta1,
            beta2=beta2,
            k=k,
            eps=eps,
        )

class DiagonalNewtonLS(LineSearchOptimizer):
    """
    Diagonal Newton approximation (using Hutchinson's method) with line search.

    Parameters
    ----------
    params : Params
        Parameter tensors.
    lr_init : float, default=1
        Initial learning rate.
    lr_method : str or None, default=None
        Learning-rate initialization method.
    beta1, beta2, k, eps, n_samples : same as in AdaHessian.
    c1, c2, tau, max_iter, tol : line-search parameters.
    line_search_method : str, default="backtrack"
        Line-search method.
    line_search_cond : str, default="armijo"
        Line-search condition.
    """

    def __init__(
        self,
        params: Params,
        lr_init: float = 1,
        lr_method: str | None = None,
        n_samples: int = 10,
        c1: float = 1e-4,
        c2: float = 0.9,
        tau: float = 0.1,
        max_iter: int = 20,
        tol: float = 1e-8,
        line_search_method: str = "backtrack",
        line_search_cond: str = "armijo",
    ):
        super().__init__(
            params,
            curvature_estimator=HutchinsonDiagonalApproximation(n_samples=n_samples),
            lr_init=lr_init,
            lr_method=lr_method,
            line_search=create_line_search_solver(
                method=line_search_method, condition=line_search_cond, c1=c1, c2=c2, tau=tau, max_iter=max_iter, tol=tol
            ),
        )