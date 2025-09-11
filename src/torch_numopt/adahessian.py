from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn
from torch.autograd.functional import hessian
from torch.func import functional_call
from .utils import param_reshape_like, param_flatten
from .second_order_optimizer import SecondOrderOptimizer
from .utils import fix_stability, pinv_svd_trunc


class AdaHessian(SecondOrderOptimizer):
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
    """

    def __init__(
        self,
        model: nn.Module,
        lr_init: float = 1,
        lr_method: str = None,
        beta1 = 0.9,
        beta2 = 0.999,
        c1: float = 1e-4,
        c2: float = 0.9,
        tau: float = 0.1,
        k: float = 1,
        line_search_method: str = "const",
        line_search_cond: str = "armijo",
        **kwargs,
    ):
        super().__init__(
            model,
            lr_init=lr_init,
            lr_method=lr_method,
            line_search_cond=line_search_cond,
            line_search_method=line_search_method,
            c1=c1,
            c2=c2,
            tau=tau,
            batch_size=None,
        )

        self.samples = 5
        self.skip_iters = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1_acc = beta1
        self.beta2_acc = beta2

        self.prev_first_moment = 0
        self.prev_hess_moment = 0
        self.k = k
    

    def get_step_direction(self, d_p_list, h_list):
        """ """
        grad = torch.hstack([i.flatten() for i in d_p_list])
        h_diag = torch.hstack([i.flatten() for i in h_list])
        eps = torch.finfo(grad.dtype).eps

        # Calculate first unbiased moment of the gradient
        first_moment = self.beta1 * self.prev_first_moment + (1 - self.beta1) * grad
        self.prev_first_moment = first_moment
        first_moment_unbias = first_moment / (1 - self.beta1_acc)
        self.beta1_acc *= self.beta1

        # Calculate second unbiased moment of the hessian diagonal
        hess_moment = self.beta2 * self.prev_hess_moment + (1 - self.beta2) * h_diag * h_diag
        self.prev_hess_moment = hess_moment
        hess_moment_unbias = hess_moment / (1 - self.beta2_acc)
        self.beta2_acc *= self.beta2

        # Calculate the next step direction
        next_dir_flat = first_moment_unbias / (hess_moment_unbias**(0.5*self.k) + eps)

        next_dir = param_reshape_like(next_dir_flat, d_p_list)
        return next_dir


    def get_scaling_matrix(self, 
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: nn.Module
    ):
        return self.hutchinson_diagonal(x, y, loss_fn, n_samples=self.samples, vectorize=True)