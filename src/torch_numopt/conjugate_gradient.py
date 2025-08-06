from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn
from torch.func import functional_call
from .line_search_optimizer import LineSearchOptimizer
from .custom_optimizer import CustomOptimizer
from copy import copy


class ConjugateGradient(LineSearchOptimizer):
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
        line_search_method: str = "backtrack",
        line_search_cond: str = "armijo",
        cg_method: str = "PRP+",
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
        )

        # Conjugate gradient memory
        self.prev_dir = None
        self.cg_method = cg_method

    def get_step_direction(self, d_p_list, h_list=None):
        """ """
        if self.prev_dir is None:
            return d_p_list

        next_grad = [None] * len(d_p_list)
        for idx, (res, prev_res) in enumerate(zip(d_p_list, self.prev_dir)):
            eps = torch.finfo(res.dtype).eps
            res = res.view((-1, 1))
            prev_res = prev_res.view((-1, 1))

            match cg_method:
                case "FR":
                    beta = (res.T @ res) / (prev_res.T @ prev_res + eps)
                case "PR":
                    beta = (res.T @ (res - prev_res)) / (prev_res.T @ prev_res + eps)
                case "PRP+":
                    beta = torch.relu((res.T @ (res - prev_res)) / (prev_res.T @ prev_res + eps))
                case _:
                    raise ValueError("Incorrect conjugate gradient method, try 'FR', 'PR' or 'PRP+'.")

            res_reshaped = res.view(next_grad[idx].shape)
            next_grad[idx].add_(res_reshaped, alpha=-beta)

        self.prev_dir = next_grad

        return next_grad

    @torch.no_grad()
    def step(self, x, y, loss_fn):
        def eval_model(*input_params):
            out = functional_call(self._model, dict(zip(self._param_keys, input_params)), x)
            return loss_fn(out, y)

        for group in self.param_groups:
            # Calculate gradients
            params_with_grad = []
            d_p_list = []
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            self.apply_gradients(params=params_with_grad, d_p_list=d_p_list, eval_model=eval_model)
