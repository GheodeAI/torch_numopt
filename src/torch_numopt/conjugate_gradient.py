from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn
from torch.func import functional_call
from .line_search_optimizer import LineSearchOptimizer
from .custom_optimizer import CustomOptimizer
from copy import copy, deepcopy
from .utils import param_reshape_like


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
        self.cg_method = cg_method

    def get_step_direction(self, d_p_list, h_list=None):
        """ """

        if self.prev_grad is None:
            return d_p_list

        grad = torch.hstack([i.flatten() for i in d_p_list])
        prev_grad = torch.hstack([i.flatten() for i in self.prev_grad])
        prev_step = torch.hstack([i.flatten() for i in self.prev_step_dir])

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
        next_dir = param_reshape_like(grad - beta * res , d_p_list)
        return next_dir

    def get_scaling_matrix(self, 
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: nn.Module
    ):
        return None