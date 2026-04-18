""" """

from abc import ABC, abstractmethod
from typing import Callable, Iterable
import torch
import torch.nn as nn
from torch.func import functional_call
from .utils import fix_stability, pinv_svd_trunc
from .custom_optimizer import CustomOptimizer
from .scaling_matrix_calculator import ScalingMatrixCalculator
from .line_search import LineSearchSolver

lr_init_methods = ["scaled", "BB1", "BB2", "quadratic", "lipschitz", "keep", None]


class NumericalOptimizer(CustomOptimizer, ABC):
    """
    Base class for gradient-based optimization algorithms with line search.

    Parameters
    ----------
    model: nn.Module
    lr_init: float
        Maximum learning rate in backtracking line search, if the learning rate is set as constant, this will be the value used.
    lr_method: str
        Method to use to initialize the learning rate before applying line search.
    line_search_cond: str (optional)
    line_search_method: str (optional)
    c1: float (optional)
    c2: float (optional)
    tau: float (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        scaling_matrix: ScalingMatrixCalculator,
        lr_init: float = 1,
        lr_method: str | None = None,
        solver="solve",
    ):
        assert lr_init > 0, "Learning rate must be a positive number."

        super().__init__(model.parameters(), {"lr": lr_init})

        self.lr_init = lr_init
        self.lr_method = lr_method
        self.solver = solver

        self.prev_lr = None
        self.prev_grad = None
        self.prev_step_dir = None
        self.prev_params = None

        self._model = model
        self._param_keys = dict(model.named_parameters()).keys()
        self._params = self.param_groups[0]["params"]

        self.scaling_matrix = scaling_matrix

    def initialize_lr(self, lr: float, grad: list, step_dir: list, eval_model: Callable, params: list):
        """

        Parameters
        ----------

        lr: float
        grad: list
        step_dir: list
        eval_model: Callable
        params: list
        """

        if self.prev_lr is None:
            return lr

        grad_flat = torch.hstack([i.flatten() for i in grad])
        step_flat = torch.hstack([i.flatten() for i in step_dir])
        prev_grad_flat = torch.hstack([i.flatten() for i in self.prev_grad])
        prev_step_flat = torch.hstack([i.flatten() for i in self.prev_step_dir])

        new_lr = None
        eps = torch.finfo(params[0].dtype).eps
        match self.lr_method:
            case "scaled":
                new_lr = self.prev_lr_init * (prev_grad_flat @ prev_step_flat) / (grad_flat @ step_flat + eps)
            # Barzilai-Borwein
            case "BB1":
                new_lr = (prev_step_flat @ prev_step_flat) / (prev_step_flat @ prev_grad_flat + eps)
            case "BB2":
                new_lr = (prev_step_flat @ prev_grad_flat) / (prev_grad_flat @ prev_grad_flat + eps)
            case "quadratic":
                loss = eval_model(*params)
                new_lr = 2 * abs(loss - self.prev_loss) / (prev_grad_flat @ prev_step_flat + eps)
                new_lr = min(1.01 * new_lr, 1)
            case "lipschitz":
                grad_dist = torch.norm(grad_flat - prev_grad_flat)
                step_dist = torch.norm(step_flat - prev_step_flat)
                new_lr = step_dist / (grad_dist + eps)
            case "keep":
                new_lr = self.prev_lr
            case None:
                new_lr = lr
            case _:
                lr_init_methods_str = ", ".join([f"'{i}'" if i is not None else "None" for i in lr_init_methods])
                last_comma_idx = lr_init_methods_str.rfind(",")
                lr_init_methods_str = lr_init_methods_str[:last_comma_idx] + " or" + lr_init_methods_str[last_comma_idx + 1 :]
                raise ValueError(f"Learning rate initialization method {self.lr_init} does not exist. Try {lr_init_methods_str}.")

        return new_lr

    def apply_gradients(self, eval_model: Callable, params: list, d_p_list: list, h_list: list):
        """
        Updates the parameters of the network using a direction and a step length.

        Parameters
        ----------

        lr: float
        eval_model: Callable
        params: list
        d_p_list: list
        h_list: list, optional
        """

        step_dir = self.get_step_direction(d_p_list, h_list)
        lr_init = self.initialize_lr(self.lr_init, d_p_list, step_dir, eval_model, params)

        lr = lr_init
        new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))

        self.prev_lr = lr
        self.prev_lr_init = lr_init
        self.prev_params = params
        self.prev_step_dir = step_dir
        self.prev_grad = d_p_list
        self.prev_loss = eval_model(*params)

        # Apply new parameters
        for param, new_param in zip(params, new_params):
            with torch.no_grad():
                param.copy_(new_param)

    def get_step_direction(self, d_p_list, h_list) -> Iterable:
        eps = torch.finfo(d_p_list[0].dtype).eps
        if h_list is None:
            return d_p_list

        if h_list[0].ndim == 2:
            dir_list = [None] * len(d_p_list)
            for i, (d_p, h) in enumerate(zip(d_p_list, h_list)):
                if torch.linalg.cond(h) > 1e8:
                    h = fix_stability(h)

                match self.solver:
                    case "pinv":
                        h_inv = h.pinverse()
                        d2_p = (h_inv @ d_p.flatten()).reshape(d_p.shape)
                    case "pinv-trunc":
                        h_inv = pinv_svd_trunc(h)
                        d2_p = (h_inv @ d_p.flatten()).reshape(d_p.shape)
                    case "solve":
                        d2_p = torch.linalg.solve(h, d_p.flatten()).reshape(d_p.shape)

                dir_list[i] = d2_p

        elif h_list[0].ndim == 1:
            dir_list = [d_p / (h + eps) for d_p, h in zip(d_p_list, h_list)]
        elif h_list[0].ndim == 0:
            h = h_list[0]
            dir_list = [d_p / (h + eps) for d_p in d_p_list]
        elif h_list is None:
            raise ValueError("Incorrectly dimensioned hessian.")

        return dir_list

    @torch.no_grad()
    def step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: nn.Module,
    ):
        """
        Method to update the parameters of the Neural Network.

        Parameters
        ----------

        x: torch.Tensor
            Inputs of the Neural Network.
        y: torch.Tensor
            Targets of the Neural Network.
        loss_fn: nn.Module
            Loss function to be optimized.
        """

        device = self._params[0].device
        x = x.to(device)
        y = y.to(device)

        def eval_model(*input_params):
            out = functional_call(self._model, dict(zip(self._param_keys, input_params)), x)
            return loss_fn(out, y)

        # Calculate exact Hessian matrix
        h_list = self.scaling_matrix(x, y, loss_fn)

        for group in self.param_groups:
            # Calculate gradients
            params_with_grad = []
            d_p_list = []
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            self.apply_gradients(
                params=params_with_grad,
                d_p_list=d_p_list,
                h_list=h_list,
                eval_model=eval_model,
            )


class LineSearchOptimizer(NumericalOptimizer, ABC):
    """
    Base class for gradient-based optimization algorithms with line search.

    Parameters
    ----------
    model: nn.Module
    lr_init: float
        Maximum learning rate in backtracking line search, if the learning rate is set as constant, this will be the value used.
    lr_method: str
        Method to use to initialize the learning rate before applying line search.
    line_search_cond: str (optional)
    line_search_method: str (optional)
    c1: float (optional)
    c2: float (optional)
    tau: float (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        scaling_matrix: ScalingMatrixCalculator,
        line_search: LineSearchSolver,
        lr_init: float = 1,
        lr_method: str | None = None,
        solver="solve",
    ):
        super().__init__(model=model, scaling_matrix=scaling_matrix, lr_init=lr_init, lr_method=lr_method, solver=solver)

        self.line_search = line_search

    def apply_gradients(self, eval_model: Callable, params: list, d_p_list: list, h_list: list):
        """
        Updates the parameters of the network using a direction and a step length.

        Parameters
        ----------

        lr: float
        eval_model: Callable
        params: list
        d_p_list: list
        h_list: list, optional
        """

        step_dir = self.get_step_direction(d_p_list, h_list)
        lr_init = self.initialize_lr(self.lr_init, d_p_list, step_dir, eval_model, params)

        new_params, lr = self.line_search(params, step_dir, d_p_list, lr_init, eval_model)

        self.prev_lr = lr
        self.prev_lr_init = lr_init
        self.prev_params = params
        self.prev_step_dir = step_dir
        self.prev_grad = d_p_list
        self.prev_loss = eval_model(*params)

        # Apply new parameters
        for param, new_param in zip(params, new_params):
            with torch.no_grad():
                param.copy_(new_param)
