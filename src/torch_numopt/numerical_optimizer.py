""" """

from abc import ABC
from typing import Callable, Iterable
import logging
import torch
import torch.nn as nn
from torch.func import functional_call
from .utils import fix_stability, pinv_svd_trunc, param_sub, param_scalar_prod
from .custom_optimizer import CustomOptimizer
from .line_search import LineSearchSolver
from .trust_region import TrustRegionSolver
from .curvature_estimator import CurvatureEstimator
from .solve_system import solve_system

logger = logging.getLogger(__name__)

lr_init_methods = {"scaled", "BB1", "BB2", "quadratic", "lipschitz", "keep", None}


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
        curvature_estimator: CurvatureEstimator,
        lr_init: float = 1,
        lr_method: str | None = None,
        solver="solve",
    ):
        assert lr_init > 0, "Learning rate must be a positive number."

        super().__init__(model.parameters(), {"lr": lr_init})

        self.lr_init = lr_init
        self.lr_method = lr_method
        self.curvature_estimator = curvature_estimator
        self.solver = solver

        self.model = model
        self.param_keys = dict(model.named_parameters()).keys()
        self.params = self.param_groups[0]["params"]

        # Storage of previous solutions
        self.prev_lr_ = None
        self.prev_lr_init_ = None
        self.prev_grad_ = None
        self.prev_step_dir_ = None
        self.prev_params_ = None
        self.prev_loss_ = None

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

        if self.prev_lr_ is None:
            return lr

        grad_flat = torch.hstack([i.ravel() for i in grad])
        step_flat = torch.hstack([i.ravel() for i in step_dir])
        prev_grad_flat = torch.hstack([i.ravel() for i in self.prev_grad_])
        prev_step_flat = torch.hstack([i.ravel() for i in self.prev_step_dir_])

        new_lr = None
        eps = torch.finfo(params[0].dtype).eps
        match self.lr_method:
            case "scaled":
                new_lr = self.prev_lr_init_ * (prev_grad_flat @ prev_step_flat) / (grad_flat @ step_flat + eps)
            case "BB1":
                # Barzilai-Borwein first formula
                new_lr = (prev_step_flat @ prev_step_flat) / (prev_step_flat @ prev_grad_flat + eps)
            case "BB2":
                # Barzilai-Borwein second formula
                new_lr = (prev_step_flat @ prev_grad_flat) / (prev_grad_flat @ prev_grad_flat + eps)
            case "quadratic":
                loss = eval_model(*params)
                new_lr = 2 * abs(loss - self.prev_loss_) / (prev_grad_flat @ prev_step_flat + eps)
                new_lr = min(1.01 * new_lr, lr)
            case "lipschitz":
                grad_dist = torch.norm(grad_flat - prev_grad_flat)
                step_dist = torch.norm(step_flat - prev_step_flat)
                new_lr = step_dist / (grad_dist + eps)
            case "keep":
                new_lr = self.prev_lr_
            case None:
                new_lr = lr
            case _:
                lr_init_methods_str = ", ".join([f"'{i}'" if i is not None else "None" for i in lr_init_methods])
                last_comma_idx = lr_init_methods_str.rfind(",")
                lr_init_methods_str = lr_init_methods_str[:last_comma_idx] + " or" + lr_init_methods_str[last_comma_idx + 1 :]
                raise ValueError(f"Learning rate initialization method {self.lr_init} does not exist. Try {lr_init_methods_str}.")

        logger.info("Initial lr generated = %g with method %s and initial guess %g.", new_lr, self.lr_method, lr)

        return new_lr

    def apply_gradients(self, eval_model: Callable, params: list, d_p_list: list):
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

        step_dir = solve_system(self.curvature_estimator, d_p_list, solver=self.solver)
        lr = self.initialize_lr(self.lr_init, d_p_list, step_dir, eval_model, params)

        new_params = param_sub(params, param_scalar_prod(lr, step_dir))

        # Apply new parameters
        for param, new_param in zip(params, new_params):
            with torch.no_grad():
                param.copy_(new_param)

        self.prev_lr_ = lr
        self.prev_params_ = new_params
        self.prev_step_dir_ = step_dir
        self.prev_grad_ = d_p_list
        self.prev_loss_ = eval_model(*new_params)

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

        device = self.params[0].device
        x = x.to(device)
        y = y.to(device)

        def eval_model(*input_params):
            out = functional_call(self.model, dict(zip(self.param_keys, input_params)), x)
            return loss_fn(out, y)

        # Calculate exact Hessian matrix
        self.curvature_estimator.store_data(x, y, loss_fn)

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
        curvature_estimator: CurvatureEstimator,
        line_search: LineSearchSolver,
        lr_init: float = 1,
        lr_method: str | None = None,
        solver="solve",
    ):
        super().__init__(model=model, curvature_estimator=curvature_estimator, lr_init=lr_init, lr_method=lr_method, solver=solver)

        self.line_search = line_search

    def apply_gradients(self, eval_model: Callable, params: list, d_p_list: list):
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

        step_dir = solve_system(self.curvature_estimator, d_p_list, solver=self.solver)
        lr_init = self.initialize_lr(self.lr_init, d_p_list, step_dir, eval_model, params)

        new_params, lr = self.line_search(params, step_dir, d_p_list, lr_init, eval_model)

        # Apply new parameters
        for param, new_param in zip(params, new_params):
            with torch.no_grad():
                param.copy_(new_param)

        self.prev_lr_ = lr
        self.prev_lr_init_ = lr_init
        self.prev_params_ = new_params
        self.prev_step_dir_ = step_dir
        self.prev_grad_ = d_p_list
        self.prev_loss_ = eval_model(*new_params)


class TrustRegionOptimizer(NumericalOptimizer, ABC):
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
        curvature_estimator: CurvatureEstimator,
        trust_region: TrustRegionSolver,
        radius_init: float = 1.0,
        solver="solve",
    ):
        super().__init__(model=model, curvature_estimator=curvature_estimator, lr_init=radius_init, solver=solver)

        self.trust_region = trust_region

    def update_model_radius(self, radius, radius_init, loss, d_p_list, new_loss, step_dir):
        """
        Update the model radius from the loss in the current iteration.
        """

        m_0 = self.trust_region.model(0, loss, d_p_list)
        m_p = self.trust_region.model(step_dir, loss, d_p_list)

        rho = (loss - new_loss) / (m_0 - m_p)

        param_norm = sum(torch.sum(p**2) for p in step_dir)

        if rho < 0.25:
            radius *= 0.25
        elif rho > 0.75 and torch.isclose(param_norm, torch.tensor(radius), rtol=1e-8, atol=1e-10):
            radius = min(2 * radius, radius_init)

        return rho, radius

    def apply_gradients(self, eval_model: Callable, params: list, d_p_list: list):
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

        prev_loss = eval_model(*params)
        model_radius = self.lr_init if self.prev_lr_ is None else self.prev_lr_

        logging.info("Starting trust region loop with radius %g.", model_radius)
        a = self.trust_region.optimize_model(params, model_radius, d_p_list)
        new_params, step_dir = self.trust_region.optimize_model(params, model_radius, d_p_list)

        with torch.inference_mode():
            new_loss = eval_model(*new_params)

        rho, model_radius = self.update_model_radius(model_radius, self.lr_init, prev_loss, d_p_list, new_loss, step_dir)

        logging.info("Finished trust region search, rho = %g, final model radius = %g.", rho, model_radius)

        # Apply new parameters
        if rho > 0.01:
            for param, new_param in zip(params, new_params):
                with torch.no_grad():
                    param.copy_(new_param)

        self.prev_lr_ = model_radius
        self.prev_params_ = new_params
        self.prev_step_dir_ = step_dir
        self.prev_grad_ = d_p_list
        self.prev_loss_ = eval_model(*new_params)
