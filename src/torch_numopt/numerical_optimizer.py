""" """

from abc import ABC
from typing import Iterable
import logging
import torch
from torch.optim import Optimizer
from itertools import tee

from .utils import param_diff, param_scalar_prod, param_dot, param_neg, param_norm, param_scaled_add, param_copy, Params
from .line_search import LineSearchSolver
from .trust_region import TrustRegionSolver
from .curvature_estimator import CurvatureEstimator
from .solve_system import solve_system
from .objective import ObjectiveFunction

logger = logging.getLogger(__name__)

lr_init_methods = {"scaled", "BB1", "BB2", "quadratic", "lipschitz", "keep", None}


class NumericalOptimizer(Optimizer, ABC):
    """
    Base class for gradient-based optimization algorithms with line search.

    Parameters
    ----------
    model: nn.Module
    lr_init: float
        Maximum learning rate in backtracking line search, if the learning rate is set as constant, this will be the value used.
    lr_method: str
        Method to use to initialize the learning rate before applying line search.
    """

    def __init__(
        self,
        params: Params,
        curvature_estimator: CurvatureEstimator,
        lr_init: float = 1,
        lr_method: str | None = None,
        solver="solve",
        fix_ascent: bool = True,
    ):
        assert lr_init > 0, "Learning rate must be a positive number."

        params = tuple(params)
        super().__init__(params=params, defaults={})

        self.params = params
        self.lr_init = lr_init
        self.lr_method = lr_method
        self.curvature_estimator = curvature_estimator
        self.solver = solver
        self.fix_ascent = fix_ascent

        # Storage of previous solutions
        self.prev_lr = None
        self.prev_lr_init = lr_init
        self.prev_grad = None
        self.prev_step_dir = None
        self.prev_params = None
        self.prev_loss = None

        self.curr_loss = None
        self.curr_lr = None
        self.curr_lr_init = None
        self.curr_params = None
        self.curr_step_dir = None
        self.curr_grad = None

    def initialize_lr(self, lr: float, grad_params: Params, step_dir: Params, objective: ObjectiveFunction, params: Params):
        """

        Parameters
        ----------

        lr: float
        grad: Params
        step_dir: Params
        eval_model: Callable
        params: Params
        """

        if self.prev_lr is None:
            return lr

        prev_grad = self.prev_grad
        prev_step_dir = self.prev_step_dir
        prev_loss = self.prev_loss

        s = param_scalar_prod(self.prev_lr, prev_step_dir)
        y = param_diff(grad_params, prev_grad)

        new_lr = None
        eps = torch.finfo(params[0].dtype).eps
        match self.lr_method:
            case None:
                new_lr = lr
            case "keep":
                new_lr = self.prev_lr
            case "scaled":
                new_lr = self.prev_lr_init * (param_dot(prev_grad, prev_step_dir) / (eps + param_dot(grad_params, step_dir)))
            case "lipschitz":
                new_lr = param_norm(s) / (param_norm(y) + eps)
            case "BB1":
                # Barzilai-Borwein first formula
                new_lr = param_dot(s, s) / (eps + param_dot(s, y))
            case "BB2":
                # Barzilai-Borwein second formula
                new_lr = param_dot(s, y) / (eps + param_dot(y, y))
            case "quadratic":
                with torch.inference_mode():
                    loss = objective.loss(*params)
                new_lr = 2 * abs(loss - prev_loss) / (eps + abs(self.prev_lr * param_dot(prev_grad, prev_step_dir)))
                new_lr = min(1.01 * new_lr, lr)
            case _:
                lr_init_methods_str = ", ".join([f"'{i}'" if i is not None else "None" for i in lr_init_methods])
                last_comma_idx = lr_init_methods_str.rfind(",")
                lr_init_methods_str = lr_init_methods_str[:last_comma_idx] + " or" + lr_init_methods_str[last_comma_idx + 1 :]
                raise ValueError(f"Learning rate initialization method {self.lr_method} does not exist. Try {lr_init_methods_str}.")

        if new_lr < 0:
            logger.error("Estimated lr (%g) will yield an ascent direction. Falling back to guess %g", new_lr, lr)
            new_lr = lr

        logger.info("Initial lr generated = %g with method %s and initial guess %g.", new_lr, self.lr_method, lr)

        return new_lr

    def get_step_direction(self, objective, grad_params):
        step_direction = solve_system(self.curvature_estimator, objective, grad_params, solver=self.solver)
        return param_neg(step_direction)

    def apply_gradients(self, objective: ObjectiveFunction, params: Params, grad_params: Params):
        """
        Updates the parameters of the network using a direction and a step length.

        Parameters
        ----------

        lr: float
        eval_model: Callable
        params: Params
        grad_params: Params
        """

        step_dir = self.get_step_direction(objective, grad_params)
        if self.fix_ascent and param_dot(grad_params, step_dir) > 1e-8:
            logger.warning("Ascent direction detected, falling back to steepest descent. ")
            step_dir = param_neg(grad_params)
        lr = self.initialize_lr(self.lr_init, grad_params, step_dir, objective, params)

        new_params = param_scaled_add(params, step_dir, scale=lr)

        # Apply new parameters
        with torch.no_grad():
            for param, new_param in zip(params, new_params):
                param.copy_(new_param)

        with torch.inference_mode():
            self.curr_loss = objective.loss(*new_params)
        self.curr_lr = lr
        self.curr_lr_init = lr
        self.curr_params = new_params
        self.curr_step_dir = step_dir
        self.curr_grad = grad_params

    def update(self):
        self.prev_lr = self.curr_lr
        self.prev_lr_init = self.curr_lr_init
        self.prev_params = param_copy(self.curr_params)
        self.prev_step_dir = self.curr_step_dir
        self.prev_grad = self.curr_grad
        self.prev_loss = self.curr_loss

    def step(self, objective: ObjectiveFunction):
        """
        Method to update the parameters of the Neural Network.

        Parameters
        ----------

        closure: ObjectiveFunction
        """

        objective.closure()

        for group in self.param_groups:
            # Calculate gradients
            params_with_grad = []
            gradient = []
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    gradient.append(p.grad)

            self.apply_gradients(
                objective=objective,
                params=tuple(params_with_grad),
                grad_params=tuple(gradient),
            )

        self.update()


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
        params: Iterable[torch.Tensor],
        curvature_estimator: CurvatureEstimator,
        line_search: LineSearchSolver,
        lr_init: float = 1,
        lr_method: str | None = None,
        solver="solve",
    ):
        super().__init__(params=params, curvature_estimator=curvature_estimator, lr_init=lr_init, lr_method=lr_method, solver=solver)
        self.line_search = line_search

    def apply_gradients(self, objective: ObjectiveFunction, params: Params, grad_params: Params):
        """
        Updates the parameters of the network using a direction and a step length.

        Parameters
        ----------

        lr: float
        eval_model: Callable
        params: Params
        grad_params: Params
        """

        step_dir = self.get_step_direction(objective, grad_params)
        if self.fix_ascent and param_dot(grad_params, step_dir) > 0:
            logger.warning("Ascent direction detected, falling back to steepest descent. ")
            step_dir = param_neg(grad_params)

        lr_init = self.initialize_lr(self.lr_init, grad_params, step_dir, objective, params)

        new_params, lr = self.line_search(params, step_dir, grad_params, lr_init, objective)

        # Apply new parameters
        with torch.no_grad():
            for param, new_param in zip(params, new_params):
                param.copy_(new_param)

        with torch.inference_mode():
            self.curr_loss = objective.loss(*new_params)
        self.curr_lr = lr
        self.curr_lr_init = lr_init
        self.curr_params = new_params
        self.curr_step_dir = step_dir
        self.curr_grad = grad_params


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
        params: Params,
        trust_region: TrustRegionSolver,
        radius_init: float = 1.0,
        accept_tol: float = 0.1,
        solver="solve",
    ):
        super().__init__(params=params, curvature_estimator=None, lr_init=radius_init, solver=solver)
        self.trust_region = trust_region
        self.accept_tol = accept_tol

    def update_model_radius(self, objective, radius, radius_init, loss, params, grad_params, new_loss, step_dir):
        """
        Update the model radius from the loss in the current iteration.
        """

        eps = torch.finfo(loss.dtype).eps

        m_0 = self.trust_region.model(objective, 0, params, loss, grad_params)
        m_p = self.trust_region.model(objective, step_dir, params, loss, grad_params)

        rho = (loss - new_loss) / (m_0 - m_p + eps)

        if rho < 0.25:
            radius *= 0.25
        elif rho > 0.75 and torch.isclose(param_norm(step_dir), torch.tensor(radius, dtype=loss.dtype), rtol=1e-8, atol=1e-10):
            radius = min(2 * radius, radius_init)

        logger.debug(f"[TR] ρ={rho:+.4f}  Δ={radius:8.6f}  loss {loss:.6f} → {new_loss:.6f}")

        return rho, radius

    def apply_gradients(self, objective: ObjectiveFunction, params: Params, grad_params: Params):
        """
        Updates the parameters of the network using a direction and a step length.

        Parameters
        ----------

        lr: float
        objective: ObjectiveFunction
        params: Params
        grad_params: Params
        """

        prev_loss = objective.loss(*params)
        model_radius = self.lr_init if self.prev_lr is None else self.prev_lr

        logging.info("Starting trust region loop with radius %g.", model_radius)
        new_params, step_dir = self.trust_region.optimize_model(objective, params, model_radius, grad_params)
        if self.fix_ascent and param_dot(grad_params, step_dir) > 0:
            logger.warning("Ascent direction detected, falling back to steepest descent. ")
            step_dir = param_neg(grad_params)

        with torch.inference_mode():
            new_loss = objective.loss(*new_params)

        rho, model_radius = self.update_model_radius(objective, model_radius, self.lr_init, prev_loss, params, grad_params, new_loss, step_dir)

        logging.info("Finished trust region search, rho = %g, final model radius = %g.", rho, model_radius)

        # Apply new parameters
        if rho > self.accept_tol:
            with torch.no_grad():
                for param, new_param in zip(params, new_params):
                    param.copy_(new_param)

            with torch.inference_mode():
                self.curr_loss = objective.loss(*new_params)
                self.curr_params = new_params
            self.curr_step_dir = step_dir
            self.curr_grad = grad_params
        elif self.curr_params is None:
            with torch.inference_mode():
                self.curr_loss = objective.loss(*params)
                self.curr_params = params
            self.curr_step_dir = step_dir
            self.curr_grad = grad_params

        self.curr_lr_init = self.lr_init
        self.curr_lr = model_radius
