"""
Base classes for numerical optimizers that use curvature information.

This module provides the core ``NumericalOptimizer`` abstract class and its
subclasses for line-search and trust-region strategies. It handles parameter
updates, learning-rate initialization, and direction computation using a
curvature estimator.
"""

from abc import ABC
from typing import Iterable
import logging
import math
import torch
from torch.optim import Optimizer

from .utils import param_add, param_is_finite, param_dot, param_neg, param_norm, param_scaled_add, param_detach, Params, torch_to_float
from .step_initializer import StepSizeInitializer, ConstantStepSize, create_step_size_init
from .line_search import LineSearchSolver
from .trust_region import TrustRegionSolver
from .curvature_estimator import CurvatureEstimator
from .solve_system import solve_system
from .objective import ObjectiveFunction

logger = logging.getLogger(__name__)

lr_init_methods = {"scaled", "BB1", "BB2", "quadratic", "lipschitz", "keep", None}


class NumericalOptimizer(Optimizer, ABC):
    """
    Base optimizer that uses a curvature estimator to compute a step direction.

    Subclasses must implement the strategy for determining the step (line-search
    or trust-region). This class handles storage of previous iterates and
    learning-rate initialization methods.

    Parameters
    ----------
    params : Params
        Iterable of parameter tensors to optimize.
    curvature_estimator : CurvatureEstimator
        Object that provides the second-order (or approximate) curvature.
    lr_init : float, default=1
        Initial guess for the learning rate (or radius).
    lr_method : str or None, default=None
        Method for initializing the learning rate. Options:
        - ``None``: use the supplied ``lr_init`` directly.
        - ``"keep"``: reuse the previous learning rate.
        - ``"scaled"``: scale based on gradient and step direction.
        - ``"quadratic"``: use the curvature quadratic form.
        - ``"interpolate"``: interpolate from previous loss change.
        - ``"lipschitz"``: estimate Lipschitz constant.
        - ``"BB1"``, ``"BB2"``: Barzilai-Borwein formulas.
    lr_tol : float, default=1e-18
        Tolerance for the learning rate; if the estimated rate falls below this,
        it is replaced by the initial guess.
    solver : str, default="solve"
        Solver used to invert the curvature system (see :mod:`solve_system`).
    fix_ascent : bool, default=True
        If ``True``, detect and correct ascent directions (fall back to steepest
        descent).
    """

    def __init__(
        self,
        params: Params,
        curvature_estimator: CurvatureEstimator,
        lr_method: str | StepSizeInitializer = None,
        lr_init: float = 1,
        solver: str = "solve",
        fix_ascent: bool = True,
        min_lr: float = 0,
        max_lr: float = 100,
    ):
        params = tuple(params)
        super().__init__(params=params, defaults={})

        if lr_method is None:
            lr_method = ConstantStepSize(lr_init, curvature_estimator, min_lr, max_lr)
        elif isinstance(lr_method, str):
            lr_method = create_step_size_init(lr_method, lr_init, curvature_estimator, min_lr, max_lr)

        self.params = params
        self.lr_method = lr_method
        self.lr_init = lr_method.lr_init
        self.curvature_estimator = curvature_estimator
        self.solver = solver
        self.fix_ascent = fix_ascent

        # Storage of previous solutions
        self.prev_lr = None
        self.prev_lr_init = None
        self.prev_grad = None
        self.prev_step_dir = None
        self.prev_params = None
        self.prev_loss = None
        self.delta_loss = None

        self.reset = False

    def get_step_direction(self, objective: ObjectiveFunction, grad_params: Params):
        """
        Compute the un-scaled step direction by solving the system
        ``H * p = -grad`` (or an approximation).

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function.
        grad_params : Params
            Current gradient.

        Returns
        -------
        Params
            Step direction (not yet multiplied by learning rate).
        """

        return solve_system(self.curvature_estimator, objective, param_neg(grad_params), solver=self.solver)

    def apply_gradients(self, objective: ObjectiveFunction, params: Params, grad_params: Params):
        """
        Update parameters using the current gradient and curvature.

        This method computes a step direction, determines a step length
        (via learning-rate initialization or line-search/trust-region), and
        applies the update. It also updates stored previous iterates.

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function.
        params : Params
            Current parameters (in-place updated).
        grad_params : Params
            Current gradient.
        """

        step_dir = self.get_step_direction(objective, grad_params)
        if self.fix_ascent and param_dot(grad_params, step_dir) > 1e-8:
            logger.warning("Ascent direction detected, falling back to steepest descent. ")
            step_dir = param_neg(grad_params)
        lr = self.lr_method(objective, params, grad_params, self.prev_grad, step_dir, self.prev_step_dir, self.prev_lr, self.delta_loss)

        new_params = param_scaled_add(params, step_dir, scale=lr)

        with torch.inference_mode():
            new_loss = torch_to_float(objective.loss(*new_params))

        if param_is_finite(new_params) and math.isfinite(new_loss):
            with torch.no_grad():
                for param, new_param in zip(params, new_params):
                    param.copy_(new_param)
        else:
            logger.critical("NaN found in loss or parameters. Skipping iteration.")
            self.reset = True

        if not self.reset:
            if self.prev_loss is not None:
                self.delta_loss = new_loss - self.prev_loss
            self.prev_loss = new_loss
            self.prev_lr = torch_to_float(lr)
            self.prev_params = param_detach(new_params)
            self.prev_step_dir = param_detach(step_dir)
            self.prev_grad = param_detach(grad_params)
        else:
            self.prev_lr = None
            self.prev_grad = None
            self.prev_step_dir = None
            self.prev_params = None
            self.prev_loss = None
            self.delta_loss = None
            self.reset = False

    def step(self, objective: ObjectiveFunction):
        """
        Perform one optimization step.

        This is the main entry point called by the training loop. It calls
        ``objective.closure()`` to compute the loss and gradients, then calls
        ``apply_gradients`` for each parameter group.

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function.
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
                    p.grad = None

            self.apply_gradients(
                objective=objective,
                params=tuple(params_with_grad),
                grad_params=tuple(gradient),
            )

        if self.curvature_estimator is not None:
            self.curvature_estimator.update()


class LineSearchOptimizer(NumericalOptimizer, ABC):
    """
    Numerical optimizer that uses a line-search algorithm to determine the
    step length.

    Parameters
    ----------
    params : Params
        Parameter tensors.
    curvature_estimator : CurvatureEstimator
        Curvature estimator.
    line_search : LineSearchSolver
        Line-search solver (backtracking, interpolation, etc.).
    lr_init : float, default=1
        Initial learning-rate guess.
    lr_method : str or None, default=None
        Learning-rate initialization method.
    solver : str, default="solve"
        Linear solver for the step direction.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        curvature_estimator: CurvatureEstimator,
        line_search: LineSearchSolver,
        lr_method: str | StepSizeInitializer = None,
        lr_init: float = 1,
        solver: str = "solve",
        min_lr: float = 0,
        max_lr: float = 100,
    ):
        super().__init__(
            params=params, curvature_estimator=curvature_estimator, lr_init=lr_init, lr_method=lr_method, solver=solver, min_lr=min_lr, max_lr=max_lr
        )
        self.line_search = line_search

    def apply_gradients(self, objective: ObjectiveFunction, params: Params, grad_params: Params):
        step_dir = self.get_step_direction(objective, grad_params)
        if self.fix_ascent and param_dot(grad_params, step_dir) > 0:
            logger.warning("Ascent direction detected, falling back to steepest descent. ")
            step_dir = param_neg(grad_params)

        lr_init = self.lr_method(objective, params, grad_params, self.prev_grad, step_dir, self.prev_step_dir, self.prev_lr, self.delta_loss)

        new_params, lr = self.line_search.find_step_size(params, step_dir, grad_params, lr_init, objective)

        with torch.inference_mode():
            new_loss = torch_to_float(objective.loss(*new_params))

        if param_is_finite(new_params) and math.isfinite(new_loss):
            with torch.no_grad():
                for param, new_param in zip(params, new_params):
                    param.copy_(new_param)
        else:
            logger.critical("NaN found in loss or parameters. Skipping iteration.")
            self.reset = True

        if not self.reset:
            if self.prev_loss is not None:
                self.delta_loss = new_loss - self.prev_loss
            self.prev_loss = new_loss
            self.prev_lr = torch_to_float(lr)
            self.prev_lr_init = torch_to_float(lr_init)
            self.prev_params = param_detach(new_params)
            self.prev_step_dir = param_detach(step_dir)
            self.prev_grad = param_detach(grad_params)
        else:
            self.prev_lr = None
            self.prev_grad = None
            self.prev_step_dir = None
            self.prev_params = None
            self.prev_loss = None
            self.delta_loss = None
            self.reset = False


class TrustRegionOptimizer(NumericalOptimizer, ABC):
    """
    Numerical optimizer that uses a trust-region strategy.

    Parameters
    ----------
    params : Params
        Parameter tensors.
    trust_region : TrustRegionSolver
        Trust-region solver that computes the step within a region.
    radius_init : float, default=1.0
        Initial trust-region radius.
    accept_tol : float, default=0.1
        Threshold for the ratio ``rho``; if ``rho > accept_tol`` the step is
        accepted.
    curvature_estimator : CurvatureEstimator, optional
        Curvature estimator; Not directly used by the method, kept for compatibility. The
        system will be solved internally by the trust region solver.
    """

    def __init__(
        self,
        params: Params,
        trust_region: TrustRegionSolver,
        radius_init: float = 1.0,
        accept_tol: float = 0.1,
        curvature_estimator: CurvatureEstimator = None,
    ):
        super().__init__(params=params, curvature_estimator=curvature_estimator, lr_init=radius_init)
        self.trust_region = trust_region
        self.accept_tol = accept_tol

    def new_model_radius(
        self,
        objective: ObjectiveFunction,
        radius: float,
        radius_init: float,
        loss: float,
        params: Params,
        grad_params: Params,
        new_loss: float,
        step_dir: Params,
    ):
        """
        Update the trust-region radius based on the ratio rho.

        The ratio rho measures the agreement between the actual reduction and
        the predicted reduction. If rho is small, the model is poor, so shrink
        the radius; if rho is large and the step is at the boundary, expand it.

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function.
        radius : float
            Current trust-region radius.
        radius_init : float
            Initial radius (used as upper bound for expansion).
        loss : torch.Tensor
            Loss value at current parameters.
        params : Params
            Current parameters.
        grad_params : Params
            Gradient at current parameters.
        new_loss : torch.Tensor
            Loss value at proposed new parameters.
        step_dir : Params
            Proposed step direction.

        Returns
        -------
        tuple (rho, new_radius)
            rho : float
                Ratio of actual to predicted reduction.
            new_radius : float
                Updated trust-region radius.
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
        prev_loss = objective.loss(*params)
        model_radius = self.lr_init if self.prev_lr is None else self.prev_lr

        logging.info("Starting trust region loop with radius %g.", model_radius)
        step_dir = self.trust_region.optimize_model(objective, params, model_radius, grad_params)
        if self.fix_ascent and param_dot(grad_params, step_dir) > 1e-8:
            logger.warning("Ascent direction detected, falling back to steepest descent. ")
            step_dir = param_neg(grad_params)
        new_params = param_add(params, step_dir)

        with torch.inference_mode():
            new_loss = objective.loss(*new_params)

        rho, model_radius = self.new_model_radius(objective, model_radius, self.lr_init, prev_loss, params, grad_params, new_loss, step_dir)

        logging.info("Finished trust region search, rho = %g, final model radius = %g.", rho, model_radius)

        numerical_crash = not param_is_finite(new_params) or not math.isfinite(new_loss)
        if not numerical_crash and (self.prev_params is None and (new_loss < prev_loss)) or rho > self.accept_tol:
            # Accept new parameters
            with torch.inference_mode():
                for param, new_param in zip(params, new_params):
                    param.copy_(new_param)

            if self.prev_loss is not None:
                self.delta_loss = new_loss - self.prev_loss
            self.prev_loss = new_loss
            self.prev_params = param_detach(new_params)
            self.prev_step_dir = param_detach(step_dir)
        elif numerical_crash:
            logger.critical("NaN found in loss or parameters. Skipping iteration.")
            self.prev_loss = prev_loss
            self.reset = True
        else:
            logging.info("Parameters were not accepted.")
            self.prev_loss = prev_loss
            self.prev_params = param_detach(params)

        self.prev_grad = param_detach(grad_params)
        self.prev_lr_init = self.prev_lr
        self.prev_lr = torch_to_float(model_radius)

        if self.reset:
            self.prev_lr = None
            self.prev_grad = None
            self.prev_step_dir = None
            self.prev_params = None
            self.prev_loss = None
            self.delta_loss = None
            self.reset = False
