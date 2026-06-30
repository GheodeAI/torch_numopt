from __future__ import annotations
import logging
import torch

from torch_numopt.objective import ObjectiveFunction
from torch_numopt.utils.param_operations import Params, param_detach, param_dot, param_neg, param_norm, param_scaled_add
from torch_numopt.utils.utils import torch_to_float

from ..line_search import create_line_search_solver
from ..numerical_optimizer import NumericalOptimizer, LineSearchOptimizer, TrustRegionOptimizer
from ..curvature import GaussNewtonBlockApproximation

# from ..utils import Params, param_dot, param_scalar_prod, param_norm, param_copy, param_diff
from ..utils import *
from ..trust_region import CauchyPointTRSolver

logger = logging.getLogger(__name__)


class LevenbergMarquardt(TrustRegionOptimizer):
    """
    Heavily inspired by https://github.com/hahnec/torchimize/blob/master/torchimize/optimizer/gna_opt.py
    and the matlab implementation of 'learnlm' https://es.mathworks.com/help/deeplearning/ref/trainlm.html#d126e69092

    Parameters
    ----------

    model: nn.Module
        The model to be optimized
    lr_init: float
        Maximum learning rate in backtracking line search, if the learning rate is set as constant, this will be the value used.
    lr_method: str
        Method to use to initialize the learning rate before applying line search.
    mu: float
        Initial value for the coefficient used when adding a diagonal matrix to the Hessian approximation.
    mu_dec: float
        Factor with which to decrease the coefficient of the diagonal matrix if the previous iteration didn't improve the model.
    mu_max: float
        Factor with which to increase the coefficient of the diagonal matrix if the previous iteration improved the model.
    use_diagonal: bool
        Whether to use the diagonal of the Hessian approximation instead of an identity matrix to adjust the Hessian matrix.
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
    solver: str
        Method to use to invert the hessian.
    batch_size: int
        Size of the amount of data to use at a time to calculate the hessian matrix.
    """

    def __init__(
        self,
        params: Params,
        mu: float = 1e-2,
        mu_dec: float = 0.1,
        mu_max: float = 1e10,
        accept_tol = 0,
        damping: str = "fletcher",
        solver: str = "cholesky",
    ):
        assert damping is not None
        super().__init__(
            params,
            trust_region=CauchyPointTRSolver(curvature_estimator=GaussNewtonBlockApproximation(damping=None)),
            curvature_estimator=GaussNewtonBlockApproximation(damping=damping, mu=mu),
            accept_tol=accept_tol,
        )
        self.solver = solver
        self.mu_dec = mu_dec
        self.mu_max = mu_max

    def new_model_radius(self, objective, radius, radius_init, loss, params, grad_params, new_loss, step_dir):
        """
        Update the model radius from the loss in the current iteration.
        """

        eps = torch.finfo(loss.dtype).eps

        m_0 = self.trust_region.model(objective, 0, params, loss, grad_params)
        m_p = self.trust_region.model(objective, step_dir, params, loss, grad_params)

        rho = (loss - new_loss) / (m_0 - m_p + eps)

        if rho < 0.25:
            radius = min(radius / self.mu_dec, self.mu_max)
        elif rho > 0.75:
            radius *= self.mu_dec

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
        model_radius = self.curvature_estimator.mu

        logging.info("Starting trust region loop with radius %g.", model_radius)
        step_dir = self.get_step_direction(objective, grad_params)
        if self.fix_ascent and param_dot(grad_params, step_dir) > 0:
            logger.warning("Ascent direction detected, falling back to steepest descent. ")
            step_dir = param_neg(grad_params)

        new_params = param_scaled_add(params, step_dir, scale=1)

        with torch.inference_mode():
            new_loss = objective.loss(*new_params)

        rho, model_radius = self.new_model_radius(objective, model_radius, self.lr_init, prev_loss, params, grad_params, new_loss, step_dir)
        self.curvature_estimator.mu = model_radius

        logging.info("Finished trust region search, rho = %g, final model radius = %g.", rho, model_radius)

        if (self.prev_params is None and (new_loss < prev_loss)) or rho > self.accept_tol:
            # Accept new parameters
            with torch.inference_mode():
                for param, new_param in zip(params, new_params):
                    param.copy_(new_param)

            if self.prev_loss is not None:
                self.delta_loss = new_loss - self.prev_loss
            self.prev_loss = new_loss
            self.prev_params = param_detach(new_params)
            self.prev_step_dir = param_detach(step_dir)
            self.prev_grad = param_detach(grad_params)
        else:
            logging.info("Parameters were not accepted.")

        self.prev_lr_init = self.prev_lr
        self.prev_lr = torch_to_float(model_radius)

        if self.reset:
            self.prev_lr = None
            self.prev_grad = None
            self.prev_step_dir = None
            self.prev_params = None
            self.prev_loss = None
            self.delta_loss = None
