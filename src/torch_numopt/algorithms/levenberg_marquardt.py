"""
Levenberg-Marquardt optimizer (trust-region variant).

The Levenberg-Marquardt algorithm interpolates between the Gauss-Newton method
and gradient descent by adaptively adjusting a damping parameter (mu). It is
particularly effective for nonlinear least-squares problems and is implemented
here as a trust-region optimizer with Fletcher's damping strategy.
"""

from __future__ import annotations
import logging
import torch

from ..utils import Params, param_detach, param_dot, param_neg, param_norm, param_scaled_add, torch_to_float
from ..objective import ObjectiveFunction
from ..numerical_optimizer import TrustRegionOptimizer
from ..curvature import GaussNewtonBlockApproximation, GaussNewtonApproximation

from ..trust_region import CauchyPointTRSolver

logger = logging.getLogger(__name__)


class LevenbergMarquardt(TrustRegionOptimizer):
    """
    Levenberg-Marquardt optimizer (trust-region variant).

    This optimizer solves the least-squares problem by adaptively combining
    Gauss-Newton and gradient descent via a damping parameter (mu). The step is
    computed by solving (JᵀJ + mu I) p = -g. The damping is adjusted based on
    the ratio rho.

    Parameters
    ----------
    params : Params
        Parameter tensors.
    mu : float, default=1e-2
        Initial damping parameter.
    mu_dec : float, default=0.1
        Factor by which mu is multiplied when the step is successful (reduction).
    mu_max : float, default=1e10
        Maximum allowed damping.
    accept_tol : float, default=0
        Threshold for rho to accept the step.
    damping : str, default="fletcher"
        Damping strategy for the curvature estimator (e.g., "identity" or "fletcher").
    solver : str, default="cholesky"
        Linear solver for the system.
    """

    def __init__(
        self,
        params: Params,
        mu: float = 1,
        mu_dec: float = 0.1,
        mu_max: float = 1e10,
        accept_tol: float = 0,
        damping: str = "fletcher",
        solver: str = "cholesky",
        block_hessian: bool = True,
    ):
        assert damping is not None, "Levenberg-Marquardt must use a damping strategy."
        if block_hessian:
            curvature_estimator = GaussNewtonBlockApproximation(damping=damping, mu=mu)
        else:
            curvature_estimator = GaussNewtonApproximation(damping=damping, mu=mu)

        super().__init__(
            params,
            trust_region=CauchyPointTRSolver(curvature_estimator=GaussNewtonBlockApproximation(damping=None)),
            curvature_estimator=curvature_estimator,
            accept_tol=accept_tol,
        )
        self.solver = solver
        self.mu_dec = mu_dec
        self.mu_max = mu_max

    def new_model_radius(self, objective, radius, radius_init, loss, params, grad_params, new_loss, step_dir):
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
