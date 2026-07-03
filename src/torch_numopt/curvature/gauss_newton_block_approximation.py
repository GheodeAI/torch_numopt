"""
Block-diagonal Gauss-Newton approximation.

Similar to the full Gauss-Newton, but only the diagonal blocks of Jᵀ J are
computed, one per parameter group. This saves memory and is often sufficient.
"""

from __future__ import annotations
from typing import Optional
import logging
import torch
from functools import partial
from ..curvature_estimator import CurvatureEstimator
from ..objective import ObjectiveFunction
from ..utils import param_dot, param_scalar_prod, param_add, Params

logger = logging.getLogger(__name__)


class GaussNewtonBlockApproximation(CurvatureEstimator):
    """
    Block-diagonal Gauss-Newton Hessian approximation.

    Each block is formed as J_iᵀ J_i, where J_i is the Jacobian of the residual
    with respect to the i-th parameter group. Cross-group derivatives are
    ignored.

    Parameters
    ----------
    vectorize : bool, default=True
        Use vectorized Jacobian computation.
    damping : str or None, default=None
        Damping strategy.
    mu : float, default=1e-4
        Damping coefficient.
    """

    def __init__(
        self,
        vectorize: bool = True,
        damping: Optional[str] = None,
        mu: float = 1e-4,
    ):
        super().__init__(ndim=2, uses_blocks=True)
        self.vectorize = vectorize
        self.damping = damping
        self.mu = mu

    def scaling_matrix(self, objective: ObjectiveFunction, params: Params) -> Params:
        r"""
        Calculation of the an approximate hessian of the Neural network given a dataset as in the Gauss-Newton algorithm.
        The approximate Hessian is calculated as the square of the Jacobian of the residual of every data point with respect to the parameters.

        Let the loss function be, for example the MSE:

        :math:`\mathcal{L}(x,y;\theta) = \sum^{N}_{i=1} (f(x_i; \theta) - y_i)^2 = \sum^{N}_{i=1} r_i`

        Then the Jacobian of the residuals will be the matrix:

        :math:`(J_{\theta}[\mathcal{L}])_{i,j} = \dfrac{\partial r_i}{\partial \theta_j}`

        Then, we will approximate the hessian as the product of the Jacobian with it's transpose, noting that the result
        will be a square matrix with size :math:`p\\times p` with :math:`p` being the number of parameters of the model:

        :math:`H_{\theta}[\mathcal{L}] \approx J_{\theta}[\mathcal{L}]^{\intercal} \cdot J_{\theta}[\mathcal{L}]`

        Parameters
        ----------
        x: torch.Tensor
            Input dataset for calculating the loss.
        y: torch.Tensor
            Target dataset for calculating the loss.
        loss_fn: torch.Module
            Loss function for which to calculate the hessian.
        vectorize: boolean
            Use vectorization in pytorch's implementation of the hessian calculation.
        """

        # Calculate approximate Hessian matrix
        if not objective.batched:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Computing the Gauss-Newton approximate Hessian matrix.")

            h_params = [None] * len(params)
            for i, p in enumerate(params):
                j_params_block = torch.func.jacrev(objective.residual, argnums=i)(*params)
                j_params_block = j_params_block.view(j_params_block.shape[0], -1)
                h_params[i] = j_params_block.T @ j_params_block
        else:
            # Calculate hessian for each batch and add the results
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Computing the Gauss-Newton approximate Hessian matrix split in %d batches of size %d.", objective.n_batches, objective.batch_size
                )

            h_params = None
            for i in range(objective.n_batches):
                # Calculate approximate hessian of the batch
                residual_batch = partial(objective.residual, batch_idx=i)
                h_params_batch = [None] * len(params)
                for i, p in enumerate(params):
                    j_params_block = torch.func.jacrev(residual_batch, argnums=i)(*params)
                    j_params_block = j_params_block.view(j_params_block.shape[0], -1)
                    h_params_batch[i] = j_params_block.T @ j_params_block

                # Aggregate result
                if h_params is None:
                    h_params = h_params_batch
                else:
                    h_params = param_add(h_params, h_params_batch)

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Computed batch %d for the approximate hessian...", i)

        if objective.reduction == "mean":
            h_params = list(param_scalar_prod(2 / objective.data_size, h_params))
        else:
            h_params = list(param_scalar_prod(2, h_params))

        # Damp matrix
        if self.damping is not None:
            for i, h in enumerate(h_params):
                if self.damping == "identity":
                    h_params[i] = h + self.mu * torch.eye(h.shape[0], device=h.device)
                elif self.damping == "fletcher":
                    h_params[i] = h + self.mu * torch.diag(h.diagonal())
                else:
                    raise ValueError("Invalid damping strategy.")

        return tuple(h_params)

    def jvp(self, objective, params, step_dir):
        params = tuple(params)
        step_dir = tuple(step_dir)
        if not objective.batched:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Computing the Jacobian vector product.")

            jac_dot_step = [None] * len(params)
            zero_params = tuple(torch.zeros(p.shape, device=p.device, dtype=p.dtype) for p in params)
            for i, (p, s_d) in enumerate(zip(params, step_dir)):
                tangents = zero_params[:i] + (s_d,) + zero_params[i + 1 :]
                _, j_dot_step_p = torch.func.jvp(objective.residual, params, tuple(tangents))
                jac_dot_step[i] = j_dot_step_p
            jac_dot_step = tuple(jac_dot_step)
        else:
            # Calculate hessian for each batch and add the results
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Computing the Jacobian vector product, split in %d batches of size %d.", objective.n_batches, objective.batch_size)

            jac_dot_step = None
            for i in range(objective.n_batches):
                # Calculate approximate hessian of the batch
                get_residuals = partial(objective.residual, batch_idx=i)
                _, jac_dot_step_batch = torch.autograd.functional.jvp(get_residuals, params, v=step_dir)

                if jac_dot_step is None:
                    jac_dot_step = jac_dot_step_batch
                else:
                    jac_dot_step = param_add(jac_dot_step, jac_dot_step_batch)

                logger.info("Computed batch %d for the Jacobian vector product...", i)

        return jac_dot_step

    def hvp(self, objective, params, step_dir):
        params = tuple(params)
        step_dir = tuple(step_dir)
        if not objective.batched:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Computing the Gauss-Newton Hessian-vector product.")

            hess_dot_step = [None] * len(params)
            zero_params = tuple(torch.zeros(p.shape, device=p.device, dtype=p.dtype) for p in params)
            for i, (p, s_d) in enumerate(zip(params, step_dir)):
                tangents = zero_params[:i] + (s_d,) + zero_params[i + 1 :]
                _, Jv = torch.func.jvp(objective.residual, params, tuple(tangents))
                _, jac_fn = torch.func.vjp(objective.residual, *params)
                hess_dot_step[i] = jac_fn(Jv)[i]
        else:
            # Calculate hessian for each batch and add the results
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Computing the Gauss-Newton Hessian-vector product, split in %d batches of size %d.", objective.n_batches, objective.batch_size
                )

            hess_dot_step = None
            for i in range(objective.n_batches):
                # Calculate approximate hessian of the batch
                hess_approx_batch = [None] * len(params)
                zero_params = tuple(torch.zeros(p.shape, device=p.device, dtype=p.dtype) for p in params)
                for i, (p, s_d) in enumerate(zip(params, step_dir)):
                    tangents = zero_params[:i] + (s_d,) + zero_params[i + 1 :]
                    _, Jv = torch.func.jvp(objective.residual, params, tuple(tangents))
                    _, jac_fn = torch.func.vjp(objective.residual, params)
                    hess_approx_batch[i] = jac_fn(*Jv)
                hess_approx_batch = tuple(hess_approx_batch)

                if hess_dot_step is None:
                    hess_dot_step = hess_approx_batch
                else:
                    hess_dot_step = param_add(hess_dot_step, hess_approx_batch)

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Computed batch %d for the Gauss-Newton hvp...", i)

        if objective.reduction == "mean":
            print(2 / objective.data_size)
            print(hess_dot_step)
            hess_dot_step = param_scalar_prod(2 / objective.data_size, hess_dot_step)
        else:
            hess_dot_step = param_scalar_prod(2, hess_dot_step)

        # Damp vector
        if self.damping is not None:
            if self.damping == "identity":
                hess_dot_step = param_add(hess_dot_step, param_scalar_prod(self.mu, step_dir))
            elif self.damping == "fletcher":
                raise NotImplementedError("Fletcher damping not available for hvp.")
            else:
                raise ValueError(f"Invalid damping strategy {self.damping}.")

        return hess_dot_step

    def quadratic_form(self, objective, params, grad_params: Params) -> Params:
        Jp = self.jvp(objective, params, grad_params)
        quadratic_form = param_dot(Jp, Jp)
        if objective.reduction == "mean":
            quadratic_form = 2 * quadratic_form / objective.data_size
        else:
            quadratic_form = 2 * quadratic_form

        # Damp vector
        if self.damping is not None:
            if self.damping == "identity":
                quadratic_form = quadratic_form + self.mu * param_dot(grad_params, grad_params)
            elif self.damping == "fletcher":
                raise NotImplementedError("Fletcher damping not available for hvp.")
            else:
                raise ValueError(f"Invalid damping strategy {self.damping}.")

        return quadratic_form
