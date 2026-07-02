"""
Gauss-Newton approximation of the Hessian.

For least-squares problems, the Gauss-Newton method approximates the Hessian as
Jᵀ J, where J is the Jacobian of the residuals. This module provides both full
and block-diagonal versions.
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


class GaussNewtonApproximation(CurvatureEstimator):
    """
    Full Gauss-Newton Hessian approximation.

    The matrix is computed as Jᵀ J, where J is the Jacobian of the residual
    vector with respect to the parameters. This estimator forms a single dense
    matrix.

    Parameters
    ----------
    vectorize : bool, default=True
        If ``True``, use vectorized Jacobian computation (may be faster).
    damping : str or None, default=None
        Damping strategy (identity or Fletcher).
    mu : float, default=1e-4
        Damping coefficient.
    """

    def __init__(
        self,
        vectorize: bool = True,
        damping: Optional[str] = None,
        mu: float = 1e-4,
    ):
        super().__init__(ndim=2, uses_blocks=False)
        self.vectorize = vectorize
        self.damping = damping
        self.mu = mu

    def _construct_gauss_newton_matrix(self, j_params: Params, params: Params) -> torch.Tensor:
        n_groups = len(j_params)
        row_blocks = []
        for i in range(n_groups):
            col_blocks = []
            Ji = j_params[i].view(j_params[i].shape[0], -1)
            for j in range(n_groups):
                Jj = j_params[j].view(j_params[j].shape[0], -1)
                col_blocks.append(Ji.T @ Jj)
            row_blocks.append(torch.cat(col_blocks, dim=1))
        full_hessian = torch.cat(row_blocks, dim=0)
        return full_hessian

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

            j_params = torch.autograd.functional.jacobian(objective.residual, params, create_graph=False, vectorize=self.vectorize)
            h_params = self._construct_gauss_newton_matrix(j_params, params)
        else:
            # Calculate hessian for each batch and add the results
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Computing the Gauss-Newton approximate Hessian matrix split in %d batches of size %d.", objective.n_batches, objective.batch_size
                )

            h_params = None
            for i in range(objective.n_batches):
                # Calculate approximate hessian of the batch
                get_residuals = partial(objective.residual, batch_idx=i)
                j_params_batch = torch.autograd.functional.jacobian(get_residuals, params, create_graph=False, vectorize=self.vectorize)
                h_param_batch = self._construct_gauss_newton_matrix(j_params_batch, params)

                # Aggregate result
                if h_params is None:
                    h_params = h_param_batch
                else:
                    h_params = h_params + h_param_batch

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Computed batch %d for the approximate hessian...", i)

        if objective.reduction == "mean":
            h_params = 2 * h_params / objective.data_size

        # Damp matrix
        if self.damping is not None:
            if self.damping == "identity":
                h_params = h_params + self.mu * torch.eye(h_params.shape[0], device=h_params.device)
            elif self.damping == "fletcher":
                h_params = h_params + self.mu * torch.diag(h_params.diagonal())
            else:
                raise ValueError(f"Invalid damping strategy {self.damping}.")

        return h_params

    def jvp(self, objective, params, step_dir):
        if not objective.batched:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Computing the Jacobian vector product.")

            _, jac_dot_step = torch.autograd.functional.jvp(objective.residual, tuple(params), v=tuple(step_dir))
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

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Computed batch %d for the Jacobian vector product...", i)

        return jac_dot_step

    def hvp(self, objective, params, step_dir):
        if not objective.batched:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Computing the Gauss-Newton Hessian-vector product.")
            _, jac_dot_step = torch.autograd.functional.jvp(objective.residual, params, v=step_dir)
            _, hess_approx = torch.autograd.functional.vjp(objective.residual, params, v=jac_dot_step)
        else:
            # Calculate hessian for each batch and add the results
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Computing the Gauss-Newton Hessian-vector product, split in %d batches of size %d.", objective.n_batches, objective.batch_size
                )

            hess_approx = None
            for i in range(objective.n_batches):
                # Calculate approximate hessian of the batch
                get_residuals = partial(objective.residual, batch_idx=i)
                _, jac_dot_step_batch = torch.autograd.functional.jvp(get_residuals, params, v=step_dir)
                _, hess_approx_batch = torch.autograd.functional.vjp(get_residuals, params, v=jac_dot_step_batch)

                if hess_approx is None:
                    hess_approx = hess_approx_batch
                else:
                    hess_approx = param_add(hess_approx, hess_approx_batch)

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Computed batch %d for the Gauss-Newton hvp...", i)

        if objective.reduction == "mean":
            hess_approx = param_scalar_prod(2 / objective.data_size, hess_approx)

        # Damp vector
        if self.damping is not None:
            if self.damping == "identity":
                hess_approx = param_add(hess_approx, param_scalar_prod(self.mu, step_dir))
            elif self.damping == "fletcher":
                raise NotImplementedError("Fletcher damping not available for hvp.")
            else:
                raise ValueError(f"Invalid damping strategy {self.damping}.")

        return hess_approx

    def quadratic_form(self, objective, params, grad_params: Params) -> Params:
        Jp = self.jvp(objective, params, grad_params)
        quadratic_form = param_dot(Jp, Jp)
        if objective.reduction == "mean":
            quadratic_form = quadratic_form * 2 / objective.data_size

        # Damp vector
        if self.damping is not None:
            if self.damping == "identity":
                quadratic_form = quadratic_form + self.mu * param_dot(grad_params, grad_params)
            elif self.damping == "fletcher":
                raise NotImplementedError("Fletcher damping not available for hvp.")
            else:
                raise ValueError(f"Invalid damping strategy {self.damping}.")

        return quadratic_form
