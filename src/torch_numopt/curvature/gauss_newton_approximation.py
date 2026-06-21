from __future__ import annotations
from typing import Iterable, Optional
import logging
import torch
from torch import nn
from functools import partial
from torch.func import functional_call
from ..curvature_estimator import CurvatureEstimator
from ..objective import ObjectiveFunction
from ..utils import param_dot, param_scalar_prod, param_add, Params

logger = logging.getLogger(__name__)


class GaussNewtonBlockApproximation(CurvatureEstimator):
    def __init__(
        self,
        vectorize: bool = True,
        damping: Optional[str] = None,
        mu: float = 1e-4,
        uses_blocks: bool = True,
    ):
        super().__init__(ndim=2, uses_blocks=uses_blocks)
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
            logger.info("Computing the Gauss-Newton approximate Hessian matrix.")

            j_params = torch.autograd.functional.jacobian(objective.residual, params, create_graph=False, vectorize=self.vectorize)
            h_params = [None] * len(j_params)
            for idx, j in enumerate(j_params):
                j = j.view(j.shape[0], -1)
                h_params[idx] = self._reshape_hessian(j.T @ j)
        else:
            # Calculate hessian for each batch and add the results
            logger.info("Computing the Gauss-Newton approximate Hessian matrix split in %d batches of size %d.", objective.n_batches, objective.batch_size)

            h_params = []
            for i in range(objective.n_batches):
                # Calculate approximate hessian of the batch
                get_residuals = partial(objective.residual, batch_id=i)
                j_params = torch.autograd.functional.jacobian(get_residuals, params, create_graph=False, vectorize=self.vectorize)
                h_list_batch = [None] * len(j_params)
                for idx, j in enumerate(j_params):
                    j = j.view(j.shape[0], -1)
                    h_list_batch[idx] = self._reshape_hessian(j.T @ j)

                # Aggregate result
                if h_params is None:
                    h_params = h_list_batch
                else:
                    h_params = param_add(h_params, h_list_batch)

                logger.info("Computed batch %d for the approximate hessian...", i)
        
        h_params = list(param_scalar_prod(objective.scale, h_params))

        # Damp matrix
        if self.damping is not None:
            logger.info("Applying damping to the approximate hessian...")
            for i, h in enumerate(h_params):
                if self.damping == "identity":
                    h_params[i] = h + self.mu * torch.eye(h.shape[0], device=h.device)
                elif self.damping == "fletcher":
                    h_params[i] = h + self.mu * torch.diag(h.diagonal())
                else:
                    raise ValueError("Invalid damping strategy.")

        return tuple(h_params)

    def _jvp(self, objective, params, step_dir):
        if not objective.batched:
            logger.info("Computing the Jacobian vector product.")
            _, jac_dot_step = torch.autograd.functional.jvp(objective.residual, params, v=step_dir)
        else:
            # Calculate hessian for each batch and add the results
            logger.info("Computing the Jacobian vector product, split in %d batches of size %d.", objective.n_batches, objective.batch_size)

            jac_dot_step = None
            for i in range(objective.n_batches):
                # Calculate approximate hessian of the batch
                get_residuals = partial(objective.residual, batch_id=i)
                _, jac_dot_step_batch = torch.autograd.functional.jvp(get_residuals, params, v=step_dir)

                if jac_dot_step is None:
                    jac_dot_step = jac_dot_step_batch
                else:
                    jac_dot_step = param_add(jac_dot_step, jac_dot_step_batch)

                logger.info("Computed batch %d for the Jacobian vector product...", i)

        return jac_dot_step

    def hvp(self, objective, params, step_dir):
        if not objective.batched:
            logger.info("Computing the Gauss-Newton Hessian-vector product.")
            residuals, jac_dot_step = torch.autograd.functional.jvp(objective.residual, params, v=step_dir)
            _, hess_approx = torch.autograd.functional.vjp(objective.residual, residuals, v=jac_dot_step)
        else:
            # Calculate hessian for each batch and add the results
            logger.info("Computing the Gauss-Newton Hessian-vector product, split in %d batches of size %d.", objective.n_batches, objective.batch_size)

            hess_approx = None
            for i in range(objective.n_batches):
                # Calculate approximate hessian of the batch
                get_residuals = partial(objective.residual, batch_idx=i)
                residuals, jac_dot_step_batch = torch.autograd.functional.jvp(get_residuals, params, v=step_dir)
                _, hess_approx_batch = torch.autograd.functional.vjp(get_residuals, residuals, v=jac_dot_step_batch)

                if hess_approx is None:
                    hess_approx = hess_approx_batch
                else:
                    hess_approx = param_add(hess_approx, hess_approx_batch)

                logger.info("Computed batch %d for the Gauss-Newton hvp...", i)

        hess_approx = param_scalar_prod(objective.scale, hess_approx)

        # Damp vector
        if self.damping is not None:
            logger.info("Applying damping to the Gauss-Newton hvp...")
            if self.damping == "identity":
                hess_approx = param_add(hess_approx, param_scalar_prod(self.mu, step_dir))
            elif self.damping == "fletcher":
                raise NotImplementedError("Fletcher damping not available for hvp.")
            else:
                raise ValueError(f"Invalid damping strategy {self.damping}.")

        return hess_approx

    def quadratic_form(self, objective, params, d_p_list: Params) -> Params:
        Jp = self._jvp(objective, params, d_p_list)
        quadratic_form = objective.scale * param_dot(Jp, Jp)

        # Damp vector
        if self.damping is not None:
            logger.info("Applying damping to the exact hessian...")
            if self.damping == "identity":
                quadratic_form = quadratic_form + self.mu * param_dot(d_p_list, d_p_list)
            elif self.damping == "fletcher":
                raise NotImplementedError("Fletcher damping not available for hvp.")
            else:
                raise ValueError(f"Invalid damping strategy {self.damping}.")

        return quadratic_form
